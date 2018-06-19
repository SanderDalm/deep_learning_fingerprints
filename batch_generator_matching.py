from tqdm import tqdm
from scipy.misc import imread, imresize
import numpy as np
from glob import glob
from skimage.filters import gabor
from skimage.morphology import skeletonize
from skimage.util import invert
from skimage.filters import threshold_otsu

class BatchGenerator_Matching:

    def __init__(self, path, imsize):


        self.imsize = imsize
        self.images, self.ids = self.parse_data(path)
        self.sample_ids = list(set([x[1:] for x in self.ids]))

        self.sample_ids_train = self.sample_ids[:1600]
        self.sample_ids_val = self.sample_ids[1600:]

    def parse_data(self, path):


        file_list = glob(path+'/*'+'/*')

        ids = list(set([x[:-4].split('/')[-1] for x in file_list]))
        ids.remove('Thumb')

        images = []

        for id in tqdm(ids):
            image_path = [x for x in file_list if x.find(id) > -1 and x.endswith('png')][0]

            img = imread(image_path)

            img = img / 255

            #filt_real, img = gabor(img, frequency=0.6)

            # thresh = threshold_otsu(img)
            # img = img > thresh
            # img = invert(img)
            # img = skeletonize(img)

            img = img.reshape([self.imsize, self.imsize, 1])
            images.append(img)

        return images, ids


    def generate_triplet_batch(self, batch_size, candidate_ids, augment):

        batch = []

        for _ in range(batch_size):

            anchor_id = np.random.choice(candidate_ids)
            anchor_index = self.ids.index('f' + anchor_id)
            pos_index = self.ids.index('s' + anchor_id)
            if np.random.rand() < .5:
                anchor_index, pos_index = pos_index, anchor_index

            neg_candidate_ids = ['f' + x for x in candidate_ids if x != anchor_id]+['s' + x for x in candidate_ids if x != anchor_id]
            neg_id = np.random.choice(neg_candidate_ids)
            neg_index = self.ids.index(neg_id)

            anchor_img, pos_img, neg_img = self.images[anchor_index], self.images[pos_index], self.images[neg_index]

            if augment:
                rotation = np.random.randint(0, 4)
                anchor_img, pos_img, neg_img = np.rot90(anchor_img, rotation), np.rot90(pos_img, rotation), np.rot90(neg_img, rotation)

                if np.random.rand() > .5:
                    anchor_img, pos_img, neg_img = np.fliplr(anchor_img), np.fliplr(pos_img), np.fliplr(neg_img)

                # anchor_img += np.random.normal(0, .1, size=anchor_img.shape)
                # pos_img += np.random.normal(0, .1, size=pos_img.shape)
                # neg_img += np.random.normal(0, .1, size=neg_img.shape)

            triplet = np.concatenate([anchor_img, pos_img, neg_img], axis=2)
            batch.append(triplet)

        return np.array(batch)


    def generate_duo_batch_with_labels(self, batch_size, candidate_ids, augment):

        x_batch = []
        y_batch = []

        for _ in range(batch_size):

            anchor_id = np.random.choice(candidate_ids)

            anchor_index = self.ids.index('f' + anchor_id)
            partner_index = self.ids.index('s' + anchor_id)

            # pos case
            if np.random.rand() < .5:
                if np.random.rand() < .5:
                    anchor_index, partner_index = partner_index, anchor_index
                label = 1

            # neg case
            else:
                neg_candidate_ids = ['f' + x for x in candidate_ids if x != anchor_id] + ['s' + x for x in
                                                                                          candidate_ids if
                                                                                                x != anchor_id]
                neg_id = np.random.choice(neg_candidate_ids)
                partner_index = self.ids.index(neg_id)
                label = 0


            anchor_img = self.images[anchor_index]
            partner_img = self.images[partner_index]

            if augment:
                rotation = np.random.randint(0, 4)
                anchor_img, partner_img = np.rot90(anchor_img, rotation), np.rot90(partner_img, rotation)

                if np.random.rand() > .5:
                    anchor_img, partner_img = np.fliplr(anchor_img), np.fliplr(partner_img)

                #anchor_img += np.random.normal(0, .1, size=anchor_img.shape)
                #partner_img += np.random.normal(0, .1, size=partner_img.shape)

            duo = np.concatenate([anchor_img, partner_img, partner_img], axis=2)

            x_batch.append(duo)
            y_batch.append(label)

        return np.array(x_batch), np.array(y_batch).reshape(batch_size, 1)


    def generate_train_triplets(self, batch_size, augment=True):

        return self.generate_triplet_batch(batch_size, self.sample_ids_train, augment)

    def generate_val_triplets(self, batch_size, augment=False):

        return self.generate_triplet_batch(batch_size, self.sample_ids_val, augment)

    def generate_train_duos(self, batch_size, augment=True):

        return self.generate_duo_batch_with_labels(batch_size, self.sample_ids_train, augment)

    def generate_val_duos(self, batch_size, augment=False):

        return self.generate_duo_batch_with_labels(batch_size, self.sample_ids_val, augment)