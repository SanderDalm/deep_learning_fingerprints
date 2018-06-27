from tqdm import tqdm
from scipy.misc import imread, imresize
import numpy as np
from glob import glob
from skimage.filters import gabor
from skimage.morphology import skeletonize
from skimage.util import invert
from skimage.filters import threshold_otsu

class BatchGenerator_Matching_Anguli:

    def __init__(self, path='/media/sander/Data/fingerprints', height=400, width=275):


        self.height = height
        self.width = width
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

            img = img.reshape([self.heigth, self.width, 1])
            images.append(img)

        return images, ids


    def generate_triplet_batch(self, batch_size, candidate_ids):

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

            triplet = np.concatenate([anchor_img, pos_img, neg_img], axis=2)
            batch.append(triplet)

        return np.array(batch)


    def generate_duo_batch_with_labels(self, batch_size, candidate_ids):

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