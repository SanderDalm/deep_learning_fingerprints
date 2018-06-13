from tqdm import tqdm
from scipy.misc import imread, imresize
import numpy as np
from glob import glob


class BatchGenerator_Classification:


    def __init__(self, path, imsize):

        self.imsize = imsize
        self.images, self.labels = self.parse_data(path)
        self.images_train, self.labels_train = self.images[:1600], self.labels[:1600]
        self.images_val, self.labels_val = self.images[1600:], self.labels[1600:]


    def parse_data(self, path):


        file_list = glob(path + '/*' + '/*')

        ids = list(set([x[:-4].split('/')[-1] for x in file_list]))
        ids.remove('Thumb')

        images = []
        labels = []

        for id in tqdm(ids):
            image_path = [x for x in file_list if x.find(id) > -1 and x.endswith('png')][0]
            label_path = [x for x in file_list if x.find(id) > -1 and x.endswith('txt')][0]

            img = imread(image_path)
            label_file = open(label_path)
            for line in label_file.readlines():
                if line.startswith('Class'):
                    label = line[7]

            img = img / 255
            images.append(img)
            labels.append(label)

        # Tokenize labels
        tokens = list(range(len(set(labels))))
        self.label_dict = {key: value for key, value in zip(set(labels), tokens)}
        labels_tokenized = [self.label_dict[x] for x in labels]
        n_values = np.max(tokens) + 1
        labels_one_hot = np.eye(n_values)[labels_tokenized]

        return images, labels_one_hot


    def generate_batch(self, batch_size, images, labels):

        x_batch = []
        y_batch = []

        for _ in range(batch_size):

            index = np.random.choice(range(len(images)))
            x_batch.append(images[index])
            y_batch.append(labels[index])

        return np.array(x_batch).reshape(batch_size, self.imsize, self.imsize, 1), np.array(y_batch)


    def generate_train_batch(self, batch_size):

        return self.generate_batch(batch_size, self.images_train, self.labels_train)


    def generate_val_batch(self, batch_size):

        return self.generate_batch(batch_size, self.images_val, self.labels_val)



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
            img = img.reshape([self.imsize, self.imsize, 1])
            img = img / 255
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

                anchor_img += np.random.normal(0, .1, shape=anchor_img.shape)
                pos_img += np.random.normal(0, .1, shape=pos_img.shape)
                neg_img += np.random.normal(0, .1, shape=neg_img.shape)

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

                anchor_img += np.random.normal(0, .1, size=anchor_img.shape)
                partner_img += np.random.normal(0, .1, size=partner_img.shape)

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