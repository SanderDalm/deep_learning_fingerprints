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

            triplet = np.concatenate([self.images[anchor_index], self.images[pos_index], self.images[neg_index]], axis=2)
            batch.append(triplet)

        return np.array(batch)


    def generate_duo_batch_with_labels(self, batch_size, candidate_ids):

        x_batch = []
        y_batch = []

        for _ in range(batch_size):

            anchor_id = np.random.choice(candidate_ids)

            anchor_index_f = self.ids.index('f' + anchor_id)
            anchor_index_s = self.ids.index('s' + anchor_id)

            randnum = np.random.rand()
            if randnum < .5:
                anchor_letter = 'f'
                anchor_index = anchor_index_f
            else:
                anchor_letter = 's'
                anchor_index = anchor_index_s


            if np.random.rand() < .5:
                # pos case
                if anchor_letter == 'f':
                    partner_index = anchor_index_s
                if anchor_letter == 's':
                    partner_index = anchor_index_f
                label = 1

            else:
                # neg case
                neg_candidate_ids = ['f' + x for x in candidate_ids if x != anchor_id] + ['s' + x for x in
                                                                                          candidate_ids if
                                                                                                x != anchor_id]
                neg_id = np.random.choice(neg_candidate_ids)
                partner_index = self.ids.index(neg_id)
                label = 0


            duo = np.concatenate([self.images[anchor_index], self.images[partner_index], self.images[partner_index]],
                                     axis=2)

            x_batch.append(duo)
            y_batch.append(label)

        return np.array(x_batch), np.array(y_batch).reshape(batch_size, 1)


    def generate_train_triplets(self, batch_size):

        return self.generate_triplet_batch(batch_size, self.sample_ids_train)

    def generate_val_triplets(self, batch_size):

        return self.generate_triplet_batch(batch_size, self.sample_ids_val)

    def generate_train_duos(self, batch_size):

        return self.generate_duo_batch_with_labels(batch_size, self.sample_ids_train)

    def generate_val_duos(self, batch_size):

        return self.generate_duo_batch_with_labels(batch_size, self.sample_ids_val)