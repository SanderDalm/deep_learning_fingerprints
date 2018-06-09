from tqdm import tqdm
from scipy.misc import imread, imresize
import numpy as np
from glob import glob


class BatchGenerator_Classification:


    def __init__(self, path, imsize):

        self.imsize = imsize
        self.images, self.labels = self.parse_data(path)
        self.cursor = 0


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

            images.append(img)
            labels.append(label)

        # Tokenize labels
        tokens = list(range(len(set(labels))))
        self.label_dict = {key: value for key, value in zip(set(labels), tokens)}
        labels_tokenized = [self.label_dict[x] for x in labels]
        n_values = np.max(tokens) + 1
        labels_one_hot = np.eye(n_values)[labels_tokenized]

        return images, labels_one_hot


    def generate_batch(self, batch_size):

        x_batch = []
        y_batch = []

        for _ in range(batch_size):

            x_batch.append(self.images[self.cursor])
            y_batch.append(self.labels[self.cursor])
            self.cursor += 1
            if self.cursor == len(self.images):
                indices = list(range(len(self.images)))
                np.random.shuffle(indices)
                self.images = [self.images[x] for x in indices]
                self.labels = [self.labels[x] for x in indices]
                self.cursor = 0
                print("Every day I'm shuffling")

        return np.array(x_batch), np.array(y_batch)


class BatchGenerator_Matching:

    def __init__(self, path, imsize):


        self.imsize = imsize
        self.images, self.ids = self.parse_data(path)
        self.sample_ids = list(set([x[1:] for x in self.ids]))

        self.sample_ids_train = self.sample_ids[:1600]
        self.sample_ids_val = self.sample_ids[1600:]

        self.cursor = 0

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

    def generate_triplet_batch(self, batch_size):

        batch = []

        for _ in range(batch_size):

            selected_id = self.sample_ids[self.cursor]
            if np.random.rand() < .5:
                anchor_index = self.ids.index('f' + selected_id)
                pos_index = self.ids.index('s' + selected_id)
            else:
                anchor_index = self.ids.index('s' + selected_id)
                pos_index = self.ids.index('f' + selected_id)

            neg_candidate_indices = list(range(len(self.ids)))
            neg_candidate_indices.remove(anchor_index)
            neg_candidate_indices.remove(pos_index)
            neg_index = np.random.choice(neg_candidate_indices)

            triplet = np.concatenate([self.images[anchor_index], self.images[pos_index], self.images[neg_index]], axis=2)
            batch.append(triplet)

            self.cursor += 1
            if self.cursor == len(self.sample_ids):
                indices = list(range(len(self.sample_ids)))
                np.random.shuffle(indices)
                self.sample_ids = [self.sample_ids[x] for x in indices]
                self.cursor = 0
                print("Every day I'm shuffling")

        return np.array(batch)


    def generate_triplet_batch_validation(self, limit=-1):

        batch = []

        for id in self.sample_ids_val[:limit]:

            if np.random.rand() < .5:
                anchor_index = self.ids.index('f' + id)
                pos_index = self.ids.index('s' + id)
            else:
                anchor_index = self.ids.index('s' + id)
                pos_index = self.ids.index('f' + id)

            neg_candidate_indices = list(range(len(self.ids)))
            neg_candidate_indices.remove(anchor_index)
            neg_candidate_indices.remove(pos_index)
            neg_index = np.random.choice(neg_candidate_indices)

            triplet = np.concatenate([self.images[anchor_index], self.images[pos_index], self.images[neg_index]],
                                     axis=2)
            batch.append(triplet)


        return np.array(batch)