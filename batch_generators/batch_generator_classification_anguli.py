import os
import pickle
from tqdm import tqdm
from scipy.misc import imread, imresize
import numpy as np
from glob import glob


class BatchGenerator_Classification_Anguli:


    def __init__(self, path='/home/sander/data/deep_learning_fingerprints/anguli/final/', height=400, width=275, n_train=130000):

        self.path = path
        self.height = height
        self.width = width

        self.image_ids, self.labels = self.parse_data()
        shuffled_indices = list(range(len(self.image_ids)))
        np.random.shuffle(shuffled_indices)
        self.image_ids, self.labels = self.image_ids[shuffled_indices], self.labels[shuffled_indices]
        self.image_ids_train, self.labels_train = self.image_ids[:n_train], self.labels[:n_train]
        self.image_ids_val, self.labels_val = self.image_ids[n_train:], self.labels[n_train:]


    def parse_data(self):

        meta_path = self.path + 'Meta Info/fp_1/'
        images = glob(self.path + 'Impression_*/fp_1/*')
        meta_files = glob(self.path + 'Meta Info/fp_1/*')

        image_ids = list(set([int(x[:-4].split('/')[-1]) for x in images]))
        meta_ids = list([int(x[:-4].split('/')[-1]) for x in meta_files])
        image_ids.sort()
        meta_ids.sort()


        if os.path.exists('ids_and_labels.p'):
            image_ids, labels_one_hot = pickle.load(open('ids_and_labels.p', 'rb'))

        else:
            labels = []
            for meta_id in tqdm(meta_ids):

                meta_file = meta_path + str(meta_id) + '.txt'

                with open(meta_file) as doc:
                    for line in doc.readlines():
                        if line.startswith('Type'):
                            label = line[7:].strip('\n')
                            if label == 'Double Loop':
                                image_ids.remove(meta_id)
                                continue
                            else:
                                labels.append(label)
            # Tokenize labels
            tokens = list(range(len(set(labels))))
            label_types = sorted(list(set(labels)))
            self.label_dict = {key: value for key, value in zip(label_types, tokens)}
            labels_tokenized = [self.label_dict[x] for x in labels]
            n_values = np.max(tokens) + 1
            labels_one_hot = np.eye(n_values)[labels_tokenized]
            pickle.dump((image_ids, labels_one_hot), open('ids_and_labels.p', 'wb'))

        return np.array(image_ids), labels_one_hot


    def generate_batch(self, batch_size, image_ids, labels):

        x_batch = []
        y_batch = []

        for _ in range(batch_size):

            index = np.random.choice(range(len(image_ids)))
            if np.random.rand() < .5:
                image = imread(self.path+'Impression_1/fp_1/'+str(image_ids[index])+'.jpg')
            else:
                image = imread(self.path+'Impression_2/fp_1/' + str(image_ids[index]) + '.jpg')

            if self.height != 400 or self.width != 275:
                image = imresize(image, [self.height, self.width])
            image = image / 255
            x_batch.append(image)
            y_batch.append(labels[index])

        return np.array(x_batch).reshape(batch_size, self.height, self.width, 1), np.array(y_batch)


    def generate_train_batch(self, batch_size):

        return self.generate_batch(batch_size, self.image_ids_train, self.labels_train)


    def generate_val_batch(self, batch_size):

        return self.generate_batch(batch_size, self.image_ids_val, self.labels_val)



# bg = BatchGenerator_Classification_Anguli()
#
# len(bg.image_ids_train)
# len(bg.labels_train)
# len(bg.image_ids_val)
# len(bg.labels_val)


# bg.label_dict
#
# for i in range(5):
#     print(np.mean(bg.labels[:, i]))
#
#
# import matplotlib.pyplot as plt
#x, y = bg.generate_val_batch(32)


# index = 20
# plt.imshow(x[index].reshape(400, 275), cmap='gray')
# plt.show()
# print(y[index])
#
# for index, label in enumerate(y):
#     print(index, label)


# a = np.array([1,2,3,4])
# indices = a != 3
# a[indices]