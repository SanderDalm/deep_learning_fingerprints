import os
import pickle
from tqdm import tqdm
from scipy.misc import imread, imresize
from skimage.color import rgb2gray
import numpy as np
from glob import glob
from collections import defaultdict


class BatchGenerator_Classification_NFI:

    def __init__(self, path=None, meta_file=None, height=512, width=512, n_train=20000):

        self.path = path
        self.meta_file = meta_file
        self.height = height
        self.width = width

        self.filenames, self.label_dict_one_hot = self.parse_data()
        shuffled_indices = list(range(len(self.filenames)))
        np.random.shuffle(shuffled_indices)
        self.filenames = self.filenames[shuffled_indices]
        self.filenames_train = self.filenames[:n_train]
        self.filenames_val = self.filenames[n_train:]


    def parse_data(self):

        images = glob(self.path + '/BMP/*')

        filenames = [x.split('/')[-1] for x in images]
        labels = defaultdict(int)
        all_labels = []

        with open(self.meta_file) as doc:
            for line in doc.readlines():
                filename = line.split(' ')[0]
                label = line.split(' ')[1]
                labels[filename] = label
                all_labels.append(label)

            # Tokenize labels
            tokens = list(range(len(set(all_labels))))
            n_values = np.max(tokens) + 1
            label_types = sorted(list(set(all_labels)))
            self.label_dict = {key: value for key, value in zip(label_types, tokens)}
            label_dict_one_hot = {label: np.eye(n_values)[token] for label, token in self.label_dict.items()}
            labels_one_hot = {filename: label_dict_one_hot[label] for filename, label in labels.items()}

        return np.array(filenames), labels_one_hot


    def generate_batch(self, batch_size, filenames):

        x_batch = []
        y_batch = []

        for _ in range(batch_size):

            filename = np.random.choice(filenames)
            image = imread(self.path+'/BMP/'+filename)
            image = rgb2gray(image)

            if self.height != 512 or self.width != 512:
                image = imresize(image, [self.height, self.width])

            #image = image / 255
            x_batch.append(image)
            y_batch.append(self.label_dict_one_hot[filename])

        return np.array(x_batch).reshape(batch_size, self.height, self.width, 1), np.array(y_batch)


    def generate_train_batch(self, batch_size):

        return self.generate_batch(batch_size, self.filenames_train)


    def generate_val_batch(self, batch_size):

        return self.generate_batch(batch_size, self.filenames_val)