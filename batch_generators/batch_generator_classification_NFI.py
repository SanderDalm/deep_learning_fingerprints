import os
import pickle
from tqdm import tqdm
from scipy.misc import imread, imresize
from skimage.color import rgb2gray
import numpy as np
from glob import glob
from collections import defaultdict


class BatchGenerator_Classification_NFI:

    def __init__(self, path=None, meta_file=None, include_aug=False, height=512, width=512, n_train=20000, detect_special_patterns=False):

        self.path = path
        self.meta_file = meta_file
        self.include_aug = include_aug
        self.height = height
        self.width = width
        self.detect_special_patterns = detect_special_patterns
        if self.detect_special_patterns:
            self.custom_labels = [
                                  'LEFT_COMPOSITE_WHORL',
                                  'LEFT_PLAIN_LOOP',
                                  'PLAIN_ARCH',
                                  'PLAIN_WHORL',
                                  'RIGHT_COMPOSITE_WHORL',
                                  'RIGHT_PLAIN_LOOP'
                                  ]

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
                if self.detect_special_patterns:
                    if label not in self.custom_labels:
                        label = 'SPECIAL'
                    else:
                        label = 'NOT SPECIAL'
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


    def generate_batch(self, batch_size, filenames, include_aug):

        x_batch = []
        y_batch = []

        for _ in range(batch_size):

            filename = np.random.choice(filenames)
            randint = np.random.choice([1, 2, 3, 4, 5, 6, 7, -1])

            if include_aug and randint > 0: # Include augmented samples
                image = np.load(self.path + '/Aug{}/'.format(randint) + filename + '.npy')
                filename = filename.replace('.npy', '')  # strip .npy for label lookup later
            else: # Read original file
                image = imread(self.path+'/BMP/'+filename)
                image = rgb2gray(image)

            if self.height != 512 or self.width != 512:
                image = imresize(image, [self.height, self.width])

            x_batch.append(image)
            y_batch.append(self.label_dict_one_hot[filename])

        return np.array(x_batch).reshape(batch_size, self.height, self.width, 1), np.array(y_batch)


    def generate_train_batch(self, batch_size):

        return self.generate_batch(batch_size, self.filenames_train, self.include_aug)


    def generate_val_batch(self, batch_size):

        return self.generate_batch(batch_size, self.filenames_val, False)