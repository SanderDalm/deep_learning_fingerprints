from tqdm import tqdm
from scipy.misc import imread, imresize
import numpy as np
from glob import glob


class BatchGenerator_Classification_NIST:


    def __init__(self, path=None, height=512, width=512, n_train=3800):

        self.height = height
        self.width = width

        self.images, self.labels = self.parse_data(path)
        shuffled_indices = list(range(len(self.images)))
        np.random.shuffle(shuffled_indices)
        self.images, self.labels = self.images[shuffled_indices], self.labels[shuffled_indices]
        self.images_train, self.labels_train = self.images[:n_train], self.labels[:n_train]
        self.images_val, self.labels_val = self.images[n_train:], self.labels[n_train:]


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
            if self.height != 512 or self.width != 512:
                img = imresize(img, [self.height, self.width])
            label_file = open(label_path)
            for line in label_file.readlines():
                if line.startswith('Class'):
                    label = line[7]

            img = img / 255
            images.append(img)
            labels.append(label)

        # Tokenize labels
        tokens = list(range(len(set(labels))))
        label_types = sorted(list(set(labels)))
        self.label_dict = {key: value for key, value in zip(label_types, tokens)}
        labels_tokenized = [self.label_dict[x] for x in labels]
        n_values = np.max(tokens) + 1
        labels_one_hot = np.eye(n_values)[labels_tokenized]

        return np.array(images), labels_one_hot


    def generate_batch(self, batch_size, images, labels):

        x_batch = []
        y_batch = []

        for _ in range(batch_size):

            index = np.random.choice(range(len(images)))
            x_batch.append(images[index])
            y_batch.append(labels[index])

        return np.array(x_batch).reshape(batch_size, self.height, self.width, 1), np.array(y_batch)


    def generate_train_batch(self, batch_size):

        return self.generate_batch(batch_size, self.images_train, self.labels_train)


    def generate_val_batch(self, batch_size):

        return self.generate_batch(batch_size, self.images_val, self.labels_val)

# bg = BatchGenerator_Classification_NIST(height=400, width=275)
#
# import matplotlib.pyplot as plt
# x, y = bg.generate_val_batch(32)
# plt.imshow(x[0].reshape(400, 275), cmap='gray')
# plt.show()



# bg.label_dict
#
# for i in range(5):
#     print(np.mean(bg.labels[:, i]))