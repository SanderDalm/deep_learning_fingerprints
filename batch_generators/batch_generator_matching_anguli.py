from scipy.misc import imread, imresize
import numpy as np
from glob import glob

class BatchGenerator_Matching_Anguli:

    def __init__(self, path=None, height=400, width=275, n_train=130000):

        self.path = path
        self.height = height
        self.width = width

        self.ids = self.parse_data()
        shuffled_indices = list(range(len(self.ids)))
        np.random.shuffle(shuffled_indices)
        self.ids = self.ids[shuffled_indices]
        self.ids_train = self.ids[:n_train].tolist()
        self.ids_val = self.ids[n_train:].tolist()

    def parse_data(self):

        images = glob(self.path + 'Impression_1/fp_1/*.jpg')
        ids = list(set([int(x[:-4].split('/')[-1]) for x in images]))
        return np.array(ids)

    def read_image(self, path):

        img = imread(path)
        if self.height != 400 or self.width != 275:
            img = imresize(img, [self.height, self.width])
        img = img.reshape([self.height, self.width, 1])
        return img / 255


    def generate_triplet_batch(self, batch_size, candidate_ids):

        batch = []

        for _ in range(batch_size):

            anchor_id = np.random.choice(candidate_ids)
            candidate_ids_neg = [x for x in candidate_ids if x != anchor_id]
            neg_id = np.random.choice(candidate_ids_neg)

            anchor = self.read_image(self.path+'Impression_1/fp_1/' + str(anchor_id) + '.jpg')
            pos = self.read_image(self.path+'Impression_2/fp_1/' + str(anchor_id) + '.jpg')

            if np.random.rand() < .5:
                anchor, pos = pos, anchor

            if np.random.randn() < .5:
                neg = self.read_image(self.path+'Impression_1/fp_1/' + str(neg_id) + '.jpg')
            else:
                neg = self.read_image(self.path + 'Impression_2/fp_1/' + str(neg_id) + '.jpg')

            triplet = np.concatenate([anchor, pos, neg], axis=2)
            batch.append(triplet)

        return np.array(batch)


    def generate_duo_batch_with_labels(self, batch_size, candidate_ids):

        x_batch = []
        y_batch = []

        for _ in range(batch_size):

            anchor_id = np.random.choice(candidate_ids)

            anchor = self.read_image(self.path+'Impression_1/fp_1/' + str(anchor_id) + '.jpg')
            partner = self.read_image(self.path + 'Impression_2/fp_1/' + str(anchor_id) + '.jpg')

            # pos case
            if np.random.rand() < .5:
                if np.random.rand() < .5:
                    anchor, partner = partner, anchor
                label = 1

            # neg case
            else:
                candidate_ids_neg = [x for x in candidate_ids if x != anchor_id]
                neg_id = np.random.choice(candidate_ids_neg)
                if np.random.rand() < .5:
                    partner = self.read_image(self.path + 'Impression_1/fp_1/' + str(neg_id) + '.jpg')
                else:
                    partner = self.read_image(self.path + 'Impression_2/fp_1/' + str(neg_id) + '.jpg')
                label = 0

            duo = np.concatenate([anchor, partner, partner], axis=2)

            x_batch.append(duo)
            y_batch.append(label)

        return np.array(x_batch), np.array(y_batch).reshape(batch_size, 1)


    def generate_train_triplets(self, batch_size):

        return self.generate_triplet_batch(batch_size, self.ids_train)

    def generate_val_triplets(self, batch_size):

        return self.generate_triplet_batch(batch_size, self.ids_val)

    def generate_train_duos(self, batch_size):

        return self.generate_duo_batch_with_labels(batch_size, self.ids_train)

    def generate_val_duos(self, batch_size):

        return self.generate_duo_batch_with_labels(batch_size, self.ids_val)


#bg = BatchGenerator_Matching_Anguli(path='/home/sander/data/deep_learning_fingerprints/anguli/final/')
#
# len(bg.ids)
# len(bg.ids_train)
# len(bg.ids_val)
#
# x, y = bg.generate_val_duos(32)
#
# import matplotlib.pyplot as plt
# plt.imshow(np.concatenate([x[0, :, :, 0].reshape(400,275), x[0, :, :, 1].reshape(400,275)], axis=0), cmap='gray')
# plt.show()
# y[0]
#
#
#
# batch = bg.generate_val_triplets(32)
#
# import matplotlib.pyplot as plt
# batch.shape
#
# anchor=batch[0, :, :, 0].reshape(400, 275)
# pos=batch[0, :, :, 1].reshape(400, 275)
# neg=batch[0, :, :, 2].reshape(400, 275)
# show = np.concatenate([anchor, pos, neg], axis=0)
# plt.imshow(show, cmap='gray')
# plt.show()