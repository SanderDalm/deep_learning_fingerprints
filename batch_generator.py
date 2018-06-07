from scipy.misc import imread, imresize
import numpy as np
from glob import glob

class BatchGenerator(object):

    def __init__(self, path, task, imsize):


        self.imsize = imsize
        if task == 'classification':
            self.images, self.labels = self.parse_data_for_classification(path)
        if task == 'matching':
            self.parse_data_for_matching(path)


    def parse_data_for_classification(self, path):


        file_list = glob(path+'/*'+'/*')

        ids = list(set([x[:-4].split('/')[-1] for x in file_list]))
        ids.remove('Thumb')

        images = []
        labels = []

        for id in ids:
            print(id)
            image_path = [x for x in file_list if x.find(id) > -1 and x.endswith('png')][0]
            label_path = [x for x in file_list if x.find(id) > -1 and x.endswith('txt')][0]

            img = imread(image_path)
            label_file = open(label_path)
            for line in label_file.readlines():
                if line.startswith('Class'):
                    label = line[7]

            images.append(img)
            labels.append(label)

        return images, labels


    def parse_data_for_matching(self):
        pass

    def generate_batch(self, batch_size):

        x_batch = []
        y_batch = []

        for _ in range(batch_size):

            filename = np.random.choice(self.file_list)
            img = imread(filename)
            img = imresize(img, [self.imsize, self.imsize])
            img = img.reshape([self.imsize, self.imsize, 3])
            img = img.astype(np.float32)
            img = img/255.0

            if 'dog' in filename:
                y_batch.append([1, 0])
            else:
                y_batch.append([0, 1])

            x_batch.append(img)

        return np.array(x_batch), np.array(y_batch).reshape([batch_size, 2])


    def create_fake_data(self):

        y_pos = np.ones((DATA_SIZE, 1))
        y_neg = np.zeros((DATA_SIZE, 1))

        def get_sequence(change):

            if change:
                x = np.linspace(0, 5, SEQUENCE_LENGTH) + np.random.normal(0, 5, [SEQUENCE_LENGTH])
            else:
                x = np.linspace(5, 0, SEQUENCE_LENGTH) + np.random.normal(0, 5, [SEQUENCE_LENGTH])

            return x.reshape([x.shape[0], 1])

        x_pos_list = []
        x_neg_list = []

        for i in range(DATA_SIZE):

            randnum = np.random.random()

            if randnum < .5:
                x_pos = np.concatenate([get_sequence(True), get_sequence(True)], axis=1)
                x_neg = np.concatenate([get_sequence(True), get_sequence(False)], axis=1)
            else:
                x_pos = np.concatenate([get_sequence(False), get_sequence(False)], axis=1)
                x_neg = np.concatenate([get_sequence(False), get_sequence(True)], axis=1)

            x_pos_list.append(x_pos)
            x_neg_list.append(x_neg)

        x_pos = np.concatenate([x_pos_list])
        x_neg = np.concatenate([x_neg_list])

        x_pos_train = x_pos[:80000]
        x_pos_test = x_pos[80000:]
        x_neg_train = x_neg[:80000]
        x_neg_test = x_neg[80000:]

        y_pos_train = y_pos[:80000]
        y_pos_test = y_pos[80000:]
        y_neg_train = y_neg[:80000]
        y_neg_test = y_neg[80000:]


        def generate_batch(x_pos, y_pos, x_neg, y_neg):

            sample = np.random.choice(range(len(x_pos)), BATCH_SIZE//2)

            batch_x = np.concatenate([x_pos[sample], x_neg[sample]], axis=0)
            batch_y = np.concatenate([y_pos[sample], y_neg[sample]], axis=0)

            return batch_x, batch_y

        plt.plot(x_pos[1])
        plt.savefig('pos.png')
        plt.clf()
        plt.plot(x_neg[1])
        plt.savefig('neg.png')
        plt.clf()