from os.path import join
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

from batch_generators.batch_generator_classification_nist import BatchGenerator_Classification_NIST
from batch_generators.batch_generator_classification_anguli import BatchGenerator_Classification_Anguli
from batch_generators.batch_generator_classification_NFI import BatchGenerator_Classification_NFI
from neural_nets.neural_net_classification import NeuralNet_Classification
from neural_nets.tf_utils import visualize_layers, visualize_embedding
import config

########################################
# Set globals
########################################0

DATAPATH = join(config.datadir, 'sd04/png_txt')
#META_FILE = 'GeneralPatterns.txt'#'CLASSIFICATION-extended pattern set.pet'
HEIGHT = 512
WIDTH = 512
BATCH_SIZE = 32
NUM_STEPS = 2001
DROPOUT = .5
AUGMENT = 1
DECAY = 1

bg = BatchGenerator_Classification_NIST(path=DATAPATH, height=HEIGHT, width=WIDTH, n_train=3000)
#bg = BatchGenerator_Classification_Anguli(path=DATAPATH, height=HEIGHT, width=WIDTH)
#bg = BatchGenerator_Classification_NFI(path=DATAPATH, meta_file=join(DATAPATH, META_FILE), include_aug=True, height=HEIGHT, width=WIDTH, detect_special_patterns=False)

nn = NeuralNet_Classification(HEIGHT, WIDTH, len(bg.label_dict))
#nn.load_weights('models/neural_net1000.ckpt')

loss, val_loss = nn.train(num_steps=NUM_STEPS,
                          batchgen=bg,
                          batch_size=BATCH_SIZE,
                          dropout_rate=DROPOUT,
                          augment=AUGMENT,
                          lr=.0001,
                          decay=DECAY)


plt.plot(loss, color='b', alpha=.7)
plt.plot(val_loss, color='g', alpha=.7)
plt.show()

#plt.plot([np.mean(loss[index:index+30]) for index, value in enumerate(loss)], color='b', alpha=.7)
#plt.plot([np.mean(val_loss[index:index+30]) for index, value in enumerate(val_loss)], color='g', alpha=.7)
#plt.show()

########################################
# Determine acc
########################################

def get_acc(bg, train_val):

    samples = 0
    correct = 0
    for i in range(10):
        if train_val == 'train':
            x, y = bg.generate_train_batch(32)
        if train_val == 'val':
            x, y = bg.generate_val_batch(32)

        for img, label in zip(x, y):
            samples += 1
            pred = nn.predict(img)
            if np.argmax(pred) == np.argmax(label):
                correct += 1

    print('{} acc: {}'.format(train_val, correct/samples))

get_acc(bg, 'train')
get_acc(bg, 'val')

########################################
# Plot embeddings with t-sne.
########################################

def get_label(filename):
    label_path = filename.replace('.png', '.txt')
    label_file = open(label_path)
    for line in label_file.readlines():
        if line.startswith('Class'):
            label = line[7]
    return label

filenames = glob(DATAPATH + '/*' + '/*.png')
filenames = [x for x in filenames if 'umb' not in x]
labels = [get_label(x) for x in filenames]

visualize_embedding(nn, bg, filenames, labels, limit=2000)

########################################
# Visualize convolutional layers
########################################

layers = [op for op in nn.session.graph.get_operations() if op.type == 'Conv2D']
input_img = np.random.uniform(size=(1, HEIGHT, WIDTH, 1))
visualize_layers(nn=nn, layer_number=5, input_img=input_img)