from os.path import join

import numpy as np
import matplotlib.pyplot as plt

from batch_generators.batch_generator_classification_nist import BatchGenerator_Classification_NIST
from batch_generators.batch_generator_classification_anguli import BatchGenerator_Classification_Anguli
from batch_generators.batch_generator_classification_NFI import BatchGenerator_Classification_NFI
from neural_nets.neural_net_classification import NeuralNet_Classification
import config

########################################
# Set globals
########################################0

DATAPATH = join(config.datadir, 'NFI')
META_FILE = 'GeneralPatterns.txt'#'CLASSIFICATION-extended pattern set.pet'
HEIGHT = 512
WIDTH = 512
BATCH_SIZE = 128
NUM_STEPS = 6001
CATEGORIES = 7

#bg_anguli = BatchGenerator_Classification_Anguli(path=DATAPATH, height=HEIGHT, width=WIDTH)
#bg_nist = BatchGenerator_Classification_NIST(path=DATAPATH, height=HEIGHT, width=WIDTH)
bg_NFI = BatchGenerator_Classification_NFI(path=DATAPATH, meta_file=join(DATAPATH, META_FILE), include_aug=True, height=HEIGHT, width=WIDTH)

nn = NeuralNet_Classification(HEIGHT, WIDTH, CATEGORIES)

loss, val_loss = nn.train(num_steps=NUM_STEPS,
                          batchgen=bg_NFI,
                          batch_size=BATCH_SIZE,
                          dropout_rate=0.5,
                          augment=1,
                          lr=.0001,
                          decay=1)

#nn.load_weights('models/neural_net2199.ckpt')

plt.plot(loss, color='b', alpha=.7)
plt.plot(val_loss, color='g', alpha=.7)
plt.show()

########################################
# Determine acc
########################################

def get_acc(bg, train_val):

    samples = 0
    correct = 0
    for i in range(100):
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

print('NFI')
get_acc(bg_NFI, 'train')
get_acc(bg_NFI, 'val')
#print('')
#print('Anguli')
#get_acc(bg_anguli, 'train')
#get_acc(bg_anguli, 'val')