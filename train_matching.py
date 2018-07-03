import numpy as np
import matplotlib.pyplot as plt

from neural_nets.neural_net_matching import NeuralNet_Matching
from batch_generators.batch_generator_matching_nist import BatchGenerator_Matching_NIST
from batch_generators.batch_generator_matching_anguli import BatchGenerator_Matching_Anguli
import config

########################################
# Set globals
########################################

HEIGHT = 400
WIDTH = 275
BATCH_SIZE = 32
NUM_STEPS = 3001


########################################
# Train model
########################################

#bg_anguli = BatchGenerator_Matching_Anguli(path=config.datadir+'/anguli/final/', height=HEIGHT, width=WIDTH)
bg_nist = BatchGenerator_Matching_NIST(path=config.datadir+'/sd04/png_txt/', height=HEIGHT, width=WIDTH)

nn = NeuralNet_Matching(height=HEIGHT, width=WIDTH, network_type='duos')

# Record: conv/conv/dropout/pool architectuur, .5 dropout, augment false, lr.0001, decay 1, 900 stappen, 93% acc, 'models/neural_net899.ckpt'

loss, val_loss = nn.train(num_steps=NUM_STEPS,
                          batchgen=bg_nist,
                          batch_size=BATCH_SIZE,
                          dropout_rate=0.5,
                          augment=1,
                          lr=.0001,
                          decay=1)


plt.plot(loss, color='b', alpha=.7)
plt.plot(val_loss, color='g', alpha=.7)
plt.show()

nn.load_weights('models/neural_net899.ckpt')

########################################
# Visualize embedding distances
########################################

distances_same = []
distances_diff = []

for _ in range(15):
    batch = bg_nist.generate_val_triplets(32)

    for triplet in batch:

        anchor = triplet[:, :, 0].reshape([HEIGHT, WIDTH, 1])
        pos = triplet[:, :, 1].reshape([HEIGHT, WIDTH, 1])
        neg = triplet[:, :, 2].reshape([HEIGHT, WIDTH, 1])

        embedding_distance_same = nn.predict(image1=anchor,
                                             image2=pos)[1]
        embedding_distance_diff = nn.predict(image1=anchor,
                                             image2=neg)[1]
        distances_same.append(embedding_distance_same)
        distances_diff.append(embedding_distance_diff)

plt.hist(distances_same, color='g', alpha=.4)
plt.hist(distances_diff, color='r', alpha=.4)
plt.show()

# LRs
hist_pos, bins = np.histogram(distances_same)
hist_neg = np.histogram(distances_diff, bins=bins)[0]

for index, value in enumerate(hist_pos):
    print(hist_neg[index]/value)


########################################
# Determine acc for threshold
########################################

# threshold = 18
#
# def match(image1, image2, threshold):
#
#     distance = nn.compute_embedding_distance(image1=image1,
#                                   image2=image2)
#     if distance < threshold:
#         return 1
#     else:
#         return 0
#
#
# sensitivity = np.mean(matches_pos)
# specificity = 1-np.mean(matches_neg)
#
# print((sensitivity+specificity)/2)

########################################
# Determine acc for duo network
########################################

def get_acc(bg, train_val):

    samples = 0
    correct = 0

    pos_preds = []
    neg_preds = []

    for i in range(15):
        if train_val == 'train':
            x, y = bg.generate_train_duos(32)
        if train_val == 'val':
            x, y = bg.generate_val_duos(32)
        for img, label in zip(x, y):
            samples += 1
            pred, _ = nn.predict(img[:, :, 0].reshape(HEIGHT, WIDTH, 1), img[:, :, 1].reshape(HEIGHT, WIDTH, 1))

            if np.round(pred, 0) == label:
                correct += 1

            if label == 0:
                neg_preds.append(pred)
            if label == 1:
                pos_preds.append(pred)

    print('{} acc: {}'.format(train_val, correct / samples))
    plt.hist(pos_preds, color='g', alpha=.4)
    plt.hist(neg_preds, color='r', alpha=.4)
    plt.show()
    plt.clf()


print('NIST')
get_acc(bg_nist, 'train')
get_acc(bg_nist, 'val')
#print('')
#print('Anguli')
#et_acc(bg_anguli, 'train')
#et_acc(bg_anguli, 'val')