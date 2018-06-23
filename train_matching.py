import numpy as np
import matplotlib.pyplot as plt

from neural_nets.neural_net_matching import NeuralNet_Matching
from batch_generators.batch_generator_matching_nist import BatchGenerator_Matching_NIST

########################################
# Set globals
########################################

path = '/home/sander/data/deep_learning_fingerprints/sd04/png_txt'##'/mnt/ssd/data/deep_learning_fingerprints/sd04/png_txt'
IMSIZE = 512
BATCH_SIZE = 16
NUM_STEPS = 2001


########################################
# Train model
########################################

bg = BatchGenerator_Matching_NIST(path=path, imsize=IMSIZE)

# Check batch gen output
for i in range(5):
    x, y = bg.generate_train_duos(1, True)
    index = 0
    y[index]
    plt.imshow(np.concatenate([x[index, :, :, 0], x[index, :, :, 1]], axis=0), cmap='gray')
    if y == 1:
        plt.savefig('same_source_{}.png'.format(i))
    else:
        plt.savefig('different_source_{}.png'.format(i))
    plt.clf()
    #plt.show()
    #print(y[0])


nn = NeuralNet_Matching(imsize=IMSIZE, batchgen=bg, network_type='duos')

# Record: conv/conv/dropout/pool architectuur, .5 dropout, augment false, lr.0001, decay 1, 900 stappen, 93% acc, 'models/neural_net899.ckpt'

loss, val_loss = nn.train(num_steps=NUM_STEPS,
         batch_size=BATCH_SIZE,
         dropout_rate=0.5,
         augment=False,
         lr=.0001,
         decay=.998)


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
    batch = bg.generate_val_triplets(32)

    for triplet in batch:

        anchor = triplet[:, :, 0].reshape([IMSIZE, IMSIZE, 1])
        pos = triplet[:, :, 1].reshape([IMSIZE, IMSIZE, 1])
        neg = triplet[:, :, 2].reshape([IMSIZE, IMSIZE, 1])

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

samples = 0
correct = 0

pos_preds = []
neg_preds = []

for i in range(15):
    x, y = bg.generate_val_duos(32)
    for img, label in zip(x, y):
        samples += 1
        pred, _ = nn.predict(img[:, :, 0].reshape(IMSIZE, IMSIZE, 1), img[:, :, 1].reshape(IMSIZE, IMSIZE, 1))

        if np.round(pred, 0) == label:
            correct += 1

        if label == 0:
            neg_preds.append(pred)
        if label == 1:
            pos_preds.append(pred)
print(correct/samples)

plt.hist(pos_preds, color='g', alpha=.4)
plt.hist(neg_preds, color='r', alpha=.4)
plt.show()