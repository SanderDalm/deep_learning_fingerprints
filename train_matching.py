import numpy as np
import matplotlib.pyplot as plt

from neural_net_matching import NeuralNet_Matching
from batch_generators import BatchGenerator_Matching

########################################
# Set globals
########################################

path = '/home/sander/data/deep_learning_fingerprints/sd04/png_txt'
IMSIZE = 512
BATCH_SIZE = 32
NUM_STEPS = 2001


########################################
# Train model
########################################

bg = BatchGenerator_Matching(path=path, imsize=IMSIZE)


from time import time

start = time()
for i in range(100):
    bg.generate_val_triplets(32, False)
print(time()-start)


start = time()
for i in range(100):
    bg.generate_val_triplets(32, True)
print(time()-start)

# Check batch gen output
# x, y = bg.generate_train_duos(1, True)
# index=0
# y[index]
# plt.imshow(np.concatenate([x[index, :, :, 0], x[index, :, :, 1]], axis=0))
# plt.show()
# print(y[0])


nn = NeuralNet_Matching(imsize=IMSIZE, batchgen=bg, network_type='duos')

loss, val_loss = nn.train(num_steps=NUM_STEPS,
         batch_size=BATCH_SIZE,
         dropout_rate=0.5,
         augment=True,
         lr=.0001,
         decay=1)


plt.plot(loss, color='b', alpha=.7)
plt.plot(val_loss, color='g', alpha=.7)
plt.show()

########################################
# Determine matching threshold
########################################

distances_same = []
distances_diff = []

for _ in range(150):
    batch = bg.generate_val_triplets(32)

    for triplet in batch:

        anchor = triplet[:, :, 0].reshape([IMSIZE, IMSIZE, 1])
        pos = triplet[:, :, 1].reshape([IMSIZE, IMSIZE, 1])
        neg = triplet[:, :, 2].reshape([IMSIZE, IMSIZE, 1])

        embedding_distance_same = nn.compute_embedding_distance(image1=anchor,
                                                           image2=pos)
        embedding_distance_diff = nn.compute_embedding_distance(image1=anchor,
                                                                image2=neg)
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
# Determine threshold for accuracy
########################################

def match(image1, image2, threshold=11):

    distance = nn.compute_embedding_distance(image1=image1,
                                  image2=image2)
    if distance < threshold:
        return 1
    else:
        return 0

matches_pos = []
matches_neg = []

for _ in range(150):
    batch = bg.generate_val_triplets(32)

    for triplet in batch:

        anchor = triplet[:, :, 0].reshape([IMSIZE, IMSIZE, 1])
        pos = triplet[:, :, 1].reshape([IMSIZE, IMSIZE, 1])
        neg = triplet[:, :, 2].reshape([IMSIZE, IMSIZE, 1])

        matches_pos.append(match(anchor, pos))
        matches_neg.append(match(anchor, neg))

sensitivity = np.mean(matches_pos)
specificity = 1-np.mean(matches_neg)

print((sensitivity+specificity)/2)


########################################
# Determine acc for duo network
########################################

samples = 0
correct = 0
pos_preds = []
neg_preds = []

for i in range(10):
    x, y = bg.generate_val_duos(32)
    for img, label in zip(x, y):
        samples += 1
        pred = nn.predict(img[:, :, 0].reshape(IMSIZE, IMSIZE, 1), img[:, :, 1].reshape(IMSIZE, IMSIZE, 1))
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