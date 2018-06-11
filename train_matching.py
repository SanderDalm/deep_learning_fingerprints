import numpy as np
import matplotlib.pyplot as plt

from neural_net import NeuralNet_Matching
from batch_generators import BatchGenerator_Matching

########################################
# Set globals
########################################

path = '/home/sander/data/deep_learning_fingerprints/sd04/png_txt'
IMSIZE = 512
BATCH_SIZE = 32
NUM_STEPS = 1000


########################################
# Train model
########################################

bg = BatchGenerator_Matching(path=path, imsize=IMSIZE)


#x, y = bg.generate_val_duos(32)
#index=-7
#y[index]
#plt.imshow(np.concatenate([x[index, :, :, 0], x[index, :, :, 1]], axis=0))
#plt.show()

nn = NeuralNet_Matching(height=IMSIZE, width=IMSIZE, batchgen=bg, network_type='duos')

loss, val_acc = nn.train(num_steps=NUM_STEPS,
         batch_size=BATCH_SIZE,
         dropout_rate=0,
         lr=.0001,
         decay=1)

plt.plot(loss, color='b', alpha=.7)
plt.plot(val_acc, color='g', alpha=.7)
plt.show()

########################################
# Determine matching threshold
########################################

batch = bg.generate_triplet_batch_validation()

distances_same = []
distances_diff = []

for triplet in batch:

    anchor = triplet[:, :, 0].reshape([512, 512, 1])
    pos = triplet[:, :, 1].reshape([512, 512, 1])
    neg = triplet[:, :, 2].reshape([512, 512, 1])

    embedding_distance_same = nn.compute_embedding_distance(image1=anchor,
                                                       image2=pos,
                                                       h=IMSIZE,
                                                       w=IMSIZE)
    embedding_distance_diff = nn.compute_embedding_distance(image1=anchor,
                                                            image2=neg,
                                                            h=IMSIZE,
                                                            w=IMSIZE)
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

def match(image1, image2, threshold=10):

    distance = nn.compute_embedding_distance(image1=image1,
                                  image2=image2,
                                  h=IMSIZE,
                                  w=IMSIZE)
    if distance < threshold:
        return 1
    else:
        return 0

batch = bg.generate_triplet_batch_validation()

matches_pos = []
matches_neg = []

for triplet in batch:

    anchor = triplet[:, :, 0].reshape([512, 512, 1])
    pos = triplet[:, :, 1].reshape([512, 512, 1])
    neg = triplet[:, :, 2].reshape([512, 512, 1])

    matches_pos.append(match(anchor, pos))
    matches_neg.append(match(anchor, neg))

sensitivity = np.mean(matches_pos)
specificity = 1-np.mean(matches_neg)

print((sensitivity+specificity)/2)