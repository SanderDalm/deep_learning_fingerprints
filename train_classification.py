import numpy as np
import matplotlib.pyplot as plt
from batch_generator_classification import BatchGenerator_Classification
from neural_net_classification import NeuralNet_Classification

########################################
# Set globals
########################################

path = '/mnt/ssd/data/deep_learning_fingerprints/sd04/png_txt' #'/home/sander/data/deep_learning_fingerprints/sd04/png_txt'
IMSIZE = 512
BATCH_SIZE = 32
NUM_STEPS = 1001

bg = BatchGenerator_Classification(path, IMSIZE)

nn = NeuralNet_Classification(IMSIZE, bg)

loss, val_loss = nn.train(num_steps=NUM_STEPS,
         batch_size=BATCH_SIZE,
         dropout_rate=0.5,
         lr=.0001,
         decay=1,
         augment=False)


plt.plot(loss, color='b', alpha=.7)
plt.plot(val_loss, color='g', alpha=.7)
plt.show()


########################################
# Determine acc
########################################

samples = 0
correct = 0
for i in range(10):
    x, y = bg.generate_train_batch(32)
    for img, label in zip(x, y):
        samples += 1
        pred = nn.predict(img)
        if np.argmax(pred) == np.argmax(label):
            correct += 1

print('Train acc: {}'.format(correct/samples))

samples = 0
correct = 0
for i in range(10):
    x, y = bg.generate_val_batch(32)
    for img, label in zip(x, y):
        samples += 1
        pred = nn.predict(img)
        if np.argmax(pred) == np.argmax(label):
            correct += 1

print('Val acc: {}'.format(correct/samples))