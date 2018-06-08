import numpy as np
import matplotlib.pyplot as plt

from neural_net import NeuralNet_Matching
from batch_generator import BatchGenerator_Matching

#####################################
# Globals
#####################################

path = '/home/sander/data/deep_learning_fingerprints/sd04/png_txt'
IMSIZE = 512
BATCH_SIZE = 32
NUM_STEPS = 2500

# Get batch gen
bg = BatchGenerator_Matching(path=path, imsize=IMSIZE)

#x = bg.generate_triplet_batch(32)
#x[0].shape

# Train
nn = NeuralNet_Matching(height=IMSIZE, width=IMSIZE, batchgen=bg)

loss, val_loss = nn.train(num_steps=NUM_STEPS,
         batch_size=BATCH_SIZE,
         dropout_rate=0,
         lr=.0001,
         decay=1)