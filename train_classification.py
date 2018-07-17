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
META_FILE = 'CLASSIFICATION-extended pattern set.pet' # 'GeneralPatterns.txt'#
HEIGHT = 512
WIDTH = 512
BATCH_SIZE = 32
NUM_STEPS = 8001
DROPOUT = .5
AUGMENT = 1
DECAY = 1

#bg = BatchGenerator_Classification_Anguli(path=DATAPATH, height=HEIGHT, width=WIDTH)
#bg = BatchGenerator_Classification_NIST(path=DATAPATH, height=HEIGHT, width=WIDTH)
bg = BatchGenerator_Classification_NFI(path=DATAPATH, meta_file=join(DATAPATH, META_FILE), include_aug=True, height=HEIGHT, width=WIDTH, detect_special_patterns=True)

nn = NeuralNet_Classification(HEIGHT, WIDTH, len(bg.label_dict))

loss, val_loss = nn.train(num_steps=NUM_STEPS,
                          batchgen=bg,
                          batch_size=BATCH_SIZE,
                          dropout_rate=DROPOUT,
                          augment=AUGMENT,
                          lr=.0001,
                          decay=DECAY)

nn.load_weights('models/backup_97_procent_acc/neural_net8000.ckpt')

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
    for i in range(50):
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
get_acc(bg, 'train')
get_acc(bg, 'val')

########################################
# Plot embeddings with t-sne.
########################################

def get_embeddings(bg):

    embeddings = []
    labels = []

    for i in range(30):
        x, y = bg.generate_val_batch(32)

        for img, label in zip(x, y):
            embedding = nn.get_embedding(img)
            embeddings.append(embedding)
            labels.append(np.argmax(label))

    return np.array(embeddings), np.array(labels)

embeddings, labels = get_embeddings(bg)

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
tsne = TSNE(perplexity=20)#PCA(n_components=2)
embeddings_tsne = tsne.fit_transform(embeddings)

color_dict = {0: 'b', 1: 'g', 2: 'r', 3: 'c', 4: 'm', 5: 'y', 6: 'k'}
bg.label_dict
colors = [color_dict[x] for x in labels]

plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=colors)


########################################
# Visualize convolutional layers
########################################

import tensorflow as tf

def visualize_layer(nn, op, input, n_iter, stepsize):
    t_score = -tf.reduce_mean(op)
    t_grad = tf.gradients(t_score, nn.x)[0]

    img = input.copy()
    for i in range(n_iter):
        g, score = nn.session.run([t_grad, t_score], {nn.x: img, nn.augment: 0, nn.dropout_rate: 0})
        g /= g.std() + 1e-8
        img += g * stepsize
        print(score, end=' ')
    return img.reshape(nn.height, nn.width)


layers = [op for op in nn.session.graph.get_operations() if op.type == 'Conv2D']

for layer in layers[0:1]:
    layername = layer.name
    print(layername)
    target = nn.session.graph.get_tensor_by_name(layername + ':0')

    num_channels = target.shape.as_list()[3]

    num_rows = int(np.sqrt(num_channels))+1
    num_cols = num_rows

    fig, axs = plt.subplots(num_rows, num_cols, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.001, wspace=.001)
    axs = axs.ravel()

    for channel in range(num_channels):
        print(channel)
        img_noise = np.random.uniform(size=(1, 512, 512, 1))
        img = visualize_layer(nn=nn,
                              input=img_noise.copy(),
                              op=target[:, :, :, channel],
                              n_iter=20,
                              stepsize=1)

        axs[channel].imshow(img[:128, :128], cmap='gray')