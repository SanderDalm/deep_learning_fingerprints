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
BATCH_SIZE = 32
NUM_STEPS = 60001
DROPOUT = .5
AUGMENT = 1
DECAY = 1

#bg = BatchGenerator_Classification_Anguli(path=DATAPATH, height=HEIGHT, width=WIDTH)
#bg = BatchGenerator_Classification_NIST(path=DATAPATH, height=HEIGHT, width=WIDTH)
bg = BatchGenerator_Classification_NFI(path=DATAPATH, meta_file=join(DATAPATH, META_FILE), include_aug=True, height=HEIGHT, width=WIDTH, detect_special_patterns=False)

nn = NeuralNet_Classification(HEIGHT, WIDTH, len(bg.label_dict))
nn.load_weights('models/backup_97_procent_acc/neural_net8000.ckpt')

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
    for i in range(150):
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
    filenames = []

    for i in range(30):
        x, y, filenames_batch = bg.generate_val_batch(32, return_filenames=True)
        filenames.extend(filenames_batch)

        for img, label in zip(x, y):
            embedding = nn.get_embedding(img)
            embeddings.append(embedding)
            labels.append(np.argmax(label))

    return np.array(embeddings), np.array(labels), filenames

embeddings, labels, filenames = get_embeddings(bg)

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, NMF
tsne = TSNE(perplexity=20)#NMF(n_components=2)#PCA(n_components=2)#
embeddings_tsne = tsne.fit_transform(embeddings)

# Bokeh versie
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.palettes import Category20

output_file("toolbar.html")
source = ColumnDataSource(data=dict(
    x=embeddings_tsne[:, 0],
    y=embeddings_tsne[:, 1],
    desc=filenames,
    color=[Category20[7][x] for x in labels],
    imgs=['file://'+join(DATAPATH, 'BMP', filename) for filename in filenames]
    ))

TOOLTIPS = [("desc", "@desc")]
TOOLTIPS = """
    <div>
        <div>
            <img
                src="@imgs" height="128" alt="@imgs" width="128"
                style="float: left; margin: 0px 15px 15px 0px;"
                border="2"
            ></img>
        </div>
        <div>
            <span style="font-size: 17px; font-weight: bold;">@desc</span>
            <span style="font-size: 15px; color: #966;">[$index]</span>
        </div>
        <div>
            <span>@fonts{safe}</span>
        </div>
        <div>
            <span style="font-size: 15px;">Location</span>
            <span style="font-size: 10px; color: #696;">($x, $y)</span>
        </div>
    </div>
"""
p = figure(plot_width=1200, plot_height=800, tooltips=TOOLTIPS,
           title="Mouse over the dots")


p.circle('x', 'y', size=10, color='color', source=source)

show(p)

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

for layer in layers[8:9]:
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

        axs[channel].imshow(img[:64, :64], cmap='gray')


########################################
# Laplacian pyramid smoothing
########################################