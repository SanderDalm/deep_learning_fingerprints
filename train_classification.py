from os.path import join

import numpy as np
import matplotlib.pyplot as plt

from scipy.misc import imread, imresize
from skimage.color import rgb2gray
from batch_generators.batch_generator_classification_nist import BatchGenerator_Classification_NIST
from batch_generators.batch_generator_classification_anguli import BatchGenerator_Classification_Anguli
from batch_generators.batch_generator_classification_NFI import BatchGenerator_Classification_NFI
from neural_nets.neural_net_classification import NeuralNet_Classification
from neural_nets.tf_utils import create_visualisation
import config

########################################
# Set globals
########################################0

DATAPATH = join(config.datadir, 'NFI')
META_FILE = 'GeneralPatterns.txt'#'CLASSIFICATION-extended pattern set.pet'
HEIGHT = 512
WIDTH = 512
BATCH_SIZE = 32
NUM_STEPS = 40001
DROPOUT = .5
AUGMENT = 1
DECAY = 1

#bg = BatchGenerator_Classification_Anguli(path=DATAPATH, height=HEIGHT, width=WIDTH)
#bg = BatchGenerator_Classification_NIST(path=DATAPATH, height=HEIGHT, width=WIDTH)
bg = BatchGenerator_Classification_NFI(path=DATAPATH, meta_file=join(DATAPATH, META_FILE), include_aug=True, height=HEIGHT, width=WIDTH, detect_special_patterns=False)

nn = NeuralNet_Classification(HEIGHT, WIDTH, len(bg.label_dict))
nn.load_weights('models/classificatie/neural_net8000.ckpt')

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

    for filename in bg.filenames_val:
        img = imread(bg.path+'/BMP/'+filename)
        img = rgb2gray(img)
        if bg.height != 512 or bg.width != 512:
            img = imresize(img, [bg.height, bg.width])
        embedding = nn.get_embedding(img)
        embeddings.append(embedding)
        labels.append(np.argmax(bg.label_dict_one_hot[filename]))

    return np.array(embeddings), np.array(labels), bg.filenames_val

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

layers = [op for op in nn.session.graph.get_operations() if op.type == 'Conv2D']
input_img = np.random.uniform(size=(1, HEIGHT, WIDTH, 1))
create_visualisation(nn=nn, layer_number=6, input_img=input_img)

########################################
# Laplacian pyramid smoothing
########################################