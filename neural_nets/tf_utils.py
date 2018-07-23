from os.path import join
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def Dense(x, units, activation):
    return tf.layers.dense(inputs=x,
                           units=units,
                           activation=activation,
                           kernel_initializer=tf.keras.initializers.he_normal(),
                           kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                           activity_regularizer=tf.keras.regularizers.l2(l=0.01))


def Conv2D(x, filters, kernel_size, stride, padding='same'):
    return tf.layers.conv2d(inputs=x,
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=stride,
                            activation=tf.nn.relu,
                            kernel_initializer=tf.keras.initializers.he_normal(),
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            activity_regularizer=tf.keras.regularizers.l2(l=0.01),
                            padding=padding)


def CNN(x, dropout_rate=None):

    # x = tf.image.resize_images(x, [299, 299])
    # x = tf.concat([x, x, x], axis=3)
    # model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='max', weights=None)
    # return model(x)

    x = Conv2D(x, 16, 3, 1)
    x = Conv2D(x, 16, 3, 1)
    x = tf.layers.dropout(inputs=x, rate=dropout_rate)
    x = tf.layers.max_pooling2d(x, 2, 2)

    x = Conv2D(x, 32, 3, 1)
    x = Conv2D(x, 32, 3, 1)
    x = tf.layers.dropout(inputs=x, rate=dropout_rate)
    x = tf.layers.max_pooling2d(x, 2, 2)

    x = Conv2D(x, 64, 3, 1)
    x = Conv2D(x, 64, 3, 1)
    x = tf.layers.dropout(inputs=x, rate=dropout_rate)
    x = tf.layers.max_pooling2d(x, 2, 2)

    x = Conv2D(x, 128, 3, 1)
    x = Conv2D(x, 128, 3, 1)
    x = tf.layers.dropout(inputs=x, rate=dropout_rate)
    x = tf.layers.max_pooling2d(x, 2, 2)

    x = Conv2D(x, 128, 3, 1)
    x = Conv2D(x, 128, 3, 1)
    x = tf.layers.dropout(inputs=x, rate=dropout_rate)
    x = tf.layers.max_pooling2d(x, 2, 2)

    flatten = tf.layers.flatten(x)

    return tf.layers.dropout(inputs=flatten, rate=dropout_rate)


def augment(images):

    noise = tf.random_normal(shape=tf.shape(images), mean=0.0, stddev=0.1,
                             dtype=tf.float32)
    images = tf.add(images, noise)

    images = tf.map_fn(lambda img: tf.image.random_brightness(img, max_delta=.8), images)
    images = tf.map_fn(lambda img: tf.image.random_contrast(img, lower=0.2, upper=.8), images)

    return images


def visualize_layers(nn, layer_number, input_img):

    def visualize_layer(nn, op, input, n_iter, stepsize):
        t_score = -tf.reduce_mean(op)
        t_grad = tf.gradients(t_score, nn.x)[0]

        img = input.copy()
        for i in range(n_iter):
            g, score = nn.session.run([t_grad, t_score], {nn.x: img, nn.augment: 0, nn.dropout_rate: 0})
            g /= g.std() + 1e-8
            img += g * stepsize
            print(score, end=' ')
        if len(img.shape) == 3:
            return img.reshape(nn.height, nn.width)
        if len(img.shape) == 4:
            return img[:, :, :, 0].reshape(nn.height, nn.width)

    layers = [op for op in nn.session.graph.get_operations() if op.type == 'Conv2D']
    layer = layers[layer_number]
    layername = layer.name
    print(layername)
    target = nn.session.graph.get_tensor_by_name(layername + ':0')

    num_channels = target.shape.as_list()[3]

    num_rows = int(np.sqrt(num_channels)) + 1
    num_cols = num_rows

    fig, axs = plt.subplots(num_rows, num_cols, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.001, wspace=.001)
    axs = axs.ravel()

    for channel in range(num_channels):
        print(channel)
        img = visualize_layer(nn=nn,
                              input=input_img.copy(),
                              op=target[:, :, :, channel],
                              n_iter=20,
                              stepsize=1)
        axs[channel].imshow(img, cmap='gray')
    plt.show()


def visualize_embedding(bg, nn, datadir):
    from scipy.misc import imread, imresize
    from skimage.color import rgb2gray
    from sklearn.manifold import TSNE
    from bokeh.plotting import figure, output_file, show, ColumnDataSource
    from bokeh.palettes import Category20

    def get_embeddings(bg):

        embeddings = []
        labels = []

        for filename in bg.filenames_val:
            img = imread(bg.path + '/BMP/' + filename)
            img = rgb2gray(img)
            if bg.height != 512 or bg.width != 512:
                img = imresize(img, [bg.height, bg.width])
            embedding = nn.get_embedding(img)
            embeddings.append(embedding)
            labels.append(np.argmax(bg.label_dict_one_hot[filename]))

        return np.array(embeddings), np.array(labels), bg.filenames_val

    embeddings, labels, filenames = get_embeddings(bg)

    tsne = TSNE(perplexity=20)
    embeddings_tsne = tsne.fit_transform(embeddings)

    output_file("toolbar.html")
    source = ColumnDataSource(data=dict(
        x=embeddings_tsne[:, 0],
        y=embeddings_tsne[:, 1],
        desc=filenames,
        color=[Category20[7][x] for x in labels],
        imgs=['file://' + join(datadir, 'BMP', filename) for filename in filenames]
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

# Test augmentation

# tf.enable_eager_execution()
#
# from os.path import join
# #from batch_generators.batch_generator_classification_anguli import BatchGenerator_Classification_Anguli
# from batch_generators.batch_generator_classification_nist import BatchGenerator_Classification_NIST
# from batch_generators.batch_generator_classification_NFI import BatchGenerator_Classification_NFI
# import config
# DATAPATH = join(config.datadir, 'NFI')
# META_FILE = 'GeneralPatterns.txt'
# bg = BatchGenerator_Classification_NFI(path=DATAPATH, meta_file=join(DATAPATH, 'GeneralPatterns.txt'))
# bg = BatchGenerator_Classification_NIST(path=config.datadir+'/sd04/png_txt/')
#
#
# x, y = bg.generate_train_batch(1)
# np.min(x)
# np.max(x)
#
# x2 = augment(x)
# x3 = augment(x)
# x4 = augment(x)
# np.min(x2)
# np.max(x2)
#
# together = np.concatenate([x.reshape(512,512), x2, x3, x4], axis=1)
# plt.imshow(together.reshape(512, 2048), cmap='gray')
# plt.show()
#
# x2 = tf.image.resize_images(x, [299, 299])
#
# plt.imshow(x.reshape(512, 512), cmap='gray')
# plt.show()
#
# plt.imshow(np.array(x2).reshape(299, 299), cmap='gray')
# plt.show()