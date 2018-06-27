import tensorflow as tf

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

    noise = tf.random_normal(shape=tf.shape(images), mean=0.0, stddev=0.2,
                             dtype=tf.float32)
    images = tf.add(images, noise)

    images = tf.map_fn(lambda img: tf.image.random_brightness(img, max_delta=1.), images)
    images = tf.map_fn(lambda img: tf.image.random_contrast(img, lower=0, upper=1), images)

    return images


# Test augmentation

# tf.enable_eager_execution()
#
# from batch_generators.batch_generator_classification_anguli import BatchGenerator_Classification_Anguli
# import matplotlib.pyplot as plt
# import numpy as np
#
# bg = BatchGenerator_Classification_Anguli()
#
# x, y = bg.generate_train_batch(1)
# np.min(x)
# np.max(x)
#
# x2 = augment(x)
# np.min(x2)
# np.max(x2)
#
# together = np.concatenate([x, x2], axis=1)
# plt.imshow(together.reshape(800, 275), cmap='gray')
# plt.show()