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