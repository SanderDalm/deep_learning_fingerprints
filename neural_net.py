import tensorflow as tf
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
    x = Conv2D(x, 16, 3, 1)
    x = tf.layers.max_pooling2d(x, 2, 2)

    x = Conv2D(x, 32, 3, 1)
    x = tf.layers.max_pooling2d(x, 2, 2)

    x = Conv2D(x, 64, 3, 1)
    x = tf.layers.max_pooling2d(x, 2, 2)

    x = Conv2D(x, 128, 3, 1)
    x = tf.layers.max_pooling2d(x, 2, 2)

    return tf.layers.flatten(x)


def augment(images):

    images = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), images)
    images = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), images)
    noise = tf.random_normal(shape=tf.shape(images), mean=0.0, stddev=0.1,
                             dtype=tf.float32)
    images = tf.add(images, noise)

    return images



class NeuralNet_Classification:

    def __init__(self, imsize, batchgen):

        self.imsize = imsize

        self.batchgen = batchgen

        self.graph = tf.Graph()

        self.session = tf.Session()  # config=tf.ConfigProto(log_device_placement=True)

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, imsize, imsize, 1], name='input')
        self.x_standardized = tf.map_fn(lambda img: tf.image.per_image_standardization(img), self.x)

        #self.augment = tf.placeholder(tf.float32)
        #self.cnn_input = tf.cond(self.augment > 0, lambda: augment(self.x), lambda: self.x)

        self.dropout_rate = tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)

        self.cnn_output = CNN(self.x_standardized, self.dropout_rate)

        self.fc1 = Dense(self.cnn_output, 256, tf.nn.relu)
        self.fc2 = Dense(self.fc1, 256, tf.nn.relu)
        self.logits = Dense(self.fc2, 5, None)
        self.prediction = tf.nn.softmax(self.logits)

        self.label = tf.placeholder(tf.int32, [None, 5])
        self.loss = tf.losses.softmax_cross_entropy(logits=self.logits, onehot_labels=self.label)


        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

        self.init_op = tf.global_variables_initializer()
        self.session.run(self.init_op)
        self.saver = tf.train.Saver(max_to_keep=None,
                                    name='checkpoint_saver')


    def train(self, num_steps, batch_size, dropout_rate, lr, decay, checkpoint='models/neural_net'):

        loss_list = []
        val_loss_list = []

        for step in range(num_steps):


            x_batch, y_batch = self.batchgen.generate_train_batch(batch_size)
            feed_dict = {
                        self.x: x_batch,
                        self.label: y_batch,
                        self.dropout_rate: dropout_rate,
                        self.lr: lr
                        }

            loss_, _ = self.session.run([self.loss, self.train_step], feed_dict=feed_dict)
            lr *= decay

            if step % 100 == 0:
                x_batch, y_batch = self.batchgen.generate_val_batch(batch_size)
                feed_dict = {
                            self.x: x_batch,
                            self.label: y_batch,
                            self.dropout_rate: 0
                            }

                val_loss = self.session.run([self.loss], feed_dict=feed_dict)
                val_loss_list.append(val_loss)
                loss_list.append(loss_)
                print('step: {}'.format(step))
                print('train loss: {}'.format(loss_))
                print('val loss: {}'.format(val_loss))
                print('lr: {}'.format(lr))
                print('')

            if (step + 1) % 1000 == 0 or step == num_steps - 1:
                self.saver.save(self.session, checkpoint + str(step) + '.ckpt')
                print('Saved to {}'.format(checkpoint + str(step) + '.ckpt'))

        return loss_list, val_loss_list


    def predict(self, image):

        feed_dict = {
            self.x: image.reshape(1, self.imsize, self.imsize, 1),
            self.dropout_rate: 0
        }
        pred = self.session.run([self.prediction], feed_dict=feed_dict)

        return pred[0][0]


class NeuralNet_Matching:

    def __init__(self, imsize, batchgen, network_type='triplet'):

        self.imsize = imsize

        self.network_type = network_type

        self.batchgen = batchgen

        self.graph = tf.Graph()

        self.session = tf.Session()  # config=tf.ConfigProto(log_device_placement=True)

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.imsize, self.imsize, 3], name='input')
        self.x_standardized = tf.map_fn(lambda img: tf.image.per_image_standardization(img), self.x)

        self.dropout_rate = tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)

        self.anchor, self.pos, self.neg = tf.split(self.x, 3, axis=3)

        self.augment = tf.placeholder(tf.float32)
        self.anchor = tf.cond(self.augment > 0, lambda: augment(self.anchor), lambda: self.anchor)
        self.pos = tf.cond(self.augment > 0, lambda: augment(self.pos), lambda: self.pos)
        self.neg = tf.cond(self.augment > 0, lambda: augment(self.neg), lambda: self.neg)

        with tf.variable_scope('scope'):
            self.anchor_embedding = CNN(self.anchor, self.dropout_rate)
        with tf.variable_scope('scope', reuse=True):
            self.pos_embedding = CNN(self.pos, self.dropout_rate)
        with tf.variable_scope('scope', reuse=True):
            self.neg_embedding = CNN(self.neg, self.dropout_rate)


        if self.network_type == 'triplet':
            self.anchor_minus_pos = tf.norm(self.anchor_embedding - self.pos_embedding, axis=1)
            self.anchor_minus_neg = tf.norm(self.anchor_embedding - self.neg_embedding, axis=1)
            self.loss = tf.reduce_mean(tf.maximum(self.anchor_minus_pos - self.anchor_minus_neg + 1, 0))

        if self.network_type == 'duos':
            self.concat = tf.concat([self.anchor_embedding, self.pos_embedding], axis=1)
            self.fc1 = Dense(self.concat, 256, tf.nn.relu)
            self.fc2 = Dense(self.fc1, 256, tf.nn.relu)
            self.prediction = Dense(self.fc2, 1, tf.nn.sigmoid)

            self.label = tf.placeholder(tf.int32, [None, 1])
            self.loss = tf.losses.log_loss(labels=self.label, predictions=self.prediction)


        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

        self.init_op = tf.global_variables_initializer()
        self.session.run(self.init_op)
        self.saver = tf.train.Saver(max_to_keep=None,
                                    name='checkpoint_saver')


    def train(self, num_steps, batch_size, dropout_rate, lr, decay, checkpoint='models/neural_net'):

        loss_list = []
        val_loss_list = []

        for step in range(num_steps):

            if self.network_type == 'triplet':
                x_batch = self.batchgen.generate_train_triplets(batch_size)
                feed_dict = {
                            self.x: x_batch,
                            self.dropout_rate: dropout_rate,
                            self.augment:1,
                            self.lr: lr
                            }
            if self.network_type == 'duos':
                x_batch, y_batch = self.batchgen.generate_train_duos(batch_size)
                feed_dict = {
                            self.x: x_batch,
                            self.label: y_batch,
                            self.dropout_rate: dropout_rate,
                            self.lr: lr
                            }


            loss_, _ = self.session.run([self.loss, self.train_step], feed_dict=feed_dict)
            lr *= decay

            if step % 100 == 0:
                if self.network_type == 'triplet':
                    x_batch = self.batchgen.generate_val_triplets(batch_size)
                    feed_dict = {
                                self.x: x_batch,
                                self.augment: 0,
                                self.dropout_rate: 0
                                }
                if self.network_type == 'duos':
                    x_batch, y_batch = self.batchgen.generate_val_duos(batch_size)
                    feed_dict = {
                        self.x: x_batch,
                        self.label: y_batch,
                        self.dropout_rate: dropout_rate,
                        self.lr: lr
                    }
                val_loss = self.session.run([self.loss], feed_dict=feed_dict)
                val_loss_list.append(val_loss)
                loss_list.append(loss_)
                print('step: {}'.format(step))
                print('train loss: {}'.format(loss_))
                print('val loss: {}'.format(val_loss))
                print('lr: {}'.format(lr))
                print('')

            if (step + 1) % 1000 == 0 or step == num_steps - 1:
                self.saver.save(self.session, checkpoint + str(step) + '.ckpt')
                print('Saved to {}'.format(checkpoint + str(step) + '.ckpt'))

        return loss_list, val_loss_list


    def compute_embedding_distance(self, image1, image2):

        x = np.concatenate([image1, image2, image2], axis=2) # The second image2 is a placeholder because the network expects triplets
        x = x.reshape([1, self.imsize, self.imsize, 3])

        feed_dict = {
            self.x: x,
            self.dropout_rate: 0,
            self.augment: 0
                    }
        embedding_distance = self.session.run([self.anchor_minus_pos],
                         feed_dict=feed_dict)

        return embedding_distance[0][0]


    def predict(self, image1, image2):

        x = np.concatenate([image1, image2, image2],
                           axis=2)  # The second image2 is a placeholder because the network expects triplets
        x = x.reshape([1, self.imsize, self.imsize, 3])

        feed_dict = {
            self.x: x,
            self.dropout_rate: 0,
            self.augment: 1
            }
        pred = self.session.run([self.prediction], feed_dict=feed_dict)

        return pred[0][0][0]