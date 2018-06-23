import tensorflow as tf
import numpy as np
from neural_nets.tf_utils import Dense, CNN

class NeuralNet_Matching:

    def __init__(self, height, width, network_type='triplets'):

        self.height = height
        self.width = width

        self.network_type = network_type

        self.graph = tf.Graph()

        self.session = tf.Session()  # config=tf.ConfigProto(log_device_placement=True)

        # Feed placeholders
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.height, self.width, 1], name='input')
        self.dropout_rate = tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)

        # Split inputs
        self.anchor, self.pos, self.neg = tf.split(self.x, 3, axis=3)


        # Standardization
        self.anchor = tf.map_fn(lambda img: tf.image.per_image_standardization(img), self.anchor)
        self.pos = tf.map_fn(lambda img: tf.image.per_image_standardization(img), self.pos)
        self.neg = tf.map_fn(lambda img: tf.image.per_image_standardization(img), self.neg)


        # Run the network
        with tf.variable_scope('scope'):
            self.anchor_embedding = CNN(self.anchor, self.dropout_rate)
        with tf.variable_scope('scope', reuse=True):
            self.pos_embedding = CNN(self.pos, self.dropout_rate)
        with tf.variable_scope('scope', reuse=True):
            self.neg_embedding = CNN(self.neg, self.dropout_rate)


        if self.network_type == 'triplets':
            self.anchor_minus_pos = tf.norm(self.anchor_embedding - self.pos_embedding, axis=1)
            self.anchor_minus_neg = tf.norm(self.anchor_embedding - self.neg_embedding, axis=1)
            self.loss = tf.reduce_mean(tf.maximum(self.anchor_minus_pos - self.anchor_minus_neg + 1, 0))

        if self.network_type == 'duos':
            self.embedding_diff = tf.abs(self.anchor_embedding - self.pos_embedding)
            self.fc1 = Dense(self.embedding_diff, 256, tf.nn.relu)
            self.fc1 = tf.layers.dropout(inputs=self.fc1, rate=self.dropout_rate)
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


    def train(self, num_steps, batchgen, batch_size, dropout_rate, lr, decay, augment, checkpoint='models/neural_net'):

        loss_list = []
        val_loss_list = []

        for step in range(num_steps):

            if self.network_type == 'triplets':
                x_batch = batchgen.generate_train_triplets(batch_size, augment)
                feed_dict = {
                            self.x: x_batch,
                            self.dropout_rate: dropout_rate,
                            self.lr: lr
                            }
            if self.network_type == 'duos':
                x_batch, y_batch = batchgen.generate_train_duos(batch_size, augment)
                feed_dict = {
                            self.x: x_batch,
                            self.label: y_batch,
                            self.dropout_rate: dropout_rate,
                            self.lr: lr
                            }


            loss_, _ = self.session.run([self.loss, self.train_step], feed_dict=feed_dict)
            lr *= decay

            if step % 1000 == 0:
                if self.network_type == 'triplets':
                    x_batch = batchgen.generate_val_triplets(batch_size, False)
                    feed_dict = {
                                self.x: x_batch,
                                self.dropout_rate: 0
                                }
                if self.network_type == 'duos':
                    x_batch, y_batch = batchgen.generate_val_duos(batch_size, False)
                    feed_dict = {
                        self.x: x_batch,
                        self.label: y_batch,
                        self.dropout_rate: dropout_rate,

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


    def predict(self, image1, image2):

        x = np.concatenate([image1, image2, image2],
                           axis=2)  # The second image2 is a placeholder because the network expects triplets
        x = x.reshape([1, self.height, self.width, 3])

        feed_dict = {
            self.x: x,
            self.dropout_rate: 0
            }

        if self.network_type == 'duos':
            pred, anchor_embedding, partner_embedding = self.session.run([self.prediction, self.anchor_embedding, self.pos_embedding], feed_dict=feed_dict)
            return pred[0][0], np.linalg.norm(anchor_embedding - partner_embedding, axis=1)[0]
        if self.network_type == 'triplets':
            anchor_embedding, partner_embedding = self.session.run(
                [self.anchor_embedding, self.pos_embedding], feed_dict=feed_dict)
            return None, np.linalg.norm(anchor_embedding - partner_embedding, axis=1)[0]





    def load_weights(self, path):
        self.saver.restore(self.session, path)
        print('Weights loaded.')