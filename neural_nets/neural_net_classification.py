import tensorflow as tf
from neural_nets.tf_utils import Dense, CNN, augment

class NeuralNet_Classification:

    def __init__(self, height, width, categories):

        self.height = height
        self.width = width

        self.graph = tf.Graph()

        self.session = tf.Session()  # config=tf.ConfigProto(log_device_placement=True)

        # Feed placeholders
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.height, self.width, 1], name='input')
        self.dropout_rate = tf.placeholder(tf.float32)
        self.augment = tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)


        # Standardization and augmentation
        self.x_standardized = tf.map_fn(lambda img: tf.image.per_image_standardization(img), self.x)
        self.cnn_input = self.x_standardized#tf.cond(self.augment > 0, lambda: augment(self.x_standardized), lambda: self.x_standardized)

        # Run the network
        self.cnn_output = CNN(self.cnn_input, self.dropout_rate)

        self.fc1 = Dense(self.cnn_output, 1024, tf.nn.relu)
        self.fc1 = tf.layers.dropout(inputs=self.fc1, rate=self.dropout_rate)

        self.fc2 = Dense(self.fc1, 1024, tf.nn.relu)
        self.logits = Dense(self.fc2, categories, None)
        self.prediction = tf.nn.softmax(self.logits)

        self.label = tf.placeholder(tf.int32, [None, categories])
        self.loss = tf.losses.softmax_cross_entropy(logits=self.logits, onehot_labels=self.label)


        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

        self.init_op = tf.global_variables_initializer()
        self.session.run(self.init_op)
        self.saver = tf.train.Saver(max_to_keep=None,
                                    name='checkpoint_saver')


    def train(self, num_steps, batchgen, batch_size, dropout_rate, augment, lr, decay, checkpoint='models/neural_net'):

        loss_list = []
        val_loss_list = []

        for step in range(num_steps):


            x_batch, y_batch = batchgen.generate_train_batch(batch_size)
            feed_dict = {
                        self.x: x_batch,
                        self.label: y_batch,
                        self.dropout_rate: dropout_rate,
                        self.augment: augment,
                        self.lr: lr
                        }

            loss_, _ = self.session.run([self.loss, self.train_step], feed_dict=feed_dict)
            lr *= decay

            if step % 100 == 0:
                x_batch, y_batch = batchgen.generate_val_batch(batch_size)
                feed_dict = {
                            self.x: x_batch,
                            self.label: y_batch,
                            self.dropout_rate: 0,
                            self.augment: 0
                            }

                val_loss = self.session.run([self.loss], feed_dict=feed_dict)
                val_loss_list.append(val_loss)
                loss_list.append(loss_)
                print('step: {}'.format(step))
                print('train loss: {}'.format(loss_))
                print('val loss: {}'.format(val_loss))
                print('lr: {}'.format(lr))
                print('')

            if step % 100 == 0 or step == num_steps - 1:
                self.saver.save(self.session, checkpoint + str(step) + '.ckpt')
                print('Saved to {}'.format(checkpoint + str(step) + '.ckpt'))

        return loss_list, val_loss_list


    def predict(self, image):

        feed_dict = {
            self.x: image.reshape(1, self.height, self.width, 1),
            self.dropout_rate: 0,
            self.augment: 0
        }
        pred = self.session.run([self.prediction], feed_dict=feed_dict)

        return pred[0][0]


    def get_embedding(self, image):

        feed_dict = {
            self.x: image.reshape(1, self.height, self.width, 1),
            self.dropout_rate: 0,
            self.augment: 0
        }
        embedding = self.session.run([self.fc2], feed_dict=feed_dict)

        return embedding[0][0]


    def load_weights(self, path):
        self.saver.restore(self.session, path)
        print('Weights loaded.')


    def visualize_layer(self, op, input, n_iter, stepsize):

        # start with a gray image with a little noise
        t_score = -tf.reduce_mean(op)  # defining the optimization objective
        t_grad = tf.gradients(t_score, self.x)[0]  # behold the power of automatic differentiation!

        img = input.copy()
        for i in range(n_iter):
            g, score = self.session.run([t_grad, t_score], {self.x: img, self.augment:0, self.dropout_rate:0})
            # normalizing the gradient, so the same step size should work
            g /= g.std() + 1e-8  # for different layers and networks
            img += g * stepsize
            print(score, end=' ')
        print(img.shape)
        return img.reshape(self.height, self.width)