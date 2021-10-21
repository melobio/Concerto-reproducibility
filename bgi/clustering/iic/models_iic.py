import copy
import sys
import time
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from matplotlib.ticker import FormatStrFormatter

# import data loader
from bgi.clustering.iic.data import load

# import computational graphs
from bgi.clustering.iic.graphs import BasicNetwork, build_network, KERNEL_INIT, BIAS_INIT  # IICGraph, VGG

# import utility functions
from bgi.clustering.iic.utils import unsupervised_labels, save_performance
from bgi.metrics.clustering_metrics import ACC

# plot settings
DPI = 600


class ClusterIIC(object):
    def __init__(self, num_classes, learning_rate, num_repeats, save_dir=None):
        """
        :param num_classes: number of classes
        :param learning_rate: gradient step size
        :param num_repeats: number of data repeats for x and g(x), used to up-sample
        """
        # save configuration
        self.k_A = 3 * num_classes
        self.num_A_sub_heads = 1
        self.k_B = num_classes
        self.num_B_sub_heads = 3
        self.num_repeats = num_repeats

        # initialize losses
        self.loss_A = None
        self.loss_B = None
        self.losses = []

        # initialize outputs
        self.y_hats = None

        # initialize optimizer
        self.is_training = True
        self.learning_rate = learning_rate
        self.global_step = 100
        self.opt = tf.keras.optimizers.Adam(self.learning_rate)
        self.train_ops = []

        # initialize performance dictionary
        self.perf = None
        self.save_dir = save_dir

        # configure performance plotting
        self.fig_learn, self.ax_learn = plt.subplots(1, 2)

    def __iic_loss(self, pi_x, pi_gx, lamb=1.0, EPS=sys.float_info.epsilon):

        # up-sample non-perturbed to match the number of repeat samples
        pi_x = tf.tile(pi_x, [self.num_repeats] + [1] * len(pi_x.shape.as_list()[1:]))

        # get K
        k = pi_x.shape.as_list()[1]

        # compute P
        p = tf.transpose(pi_x) @ pi_gx

        # enforce symmetry
        p = (p + tf.transpose(p)) / 2

        # enforce minimum value
        p = tf.clip_by_value(p, clip_value_min=1e-9, clip_value_max=tf.float32.max)

        # normalize
        p /= tf.reduce_sum(p)

        # get marginals
        pi = tf.broadcast_to(tf.reshape(tf.reduce_sum(p, axis=0), (k, 1)), (k, k))
        pj = tf.broadcast_to(tf.reshape(tf.reduce_sum(p, axis=1), (1, k)), (k, k))

        # complete the loss
        loss = -tf.reduce_sum(p * (tf.math.log(p) - tf.math.log(pi) - tf.math.log(pj)))

        # k = pi_x.shape[-1]
        # # skip assertions
        #
        # # joint probability
        # pi_x = tf.tile(pi_x, [self.num_repeats] + [1] * len(pi_x.shape.as_list()[1:]))
        # p = tf.reduce_sum(tf.expand_dims(pi_x, 2) * tf.expand_dims(pi_gx, 1), 0)
        # p = (p + tf.transpose(p)) / 2  # symmetry
        # p = tf.clip_by_value(p, EPS, 1e9)
        # p /= tf.reduce_sum(p)  # normalize
        #
        # pi = tf.broadcast_to(tf.reshape(tf.reduce_sum(p, axis=0), (k, 1)), (k, k))
        # pj = tf.broadcast_to(tf.reshape(tf.reduce_sum(p, axis=1), (1, k)), (k, k))
        #
        # loss = -tf.reduce_sum(p * (tf.math.log(p) - lamb * tf.math.log(pi) - lamb * tf.math.log(pj)))
        return loss

    @staticmethod
    def __head_out(z: tf.keras.models.Model, k, name):

        # construct a new head that operates on the model's output for x
        # with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        x = z.layers[-1].output
        print("x: ", x)
        output = tf.keras.layers.Dense(
            units=k,
            activation=tf.nn.softmax,
            use_bias=True,
            kernel_initializer=KERNEL_INIT,
            bias_initializer=BIAS_INIT)(x)

        phi = tf.keras.models.Model(inputs=z.input, outputs=output)
        phi.compile(optimizer='adam', loss=tf.keras.losses.kullback_leibler_divergence)
        return phi

    def __head_loss(self, z_x, z_gx, k, num_sub_heads, head):

        # loop over the number of sub-heads
        pi_x_list = []
        pi_gx_list = []
        for i in range(num_sub_heads):
            # run the model
            pi_x = self.__head_out(z_x, k, name=head + str(i + 1))
            pi_gx = self.__head_out(z_gx, k, name=head + str(i + 1))

            pi_x_list.append(pi_x)
            pi_gx_list.append(pi_gx)

        return pi_x_list, pi_gx_list

    def __performance_dictionary_init(self, num_epochs):
        """
        :param num_epochs: maximum number of epochs (used to size buffers)
        :return: None
        """
        # initialize performance dictionary
        self.perf = dict()

        # loss terms
        self.perf.update({'loss_A': np.zeros(num_epochs)})
        self.perf.update({'loss_B': np.zeros(num_epochs)})

        # classification error
        self.perf.update({'class_err_min': np.zeros(num_epochs)})
        self.perf.update({'class_err_avg': np.zeros(num_epochs)})
        self.perf.update({'class_err_max': np.zeros(num_epochs)})

    def __classification_accuracy(self, test_ds: tf.data.Dataset, heads: list, idx):
        """
        :param sess: TensorFlow session
        :param iter_init: TensorFlow data iterator initializer associated
        :param idx: insertion index (i.e. epoch - 1)
        :param y_ph: TensorFlow placeholder for unseen labels
        :return: None
        """
        if self.perf is None:
            return

        # def test_step(heads: list, x_data, y_labels, num_sub_heads):
        #     all_preditions = {}
        #     all_labels = {}
        #     for ii in range(num_sub_heads):
        #         head_out_network = heads[ii]
        #         preditions = head_out_network.predict(x_data)
        #         predctions = tf.argmax(preditions, axis=-1)
        #         predctions = np.array(predctions)
        #         # print("preditions: ", preditions)
        #
        #         if ii not in all_preditions.keys():
        #             all_preditions[ii] = []
        #             all_labels[ii] = []
        #         all_preditions[ii].extend(predctions)
        #         all_labels[ii].extend(np.array(y_labels))
        #
        #     return all_labels, all_preditions

        # all_preditions = {}
        # all_labels = {}
        # for record in test_ds:
        #     x_data = record['x']
        #     label_data = record['label']

        for ii in range(self.num_B_sub_heads):
            head_out_network = heads[ii]

            all_preditions = []
            all_labels = []
            for record in test_ds:
                x_data = record['x']
                label_data = record['label']

                preditions = head_out_network.predict(x_data)
                predctions = tf.argmax(preditions, axis=-1)
                predctions = np.array(predctions)
                all_preditions.extend(predctions)
                all_labels.extend(np.array(label_data))

            y_true = all_labels
            y_pred = all_preditions
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            print("ACC: ", ACC(y_true, y_pred))


            #labels, predictions = test_step(heads, x_data, label_data, self.num_B_sub_heads)

            # for key in list(labels.keys()):
            #     if key not in all_preditions.keys():
            #         all_labels[key] = []
            #         all_preditions[key] = []
            #     all_labels[key].extend(labels[key])
            #     all_preditions[key].extend(predictions[key])

        # for key in list(labels.keys()):
        #     y_true = all_labels[key]
        #     y_pred = all_preditions[key]
        #     y_true = np.array(y_true)
        #     y_pred = np.array(y_pred)
        #     print("ACC: ", ACC(y_true, y_pred))

        # print("all_labels: ", all_labels)
        # print("all_preditions: ", all_preditions)
        # compute classification accuracy
        # if y_ph is not None:
        # class_errors = [unsupervised_labels(all_labels[ii], all_preditions[ii], self.k_B, self.k_B)
        #                 for ii in range(self.num_B_sub_heads)]
        # self.perf['class_err_min'][idx] = np.min(class_errors)
        # self.perf['class_err_avg'][idx] = np.mean(class_errors)
        # self.perf['class_err_max'][idx] = np.max(class_errors)

        # metrics are done
        print('Done')

    def plot_learning_curve(self, epoch):
        """
        :param epoch: epoch number
        :return: None
        """
        # generate epoch numbers
        t = np.arange(1, epoch + 1)

        # colors
        c = {'Head A': '#1f77b4', 'Head B': '#ff7f0e'}

        # plot the loss
        self.ax_learn[0].clear()
        self.ax_learn[0].set_title('Loss')
        self.ax_learn[0].plot(t, self.perf['loss_A'][:epoch], label='Head A', color=c['Head A'])
        self.ax_learn[0].plot(t, self.perf['loss_B'][:epoch], label='Head B', color=c['Head B'])
        self.ax_learn[0].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        self.ax_learn[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # plot the classification error
        self.ax_learn[1].clear()
        self.ax_learn[1].set_title('Class. Error (Min, Avg, Max)')
        self.ax_learn[1].plot(t, self.perf['class_err_avg'][:epoch], color=c['Head B'])
        self.ax_learn[1].fill_between(t,
                                      self.perf['class_err_min'][:epoch],
                                      self.perf['class_err_max'][:epoch],
                                      facecolor=c['Head B'], alpha=0.5)
        self.ax_learn[1].plot(t, self.perf['class_err_avg'][:epoch], color=c['Head B'])
        self.ax_learn[1].fill_between(t,
                                      self.perf['class_err_min'][:epoch],
                                      self.perf['class_err_max'][:epoch],
                                      facecolor=c['Head B'], alpha=0.5)
        self.ax_learn[1].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        self.ax_learn[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # make the legend
        self.ax_learn[1].legend(handles=[patches.Patch(color=val, label=key) for key, val in c.items()],
                                ncol=len(c),
                                bbox_to_anchor=(0.35, -0.06))

        # eliminate those pesky margins
        self.fig_learn.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.95, wspace=0.25, hspace=0.3)

    def train(self, train_set: tf.data.Dataset, test_set: tf.data.Dataset, num_epochs, batch_size=128,
              early_stop_buffer=15):
        """
        :param graph: the computational graph
        :param train_set: TensorFlow Dataset object that corresponds to training data
        :param test_set: TensorFlow Dataset object that corresponds to validation data
        :param num_epochs: number of epochs
        :param early_stop_buffer: early stop look-ahead distance (in epochs)
        :return: None
        """
        z_x_network = build_network(config='B', batch_norm=True, fan_out_init=64)
        z_gx_network = build_network(config='B', batch_norm=True, fan_out_init=64)

        x_randon = np.zeros((batch_size, 24, 24, 1))
        z_x = z_x_network
        z_gx = z_gx_network

        pi_x_A, pi_gx_A = self.__head_loss(z_x, z_gx, self.k_A, self.num_A_sub_heads, 'A')
        pi_x_B, pi_gx_B = self.__head_loss(z_x, z_gx, self.k_B, self.num_B_sub_heads, 'B')

        self.__performance_dictionary_init(num_epochs)

        # train_set = train_set.shuffle(1000, reshuffle_each_iteration=True)
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        loss_object = tf.keras.losses.kullback_leibler_divergence
        # loop over the number of epochs
        for i in range(num_epochs):
            # start timer
            start = time.time()

            train_loss.reset_states()

            # get epoch number
            epoch = i + 1
            # get training operation
            i_train = i % 2

            if i_train == 0:
                num_sub_heads = self.num_A_sub_heads
                pi_x_head = pi_x_A
                pi_gx_head = pi_gx_A
            else:
                num_sub_heads = self.num_B_sub_heads
                pi_x_head = pi_x_B
                pi_gx_head = pi_gx_B
            step = 0
            for record in train_set:
                x_data = record['x']
                gx_data = record['gx']

                losses = 0.0
                for ii in range(num_sub_heads):
                    with tf.GradientTape() as tape:
                        #

                        # run the model
                        pi_x = pi_x_head[ii](x_data)
                        pi_gx = pi_gx_head[ii](gx_data)
                        # accumulate the clustering loss
                        #tf.keras.losses.
                        #pi_x = tf.tile(pi_x, [self.num_repeats] + [1] * len(pi_x.shape.as_list()[1:]))
                        losses += self.__iic_loss(pi_x, pi_gx) #loss_object(pi_x, pi_gx) #

                        # take the average
                        # if num_sub_heads > 0:
                        #     losses /= num_sub_heads
                        # print("Losses: ", float(losses))
                    variables = [pi_x_head[ii].trainable_variables, pi_gx_head[ii].trainable_variables]
                    gradients = tape.gradient(losses, variables)
                    for grad, var in zip(gradients, variables):
                        self.opt.apply_gradients(zip(grad, var))

                train_loss(losses)

                # variables = [] #[z_x_network.trainable_variables, z_gx_network.trainable_variables]
                # for head in pi_x_head:
                #     variables.append(head.trainable_variables)
                # for head in pi_gx_head:
                #     variables.append(head.trainable_variables)
                #
                # gradients = tape.gradient(losses, variables)
                # for grad, var in zip(gradients, variables):
                #     self.opt.apply_gradients(zip(grad, var))

                step += 1
                if step % 100 == 0:
                    print("epoch: ", epoch, ", step: ", step, ', losses: ', float(train_loss.result()))

            #self.__classification_accuracy(test_set, pi_gx_B, epoch)


if __name__ == '__main__':

    # 动态分配显存
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # pick a data set
    DATA_SET = 'mnist'

    # define splits
    DS_CONFIG = {
        # mnist data set parameters
        'mnist': {
            'batch_size': 128,
            'num_repeats': 2,
            'mdl_input_dims': [28, 28, 1]}
    }

    # load the data set
    TRAIN_SET, TEST_SET, SET_INFO = load(data_set_name=DATA_SET, **DS_CONFIG[DATA_SET])

    # configure the common model elements
    MDL_CONFIG = {
        # mist hyper-parameters
        'mnist': {
            'num_classes': SET_INFO.features['label'].num_classes,
            'learning_rate': 1e-3,
            'num_repeats': DS_CONFIG[DATA_SET]['num_repeats'],
            'save_dir': None},
    }

    # declare the model
    mdl = ClusterIIC(**MDL_CONFIG[DATA_SET])

    # train the model
    mdl.train(TRAIN_SET, TEST_SET, num_epochs=600)

    print('All done!')
    plt.show()
