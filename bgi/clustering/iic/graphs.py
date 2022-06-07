import tensorflow as tf

# set trainable variable initialization routines
KERNEL_INIT = tf.keras.initializers.he_uniform()
WEIGHT_INIT = tf.random_normal_initializer(mean=0.0, stddev=0.01)
BIAS_INIT = tf.constant_initializer(0.0)


def convolution_layer(kernel_size, num_out_channels, activation, batch_norm, is_training, name):
    """
    :param x: input data
    :param kernel_size: convolution kernel size
    :param num_out_channels: number of output channels
    :param activation: non-linearity
    :param batch_norm: whether to use batch norm
    :param is_training: whether we are training or testing (used by batch normalization)
    :param name: variable scope name (empowers variable reuse)
    :return: layer output
    """

    layers = []
    # run convolution layer
    conv_layer = tf.keras.layers.Conv2D(
        filters=num_out_channels,
        kernel_size=[kernel_size] * 2,
        strides=[1, 1],
        padding='same',
        activation=None,
        use_bias=True,
        kernel_initializer=KERNEL_INIT,
        bias_initializer=BIAS_INIT,
        name=name)
    layers.append(conv_layer)

    # run batch norm if specified
    if batch_norm:
        bn_layer = tf.keras.layers.BatchNormalization(name=name + '_bn')
        layers.append(bn_layer)

    # run activation
    act_layer = tf.keras.layers.Activation(activation=activation)
    layers.append(act_layer)

    return layers


def max_pooling_layer(pool_size, strides, name):
    """
    :param x: input data
    :param pool_size: pooling kernel size
    :param strides: pooling stride length
    :param name: variable scope name (empowers variable reuse)
    :return: layer output
    """
    layers = []
    # run max pooling
    maxpool_layer = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=strides, padding='same', name=name)
    layers.append(maxpool_layer)

    return layers


def fully_connected_layer(num_outputs, activation, is_training, name):
    """
    :param x: input data
    :param num_outputs: number of outputs
    :param activation: non-linearity
    :param is_training: whether we are training or testing (used by batch normalization)
    :param name: variable scope name (empowers variable reuse)
    :return: layer output
    """
    layers = []
    # run dense layer
    dense_layer = tf.keras.layers.Dense(
        units=num_outputs,
        activation=None,
        use_bias=True,
        kernel_initializer=WEIGHT_INIT,
        bias_initializer=BIAS_INIT,
        name=name)
    layers.append(dense_layer)

    # run batch norm
    bn_layer = tf.keras.layers.BatchNormalization(activation_fn=activation)
    layers.append(bn_layer)

    return layers


# ResNet with BasicBlock (adapted to CIFAR-10)
class BasicNetwork(tf.keras.Model):
    def __init__(self,
                 config='B',
                 batch_norm=True,
                 fan_out_init=64,
                 n_classes=10,
                 include_head=True,
                 channels=3,
                 training=True):
        """
        :param config: character {A, B, C} that matches architecture in IIC supplementary materials
        :param fan_out_init: initial fan out (paper uses 64, but can be reduced for memory constrained systems)
        """
        super(BasicNetwork, self).__init__()

        # set activation
        self.activation = 'relu'

        # save architectural details
        self.config = config
        self.batch_norm = batch_norm
        self.fan_out_init = fan_out_init
        self.include_head = include_head
        self.n_classes = n_classes
        self.channels = channels
        self.training = training

        self.compute_layers = []
        # layer 1
        num_out_channels = self.fan_out_init

        output_layers = convolution_layer(kernel_size=5, num_out_channels=num_out_channels,
                                          activation=self.activation,
                                          batch_norm=self.batch_norm, is_training=self.training, name='conv1')
        self.compute_layers.extend(output_layers)
        output_layers = max_pooling_layer(pool_size=2, strides=2, name='pool1')
        self.compute_layers.extend(output_layers)

        # layer 2
        num_out_channels *= 2
        output_layers = convolution_layer(kernel_size=5, num_out_channels=num_out_channels,
                                          activation=self.activation,
                                          batch_norm=self.batch_norm, is_training=self.training, name='conv2')
        self.compute_layers.extend(output_layers)
        output_layers = max_pooling_layer(pool_size=2, strides=2, name='pool2')
        self.compute_layers.extend(output_layers)

        # layer 3
        num_out_channels *= 2
        output_layers = convolution_layer(kernel_size=5, num_out_channels=num_out_channels,
                                          activation=self.activation,
                                          batch_norm=self.batch_norm, is_training=self.training, name='conv3')
        self.compute_layers.extend(output_layers)
        output_layers = max_pooling_layer(pool_size=2, strides=2, name='pool3')
        self.compute_layers.extend(output_layers)

        # layer 4
        num_out_channels *= 2
        output_layers = convolution_layer(kernel_size=5, num_out_channels=num_out_channels,
                                          activation=self.activation,
                                          batch_norm=self.batch_norm, is_training=self.training, name='conv4')
        self.compute_layers.extend(output_layers)

        # flatten
        flatten_layer = tf.keras.layers.Flatten()
        self.compute_layers.append(flatten_layer)

    def call(self, x):
        """
        :param x: input data
        :param is_training: whether we are training or testing (used by batch normalization)
        :return: output of VGG
        """
        # layer 1
        # num_out_channels = self.fan_out_init

        # input = x  # tf.keras.layers.Input(shape=(None, None, channels), dtype=tf.float32)
        # output = convolution_layer(x=input, kernel_size=5, num_out_channels=num_out_channels,
        #                            activation=self.activation,
        #                            batch_norm=self.batch_norm, is_training=self.training, name='conv1')
        # output = max_pooling_layer(x=output, pool_size=2, strides=2, name='pool1')
        #
        # # layer 2
        # num_out_channels *= 2
        # output = convolution_layer(x=output, kernel_size=5, num_out_channels=num_out_channels,
        #                            activation=self.activation,
        #                            batch_norm=self.batch_norm, is_training=self.training, name='conv2')
        # output = max_pooling_layer(x=output, pool_size=2, strides=2, name='pool2')
        #
        # # layer 3
        # num_out_channels *= 2
        # output = convolution_layer(x=output, kernel_size=5, num_out_channels=num_out_channels,
        #                            activation=self.activation,
        #                            batch_norm=self.batch_norm, is_training=self.training, name='conv3')
        # output = max_pooling_layer(x=output, pool_size=2, strides=2, name='pool3')
        #
        # # layer 4
        # num_out_channels *= 2
        # output = convolution_layer(x=output, kernel_size=5, num_out_channels=num_out_channels,
        #                            activation=self.activation,
        #                            batch_norm=self.batch_norm, is_training=self.training, name='conv4')
        #
        # # flatten
        # output = tf.keras.layers.Flatten()(output)

        output = x
        for layer in self.compute_layers:
            output = layer(output)

        return output


def build_network(
        config='B',
        batch_norm=True,
        fan_out_init=64,
        n_classes=10,
        include_head=True,
        channels=1,
        training=True):
    """
    :param config: character {A, B, C} that matches architecture in IIC supplementary materials
    :param fan_out_init: initial fan out (paper uses 64, but can be reduced for memory constrained systems)
    """
    #super(BasicNetwork, self).__init__()

    # set activation
    activation = 'relu'

    # save architectural details
    # config = config
    # batch_norm = batch_norm
    # fan_out_init = fan_out_init
    # include_head = include_head
    # n_classes = n_classes
    # channels = channels
    # training = training

    compute_layers = []
    # layer 1
    num_out_channels = fan_out_init

    output_layers = convolution_layer(kernel_size=5, num_out_channels=num_out_channels,
                                      activation=activation,
                                      batch_norm=batch_norm, is_training=training, name='conv1')
    compute_layers.extend(output_layers)
    output_layers = max_pooling_layer(pool_size=2, strides=2, name='pool1')
    compute_layers.extend(output_layers)

    # layer 2
    num_out_channels *= 2
    output_layers = convolution_layer(kernel_size=5, num_out_channels=num_out_channels,
                                      activation=activation,
                                      batch_norm=batch_norm, is_training=training, name='conv2')
    compute_layers.extend(output_layers)
    output_layers = max_pooling_layer(pool_size=2, strides=2, name='pool2')
    compute_layers.extend(output_layers)

    # # layer 3
    # num_out_channels *= 2
    # output_layers = convolution_layer(kernel_size=5, num_out_channels=num_out_channels,
    #                                   activation=activation,
    #                                   batch_norm=batch_norm, is_training=training, name='conv3')
    # compute_layers.extend(output_layers)
    # output_layers = max_pooling_layer(pool_size=2, strides=2, name='pool3')
    # compute_layers.extend(output_layers)
    #
    # # layer 4
    # num_out_channels *= 2
    # output_layers = convolution_layer(kernel_size=5, num_out_channels=num_out_channels,
    #                                   activation=activation,
    #                                   batch_norm=batch_norm, is_training=training, name='conv4')
    # compute_layers.extend(output_layers)

    # flatten
    #flatten_layer = tf.keras.layers.GlobalAveragePooling2D() #.Flatten()
    #compute_layers.append(flatten_layer)

    input = tf.keras.layers.Input(shape=(28,28,channels))
    output = input
    for layer in compute_layers:
        output = layer(output)

    model = tf.keras.models.Model(inputs=input, outputs=output)
    return model


if __name__ == '__main__':
    import numpy as np

    x_data = np.zeros((100, 28, 28, 1))
    network = build_network(config='B', batch_norm=True, fan_out_init=64)

    # print("x_data: ", x_data.shape)
    # x = graph.evaluate(x_data, is_training=True, include_head=False)
    # output = network(x_data)
    # print(output)
    print(network.summary())
