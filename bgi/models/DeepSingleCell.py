import tensorflow as tf
from tensorflow.keras.models import Model  # layers, Sequential, optimizers, losses, metrics, datasets
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.layers import GlobalAveragePooling1D,Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Add
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers, losses, metrics, datasets
from bgi.layers.attention import AttentionWithContext



def multi_embedding_attention_transfer(supvised_train: bool = False,
                                    scan_train: bool = False,
                                    multi_max_features: list = [40000],
                                    mult_feature_names: list = ['Gene'],
                                    embedding_dims=128,
                                    num_classes=5,
                                    activation='softmax',
                                    head_1=128,
                                    head_2=128,
                                    head_3=128,
                                    drop_rate=0.05,
                                    include_attention: bool = False,
                                    use_bias=True,
                                    ):
    assert len(multi_max_features) == len(mult_feature_names)

    # 特征索引
    x_feature_inputs = []
    # 特征值
    x_value_inputs = []
    # 特征向量
    embeddings = []
    features = []
    weight_output_all = []
    if include_attention == True:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入
            feature_input = Input(shape=(None,), name='Input-{}-Feature'.format(name))
            value_input = Input(shape=(None,), name='Input-{}-Value'.format(name), dtype='float')
            x_feature_inputs.append(feature_input)
            x_value_inputs.append(value_input)

            # 向量
            embedding = Embedding(max_length, embedding_dims, input_length=None, name='{}-Embedding'.format(name))(
                feature_input)

            # 向量 * 特征值
            sparse_value = tf.expand_dims(value_input, 2, name='{}-Expend-Dims'.format(name))
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(sparse_value)
            x = tf.multiply(embedding, sparse_value, name='{}-Multiply'.format(name))
            # x = BatchNormalization(name='{}-BN-2'.format(name))(x)

            # # Attention
            weight_output,a = AttentionWithContext()(x)
            x = K.tanh(K.sum(weight_output, axis=1))

            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
            #weight_output_all.append(a)
        inputs = []
        inputs.append(x_feature_inputs)
        inputs.append(x_value_inputs)

    else:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入

            value_input = Input(shape=(max_length,), name='Input-{}-Value'.format(name), dtype='float')

            x_value_inputs.append(value_input)

            # 向量 * 特征值
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(value_input)

            x = Dense(head_1, name='{}-projection-0'.format(name), activation='relu')(sparse_value)


            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
        inputs = []
        inputs.append(x_value_inputs)
    # Concatenate
    if len(features) > 1:
    #feature = concatenate(features)
        feature = Add()([features[0],features[1]])
    else:
        feature = features[0]
    dropout = Dropout(rate=drop_rate)(feature)
    output = Dense(head_1, name='projection-1', activation='relu')(dropout)

    # inputs = []
    # inputs.append(x_feature_inputs)
    # inputs.append(x_value_inputs)
    return tf.keras.Model(inputs=inputs, outputs=output)


def multi_embedding_attention_transfer_1(supvised_train: bool = False,
                                    scan_train: bool = False,
                                    multi_max_features: list = [40000],
                                    mult_feature_names: list = ['Gene'],
                                    embedding_dims=128,
                                    num_classes=5,
                                    activation='softmax',
                                    head_1=128,
                                    head_2=128,
                                    head_3=128,
                                    drop_rate=0.05,
                                    include_attention: bool = False,
                                    use_bias=True,
                                    ):
    assert len(multi_max_features) == len(mult_feature_names)

    # 特征索引
    x_feature_inputs = []
    # 特征值
    x_value_inputs = []
    # 特征向量
    embeddings = []
    features = []
    weight_output_all = []
    if include_attention == True:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入
            feature_input = Input(shape=(None,), name='Input-{}-Feature'.format(name))
            value_input = Input(shape=(None,), name='Input-{}-Value'.format(name), dtype='float')
            x_feature_inputs.append(feature_input)
            x_value_inputs.append(value_input)

            # 向量
            embedding = Embedding(max_length, embedding_dims, input_length=None, name='{}-Embedding'.format(name))(
                feature_input)

            # 向量 * 特征值
            sparse_value = tf.expand_dims(value_input, 2, name='{}-Expend-Dims'.format(name))
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(sparse_value)
            x = tf.multiply(embedding, sparse_value, name='{}-Multiply'.format(name))
            # x = BatchNormalization(name='{}-BN-2'.format(name))(x)

            # # Attention
            weight_output,a = AttentionWithContext()(x)
            x = K.tanh(K.sum(weight_output, axis=1))

            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
            weight_output_all.append(a)
        inputs = []
        inputs.append(x_feature_inputs)
        inputs.append(x_value_inputs)

    else:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入

            value_input = Input(shape=(max_length,), name='Input-{}-Value'.format(name), dtype='float')

            x_value_inputs.append(value_input)

            # 向量 * 特征值
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(value_input)

            x = Dense(head_1, name='{}-projection-0'.format(name), activation='relu')(sparse_value)


            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
        inputs = []
        inputs.append(x_value_inputs)
    # Concatenate
    if len(features) > 1:
    #feature = concatenate(features)
        feature = Add()([features[0],features[1]])
    else:
        feature = features[0]
    dropout = Dropout(rate=drop_rate)(feature)
    output = Dense(head_1, name='projection-1', activation='relu')(dropout)

    # inputs = []
    # inputs.append(x_feature_inputs)
    # inputs.append(x_value_inputs)
    return tf.keras.Model(inputs=inputs, outputs=[output,weight_output_all])

def multi_embedding_attention_transfer_explainability(supvised_train: bool = False,
                                    scan_train: bool = False,
                                    multi_max_features: list = [40000],
                                    mult_feature_names: list = ['Gene'],
                                    embedding_dims=128,
                                    num_classes=5,
                                    activation='softmax',
                                    head_1=128,
                                    head_2=128,
                                    head_3=128,
                                    drop_rate=0.05,
                                    include_attention: bool = False,
                                    use_bias=True,
                                    ):
    assert len(multi_max_features) == len(mult_feature_names)

    # 特征索引
    x_feature_inputs = []
    # 特征值
    x_value_inputs = []
    # 特征向量
    embeddings = []
    features = []
    weight_output_all = []
    if include_attention == True:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入
            feature_input = Input(shape=(None,), name='Input-{}-Feature'.format(name))
            value_input = Input(shape=(None,), name='Input-{}-Value'.format(name), dtype='float')
            x_feature_inputs.append(feature_input)
            x_value_inputs.append(value_input)

            # 向量
            embedding = Embedding(max_length, embedding_dims, input_length=None, name='{}-Embedding'.format(name))(
                feature_input)

            # 向量 * 特征值
            sparse_value = tf.expand_dims(value_input, 2, name='{}-Expend-Dims'.format(name))
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(sparse_value)
            x = tf.multiply(embedding, sparse_value, name='{}-Multiply'.format(name))
            # x = BatchNormalization(name='{}-BN-2'.format(name))(x)

            # # Attention
            weight_output,a = AttentionWithContext()(x)
            x = K.tanh(K.sum(weight_output, axis=1))

            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
            weight_output_all.append(a)
        inputs = []
        inputs.append(x_feature_inputs)
        inputs.append(x_value_inputs)

    else:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入

            value_input = Input(shape=(max_length,), name='Input-{}-Value'.format(name), dtype='float')

            x_value_inputs.append(value_input)

            # 向量 * 特征值
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(value_input)

            x = Dense(head_1, name='{}-projection-0'.format(name), activation='relu')(sparse_value)


            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
        inputs = []
        inputs.append(x_value_inputs)
    # Concatenate
    if len(features) > 1:
    #feature = concatenate(features)
        feature = Add()([features[0],features[1]])
    else:
        feature = features[0]
    dropout = Dropout(rate=drop_rate)(feature)
    output = Dense(head_1, name='projection-1', activation='relu')(dropout)

    # inputs = []
    # inputs.append(x_feature_inputs)
    # inputs.append(x_value_inputs)
    return tf.keras.Model(inputs=inputs, outputs=[output,weight_output_all])



class EncoderHead(tf.keras.Model):

    def __init__(self, hidden_size=128, dropout=0.05):
        super(EncoderHead, self).__init__()
        # self.num_classes = num_classes
        self.feature_fc1 = tf.keras.layers.Dense(units=hidden_size, activation='relu')
        self.feature_fc2 = tf.keras.layers.Dense(units=hidden_size, activation='relu')
        self.feature_bn1 = tf.keras.layers.BatchNormalization()
        self.feature_bn2 = tf.keras.layers.BatchNormalization()
        self.feature_dropout1 = tf.keras.layers.Dropout(rate=dropout)
        self.feature_dropout2 = tf.keras.layers.Dropout(rate=dropout)

    def call(self, input):
        x = input
        feature_output = self.feature_fc1(x)
        feature_output = self.feature_bn1(feature_output)
        feature_output = self.feature_dropout1(feature_output)
        feature_output = self.feature_fc2(feature_output)

        return feature_output