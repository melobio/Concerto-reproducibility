import numpy as np
import os
import random
import tensorflow as tf


def tf_random_choice(input_length, num_indices_to_keep=3):
    # create uniform distribution over the sequence
    # for tf.__version__<1.11 use tf.random_uniform - no underscore in function name
    uniform_distribution = tf.random.uniform(
        shape=[input_length],
        minval=0,
        maxval=None,
        dtype=tf.float32,
        seed=None,
        name=None
    )
    # grab the indices of the greatest num_words_to_drop values from the distibution
    _, indices_to_keep = tf.nn.top_k(uniform_distribution, num_indices_to_keep)

    return indices_to_keep


def data_augment_process(x_feature_ids, x_value, y, rnd_mask_rate=0.005, min_augment_len: int = 1,
                         max_augment_len: int = 100):
    """
    数据增强
    :param x_feature_ids:
    :param x_value:
    :param y:
    :param rnd_mask_rate:
    :return:
    """
    # 拷贝原数据
    aug_feature_ids = x_feature_ids
    aug_value = x_value

    # + 1， 确保所有的值大于0
    c_mask = tf.cast(x_feature_ids + 1, tf.bool)
    feature_len = tf.reduce_sum(tf.cast(c_mask, tf.int32))
    aug_feature_len = tf.cast(feature_len, tf.float32)

    # 增强的数量
    aug_len = aug_feature_len * rnd_mask_rate
    aug_len = tf.cast(aug_len, tf.int32)
    aug_len = tf.math.maximum(aug_len, min_augment_len)
    aug_len = tf.math.minimum(aug_len, max_augment_len)

    choice_index = tf_random_choice(feature_len, aug_len)

    for word_pos in choice_index:
        # if word_pos == 0 or word_pos == feature_len - 1:
        #     continue

        # 随机增强
        dice = random.random()
        if dice < 0.5:
            # 50%的数据清零，模拟Dropout事件
            one_hot = tf.one_hot(word_pos, feature_len, dtype=tf.int64)
            aug_feature_ids = aug_feature_ids - aug_feature_ids[word_pos] * one_hot

            one_hot = tf.one_hot(word_pos, feature_len, dtype=tf.float32)
            aug_value = aug_value - aug_value[word_pos] * one_hot
        elif dice < 0.8:
            # 10%的表达量数据减少，模型拷贝数缩小
            one_hot = tf.one_hot(word_pos, feature_len, dtype=tf.float32)
            aug_value = aug_value - aug_value[word_pos] * one_hot * random.random()
        elif dice < 0.9:
            # 10%的表达量数据增大，模型拷贝数增加
            one_hot = tf.one_hot(word_pos, feature_len, dtype=tf.float32)
            aug_value = aug_value + aug_value[word_pos] * one_hot * random.random()

    return x_feature_ids, x_value, aug_feature_ids, aug_value


def augment_function(gene_feature, gene_value, label,domain,id):
    gene_feature_, gene_value_, aug_gene_feature_, aug_gene_value_ = data_augment_process(gene_feature, gene_value, label)

    return gene_feature_, gene_value_, aug_gene_feature_, aug_gene_value_, label, domain, id


def single_file_dataset_multi(input_file: list, name_to_features, sparse_to_denses):
    d = tf.data.TFRecordDataset(input_file)

    def single_example_parser(serialized_example):
        name_to_features = {
            'feature': tf.io.VarLenFeature(tf.int64),
            'value': tf.io.VarLenFeature(tf.float32),
            'batch': tf.io.FixedLenFeature([], tf.int64),
            'id': tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(serialized_example, name_to_features)

        feature = example['feature']
        value = example['value']
        domain = example['batch']
        id = example['id']

        feature = tf.sparse.to_dense(feature, default_value=0)
        value = tf.sparse.to_dense(value, default_value=0)

        return feature, value, domain,id

    d = d.map(single_example_parser)

    return d


def single_file_dataset_multi_supervised(input_file: list, name_to_features, sparse_to_denses):
    d = tf.data.TFRecordDataset(input_file)

    def single_example_parser(serialized_example):
        name_to_features = {
            'feature': tf.io.VarLenFeature(tf.int64),
            'value': tf.io.VarLenFeature(tf.float32),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'batch': tf.io.FixedLenFeature([], tf.int64),
            'id': tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(serialized_example, name_to_features)

        feature = example['feature']
        value = example['value']
        label = example['label']
        domain = example['batch']
        id = example['id']

        feature = tf.sparse.to_dense(feature, default_value=0)
        value = tf.sparse.to_dense(value, default_value=0)

        return feature, value, label, domain,id

    d = d.map(single_example_parser)

    return d



def create_classifier_dataset_multi(record_files: list,
                              batch_size: int,
                              is_training=True,
                              data_augment=False,
                              shuffle_size=100):
    """Creates input dataset from (tf)records files for train/eval."""
    name_to_features = {
        'feature': tf.io.VarLenFeature(tf.int64),
        'value': tf.io.VarLenFeature(tf.float32),
        'batch': tf.io.FixedLenFeature([], tf.int64),
        'id': tf.io.FixedLenFeature([], tf.string)
    }
    sparse_to_denses = ['feature', 'value', 'batch','id']

    # 读取记录
    dataset = single_file_dataset_multi(record_files, name_to_features, sparse_to_denses)

    if data_augment is True:
        dataset = dataset.map(augment_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.padded_batch(batch_size=batch_size,
                                       padded_shapes=([None], [None], [None], [None],[],[]),
                                       drop_remainder=True)
    else:
        dataset = dataset.padded_batch(batch_size=batch_size,
                                       padded_shapes=([None], [None],[],[]),
                                       drop_remainder=True)
    if is_training:
        # dataset = dataset.shuffle(shuffle_size)
        dataset = dataset.shuffle(shuffle_size, reshuffle_each_iteration=True)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def create_classifier_dataset_multi_supervised(record_files: list,
                              batch_size: int,
                              is_training=True,
                              data_augment=False,
                              shuffle_size=100):
    """Creates input dataset from (tf)records files for train/eval."""
    name_to_features = {
        'feature': tf.io.VarLenFeature(tf.int64),
        'value': tf.io.VarLenFeature(tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'batch': tf.io.FixedLenFeature([], tf.int64),
        'id': tf.io.FixedLenFeature([], tf.string)
    }
    sparse_to_denses = ['feature', 'value','label', 'batch','id']

    # 读取记录
    dataset = single_file_dataset_multi_supervised(record_files, name_to_features, sparse_to_denses)

    if data_augment is True:
        dataset = dataset.map(augment_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.padded_batch(batch_size=batch_size,
                                       padded_shapes=([None], [None], [None], [None],[],[],[]),
                                       drop_remainder=True)
    else:
        dataset = dataset.padded_batch(batch_size=batch_size,
                                       padded_shapes=([None], [None],[],[],[]),
                                       drop_remainder=True)
    if is_training:
        # dataset = dataset.shuffle(shuffle_size)
        dataset = dataset.shuffle(shuffle_size, reshuffle_each_iteration=True)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset