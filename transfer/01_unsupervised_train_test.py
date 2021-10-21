import numpy as np
import os
import random
import time
import json
import pandas as pd
import tensorflow as tf
import csv
import sys
import scanpy as sc
from collections import Counter
import argparse
import matplotlib.pyplot as plt
import matplotlib
sys.path.append("../")
from bgi.utils.data_utils import *
from bgi.models.DeepSingleCell import EncoderHead, multi_embedding_attention_transfer
from bgi.metrics.clustering_metrics import *
from bgi.losses.contrastive_loss import simclr_loss
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import f1_score, accuracy_score
import re


if __name__ == '__main__':
    _argparser = argparse.ArgumentParser(
        description='train/test model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _argparser.add_argument(
        '--data', type=str, metavar='PATH', default='./tfrecord',
        help='The path of tfrecord fold')
    _argparser.add_argument(
        '--save', type=str, metavar='PATH', default='./weight',
        help='A path where the model should be saved / restored from')
    _argparser.add_argument(
        '--result', type=str,metavar='PATH', default='./result',
        help='A path where the test result should be saved / restored from')
    _argparser.add_argument(
        '--fineturning', type=bool, default=False,
        help='whether to do fineturning ')

    _argparser.add_argument(
        '--pretrain-epochs', type=int, default=3, metavar='INTEGER',
        help='The number of epochs to pretrain')
    _argparser.add_argument(
        '--pretrain-lr', type=float, default=1e-5, metavar='FLOAT',
        help='Pretrain learning rate')
    _argparser.add_argument(
        '--batch-size', type=int, default=32, metavar='INTEGER',
        help='Training batch size')
    _argparser.add_argument(
        '--CUDA-VISIBLE-DEVICES', type=int, default=0, metavar='INTEGER',
        help='CUDA_VISIBLE_DEVICES')
    _argparser.add_argument(
        '--temperature', type=int, default=0.1, metavar='FLOAT',
        help='Number of temperature')

    _args = _argparser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.CUDA_VISIBLE_DEVICES)
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf_list_path = _args.data
    temperature = _args.temperature
    batch_size = _args.batch_size
    fineturning = _args.fineturning
    save_path = _args.save
    save_result_path = _args.result
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_result_path):
        os.makedirs(save_result_path)
    pretrain_lr = _args.pretrain_lr
    pretrain_epochs = _args.pretrain_epochs
    dirname = os.getcwd()
    f = np.load(dirname + '/label_dict_query.npz')
    label_dict_query = list(f['cell type'])
    f = np.load(dirname + '/label_dict_ref.npz')
    label_dict_ref = list(f['cell type'])
    f = np.load(dirname + '/batch_dict_query.npz')
    batch_dict_query = list(f['batch'])
    f = np.load(dirname + '/batch_dict_ref.npz')
    batch_dict_ref = list(f['batch'])

    f = np.load(dirname + '/vocab_size.npz')
    vocab_size = int(f['vocab size'])
    encode_network = multi_embedding_attention_transfer(multi_max_features=[vocab_size],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=0.1,
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    decode_network = multi_embedding_attention_transfer(multi_max_features=[vocab_size],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=0.1,
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    mu_enc = EncoderHead()
    var_enc = EncoderHead()
    tf_list_1 = os.listdir(os.path.join(tf_list_path, 'ref'))
    tf_list_2 = os.listdir(os.path.join(tf_list_path, 'query'))
    train_target_list = []
    for i in tf_list_2:
        train_target_list.append(os.path.join(tf_list_path,'query', i))
    train_source_list = []
    for i in tf_list_1:
        train_source_list.append(os.path.join(tf_list_path,'ref', i))
    cls_loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_cls_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_cls_accuracy')
    #opt_simclr = tf.keras.optimizers.Adam(learning_rate=pretrain_lr)
    total_update_steps = 300 * pretrain_epochs
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(1e-4, total_update_steps, 1e-6, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # pretrain
    for epoch in range(pretrain_epochs):
        np.random.shuffle(train_source_list)
        for file in train_source_list:
            print(file)
            train_db = create_classifier_dataset_multi([file],
                                                       batch_size=batch_size,
                                                       is_training=True,
                                                       data_augment=False,
                                                       shuffle_size=10000)

            train_loss.reset_states()
            train_cls_accuracy.reset_states()
            test_cls_accuracy.reset_states()
            for step, (source_features, source_values, source_label, source_batch, source_id) in enumerate(train_db):
                # enumerate
                with tf.GradientTape() as tape:
                    # center loss
                    z1 = encode_network([source_features, source_values], training=True)
                    z2 = decode_network([source_values], training=True)
                    mu_1 = mu_enc(z1)
                    var_1 = tf.exp(var_enc(z1))
                    ssl_loss = simclr_loss(z1, z2,temperature = temperature)
                    loss = tf.keras.losses.kullback_leibler_divergence(mu_1, var_1) + ssl_loss
                    train_loss(loss)

                variables = [encode_network.trainable_variables,
                             decode_network.trainable_variables,
                             mu_enc.trainable_variables,
                             var_enc.trainable_variables
                             ]
                grads = tape.gradient(loss, variables)
                for grad, var in zip(grads, variables):
                    opt_simclr.apply_gradients(zip(grad, var))

                if step > 0 and step % 5 == 0:
                    template = 'Epoch {}, step {}, simclr loss: {:0.4f}.'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result()))

    if fineturning is True:
        print('start fineturning')
        for epoch in range(1):
            for file in train_target_list:
                print(file)
                train_db = create_classifier_dataset_multi([file],
                                                           batch_size=batch_size,
                                                           is_training=True,
                                                           data_augment=False,
                                                           shuffle_size=10000)

                train_loss.reset_states()
                train_cls_accuracy.reset_states()
                test_cls_accuracy.reset_states()
                for step, (source_features, source_values, source_label, source_batch, source_id) in enumerate(
                        train_db):
                    # enumerate
                    with tf.GradientTape() as tape:
                        # center loss
                        z1 = encode_network([source_features, source_values], training=True)
                        z2 = decode_network([source_values], training=True)
                        mu_1 = mu_enc(z1)
                        var_1 = tf.exp(var_enc(z1))
                        ssl_loss = simclr_loss(z1, z2, temperature=temperature)
                        loss = tf.keras.losses.kullback_leibler_divergence(mu_1, var_1) + ssl_loss
                        train_loss(loss)

                    variables = [encode_network.trainable_variables,
                                 decode_network.trainable_variables,
                                 mu_enc.trainable_variables,
                                 var_enc.trainable_variables
                                 ]
                    grads = tape.gradient(loss, variables)
                    for grad, var in zip(grads, variables):
                        opt_simclr.apply_gradients(zip(grad, var))

                    if step > 0 and step % 5 == 0:
                        template = 'Epoch {}, step {}, simclr loss: {:0.4f}.'
                        print(template.format(epoch + 1,
                                              str(step),
                                              train_loss.result()))

    else:
        print('no finefurning')
    source_data_true = []
    source_data_batch = []
    source_data_feature = []
    source_data_id = []
    for file in train_source_list:
        print(file)
        valid_db = create_classifier_dataset_multi([file],
                                                   batch_size=batch_size,
                                                   is_training=False,
                                                   data_augment=False,
                                                   shuffle_size=10000)
        for step, (target_features, target_values, target_labels,target_batch,target_id) in enumerate(valid_db):
            output = encode_network([target_features, target_values], training=False)
            output = tf.nn.l2_normalize(output, axis=-1)
            source_data_feature.extend(output)
            source_data_true.extend(target_labels)
            source_data_batch.extend(target_batch)
            source_data_id.extend(target_id.numpy())

    target_data_true = []
    target_data_batch = []
    target_data_feature = []
    target_data_id = []
    for file in train_target_list:
        print(file)
        valid_db = create_classifier_dataset_multi([file],
                                                   batch_size=batch_size,
                                                   is_training=False,
                                                   data_augment=False,
                                                   shuffle_size=10000)
        for step, (target_features, target_values, target_labels,target_batch,target_id) in enumerate(valid_db):
            output = encode_network([target_features, target_values], training=False)
            output = tf.nn.l2_normalize(output, axis=-1)
            target_data_feature.extend(output)
            target_data_true.extend(target_labels)
            target_data_batch.extend(target_batch)
            target_data_id.extend(target_id.numpy())

    source_data_true = np.array(source_data_true, dtype=np.int)
    source_data_batch = np.array(source_data_batch)
    source_data_feature = np.array(source_data_feature)
    target_data_true = np.array(target_data_true, dtype=np.int)
    target_data_batch = np.array(target_data_batch)
    target_data_feature = np.array(target_data_feature)
    print('target feature shape', target_data_feature.shape)
    label_query_id = []
    label_ref_id = []
    batch_query_id = []
    batch_ref_id = []
    for j in source_data_true:
        label_ref_id.append(label_dict_ref[int(j)])
    for j in target_data_true:
        label_query_id.append(label_dict_query[int(j)])
    for j in source_data_batch:
        batch_ref_id.append(batch_dict_ref[int(j)])
    for j in target_data_batch:
        batch_query_id.append(batch_dict_query[int(j)])
    ref_adata = sc.AnnData(source_data_feature)
    ref_adata.obs['label'] = label_ref_id
    ref_adata.obs['batch'] = batch_ref_id
    query_adata = sc.AnnData(target_data_feature)
    query_adata.obs['label'] = label_query_id
    query_adata.obs['batch'] = batch_query_id
    ref_adata.write_h5ad(save_result_path + '/ref_adata.h5ad')
    query_adata.write_h5ad(save_result_path + '/query_adata.h5ad')
    sw = silhouette_score(target_data_feature,target_data_true)
    print('sw:{}'.format(sw))
    print('start clustering')
    with open(save_result_path + '/result.csv','w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['dataset','algo','knn','accuracy','median F1 score','micro F1 score',
                         'macro F1 score','nmi','ari','sw','time(s)',])
    for k in [3, 5, 10, 15, 20, 25]:
        time1 = time.time()
        target_labels, target_neighbor = knn_classifier(source_data_feature, source_data_true,
                                                                     target_data_feature,
                                                                     target_data_true, k=k, num_chunks=100)
        time2 = time.time()
        test_time = time2 - time1
        target_label_seen_list = []
        for target_label_index in range(len(target_labels)):
            target_label = target_labels[target_label_index]
            target_label_str = label_dict_query[target_label]
            if target_label_str in label_dict_ref:
                target_label_seen_list.append(target_label_index)
        target_labels_seen = target_labels[target_label_seen_list]
        target_neighbor_seen = target_neighbor[target_label_seen_list]
        target_labels_seen_str = []
        for i in target_labels_seen:
            target_labels_seen_str.append(label_dict_query[i])
        target_neighbor_seen_str = []
        for i in target_neighbor_seen:
            target_neighbor_seen_str.append(label_dict_ref[i])

        acc = np.round(accuracy_score(target_labels_seen_str, target_neighbor_seen_str), 5)
        f1_median, f1_macro, f1_micro = cal_F1(target_labels_seen_str, target_neighbor_seen_str)
        nmi = np.round(normalized_mutual_info_score(target_labels, target_neighbor), 5)
        ari = np.round(adjusted_rand_score(target_labels, target_neighbor), 5)
        print('Target(k= %.1f ): acc = %.5f, nmi = %.5f, ari = %.5f' % (k, acc, nmi, ari), '.')
        print('k:{} f1_median:{} f1_macro:{} f1_micro:{}'.format(str(k), str(f1_median), str(f1_macro),
                                                                 str(f1_micro)))
        encode_network.save_weights(
            save_path + '/k{}_acc{}_nmi{}_ari{}_f1{}.h5'.format(str(k), str(acc), str(nmi), str(ari),
                                                               str(f1_median)))
        row_data = ['HP(indrop)', 'Concerto', str(k), str(acc),str(f1_median),str(f1_micro),str(f1_macro),
                    str(nmi), str(ari), str(sw),str(test_time)]
        with open(save_result_path + '/result.csv', 'a+', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(row_data)
