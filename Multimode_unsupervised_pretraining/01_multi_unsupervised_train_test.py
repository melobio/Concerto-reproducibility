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
from bgi.models.DeepSingleCell import EncoderHead, multi_embedding_attention_transfer,multi_embedding_attention_transfer_explainability
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
        '--task', type=str, default='train', metavar='NAME',
        choices=['train','test'],
        help='The task of the model to do: "train" or "test"')
    _argparser.add_argument(
        '--RNA-data', type=str, metavar='PATH', default='./tfrecord/RNA',
        help='Path to a file of RNA data')
    _argparser.add_argument(
        '--Protein-data', type=str, metavar='PATH', default='./tfrecord/Protein',
        help='Path to a file of Protein data')
    _argparser.add_argument(
        '--save', type=str, metavar='PATH',default='./weight',
        help='A path where the model should be saved / restored from')
    _argparser.add_argument(
        '--result', type=str, metavar='PATH',default='./result',
        help='A path where the test result should be saved / restored from')
    _argparser.add_argument(
        '--saved-weight', type=str, metavar='PATH', default='./saved_weight/epoch_9_ACC_0.7811089059786653.h5',
        help='saved weight')

    _argparser.add_argument(
        '--pretrain-epochs', type=int, default=1, metavar='INTEGER',
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

    temperature = _args.temperature
    batch_size = _args.batch_size
    task = _args.task
    save_path = _args.save
    saved_weight_path = _args.saved_weight
    save_result_path = _args.result
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_result_path):
        os.makedirs(save_result_path)
    pretrain_lr = _args.pretrain_lr
    pretrain_epochs = _args.pretrain_epochs
    dirname = os.getcwd()
    f = np.load(dirname + '/vocab_size_RNA.npz')
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(dirname + '/vocab_size_Protein.npz')
    vocab_size_Protein = int(f['vocab size'])

    if task == 'train':
        tf_list_path_RNA = _args.RNA_data
        tf_list_path_Protein = _args.Protein_data
        encode_network = multi_embedding_attention_transfer(multi_max_features=[vocab_size_RNA,vocab_size_Protein],#2000
                                                            mult_feature_names=['RNA','Protein'],
                                                            embedding_dims=128,
                                                            include_attention=True,
                                                            drop_rate=0.1,
                                                            head_1=128,
                                                            head_2=128,
                                                            head_3=128)
        decode_network = multi_embedding_attention_transfer(multi_max_features=[vocab_size_RNA,vocab_size_Protein],
                                                            mult_feature_names=['RNA','Protein'],
                                                            embedding_dims=128,
                                                            include_attention=False,
                                                            drop_rate=0.1,
                                                            head_1=128,
                                                            head_2=128,
                                                            head_3=128)
        mu_enc = EncoderHead()
        var_enc = EncoderHead()
        dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
        f = np.load(dirname + '/batch_dict.npz')
        batch_dict = list(f['batch'])

        train_source_list = os.listdir(os.path.join(tf_list_path_RNA))
        train_source_list_RNA = []
        train_source_list_Protein = []
        for i in train_source_list:
            train_source_list_RNA.append(os.path.join(tf_list_path_RNA,i))
            train_source_list_Protein.append(os.path.join(tf_list_path_Protein,i))

        cls_loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_cls_accuracy')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_cls_accuracy')
        opt_simclr = tf.keras.optimizers.Adam(learning_rate=pretrain_lr)

        for epoch in range(pretrain_epochs):
            for RNA_file, Protein_file in zip(train_source_list_RNA, train_source_list_Protein):
                print(RNA_file)
                print(Protein_file)
                train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                           batch_size=batch_size,
                                                           is_training=False,
                                                           data_augment=False,
                                                           shuffle_size=10000)
                train_db_Protein = create_classifier_dataset_multi([Protein_file],
                                                           batch_size=batch_size,
                                                           is_training=False,
                                                           data_augment=False,
                                                           shuffle_size=10000)
                train_loss.reset_states()
                train_cls_accuracy.reset_states()
                test_cls_accuracy.reset_states()
                step = 0
                for (source_features_RNA, source_values_RNA, source_label_RNA,
                     source_batch_RNA, source_id_RNA), \
                    (source_features_protein, source_values_protein,
                     source_label_Protein, source_batch_Protein, source_id_Protein) \
                        in (zip(train_db_RNA, train_db_Protein)):
                    step += 1
                    with tf.GradientTape() as tape:
                        z1 = encode_network([[source_features_RNA, source_features_protein],
                                             [source_values_RNA, source_values_protein]], training=True)
                        z2 = decode_network([source_values_RNA, source_values_protein], training=True)
                        ssl_loss = simclr_loss(z1, z2, temperature=temperature)
                        mu_1 = mu_enc(z1)
                        var_1 = tf.exp(var_enc(z1))
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

                    if step > 0 and step % batch_size == 0:
                        template = 'Epoch {}, step {}, simclr loss: {:0.4f}.'
                        print(template.format(epoch + 1,
                                              str(step),
                                              train_loss.result()))
            encode_network.save_weights(
                save_path + 'epoch{}.h5'.format(str(epoch)))

    if task == 'test':
        tf_list_path_RNA = _args.RNA_data
        tf_list_path_Protein = _args.Protein_data
        weight_path = save_path
        dirname = os.getcwd()
        f = np.load(dirname + '/label_dict.npz')
        label_dict_celltype = list(f['cell type'])
        f = np.load(dirname + '/batch_dict.npz')
        batch_dict = list(f['batch'])
        encode_network = multi_embedding_attention_transfer_explainability(multi_max_features=[vocab_size_RNA,vocab_size_Protein],  # 28241
                                                            mult_feature_names=['RNA','Protein'],
                                                            embedding_dims=128,
                                                            include_attention=True,
                                                            drop_rate=0.1,
                                                            head_1=128,
                                                            head_2=128,
                                                            head_3=128)

        train_source_list = os.listdir(os.path.join(tf_list_path_RNA))
        train_source_list_RNA = []
        train_source_list_Protein = []
        for i in train_source_list:
            train_source_list_RNA.append(os.path.join(tf_list_path_RNA, i))
            train_source_list_Protein.append(os.path.join(tf_list_path_Protein, i))

        weight_id_list = []
        weight_list = [f for f in os.listdir(weight_path) if 'h5' in f]
        if len(weight_list) !=0:
            for id in weight_list:
                id_1 = re.findall('.*epoch(.*).h.*', id)  # f1
                weight_id_list.append(int(id_1[0]))
            encode_network.load_weights(weight_path + 'epoch{}.h5'.format(max(weight_id_list)),by_name=True)
        else:
            encode_network.load_weights(saved_weight_path , by_name=True)
        target_features_all = []
        target_labels_all = []
        target_batchs_all = []
        target_id_all = []
        attention_output_all_RNA = []
        attention_output_all_Protein = []
        for RNA_file, Protein_file in zip(train_source_list_RNA, train_source_list_Protein):
            print(RNA_file)
            print(Protein_file)
            train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                           batch_size=batch_size,
                                                           is_training=False,
                                                           data_augment=False,
                                                           shuffle_size=10000)
            train_db_Protein = create_classifier_dataset_multi([Protein_file],
                                                               batch_size=batch_size,
                                                               is_training=False,
                                                               data_augment=False,
                                                               shuffle_size=10000)
            step = 0
            for (source_features_RNA, source_values_RNA, source_label_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_protein, source_values_protein,
                 source_label_Protein, source_batch_Protein, source_id_Protein) \
                    in (zip(train_db_RNA, train_db_Protein)):
                step += 1
                encode_output_query, attention_output = encode_network([[source_features_RNA, source_features_protein],
                                         [source_values_RNA, source_values_protein]], training=False)
                output = tf.nn.l2_normalize(encode_output_query, axis=-1)
                attention_output_all_RNA.extend(attention_output[0])
                attention_output_all_Protein.extend(attention_output[1])
                target_features_all.extend(output)
                target_labels_all.extend(source_label_RNA)
                target_batchs_all.extend(source_batch_RNA)
                target_id_all.extend(source_id_RNA.numpy())
                if step % batch_size == 0:
                    print(step)

        print('finished')
        target_features_all = np.array(target_features_all).astype('float32')
        target_labels_all = np.array(target_labels_all).astype('int32')
        target_batchs_all = np.array(target_batchs_all).astype('int32')
        attention_output_RNA = np.array(attention_output_all_RNA)
        print(attention_output_RNA.shape)
        attention_output_Protein = np.array(attention_output_all_Protein)
        print(attention_output_Protein.shape)
        batch_list_id = []
        label_list_id = []
        for j in target_labels_all:
            label_list_id.append(label_dict_celltype[int(j)])
        for j in target_batchs_all:
            batch_list_id.append(batch_dict[int(j)])
        query_emb = target_features_all
        query_anndata = sc.AnnData(query_emb)
        query_anndata.obs['label'] = label_list_id
        query_anndata.obs['batch'] = batch_list_id
        query_anndata.obs_names = target_id_all
        query_anndata.write_h5ad(save_result_path + '/multi_embedding.h5ad')
        save_dict = {'attention_output_RNA': attention_output_RNA,
                     'attention_output_Protein': attention_output_Protein}
        np.savez_compressed(save_result_path + '/multi_attention.npz', **save_dict)