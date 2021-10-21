import tensorflow as tf
import numpy as np
import h5py
import os
import scanpy as sc
import argparse
import pandas as pd
import copy
from collections import Counter
from scipy.sparse import issparse
import scipy
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    #return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(value)))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


def preprocessing_rna(
        adata,
        min_features: int = 600,
        min_cells: int = 3,
        target_sum: int = 10000,
        n_top_features=2000,  # or gene list
        chunk_size: int = 20000,
        is_hvg = False,
        batch_key = 'batch',
        log=True
):
    if min_features is None: min_features = 600
    if n_top_features is None: n_top_features = 40000

    if not issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)

    # adata = adata[:, [gene for gene in adata.var_names
    #                   if not str(gene).startswith(tuple(['ERCC', 'MT-', 'mt-']))]]

    sc.pp.filter_cells(adata, min_genes=min_features)

    #sc.pp.filter_genes(adata, min_cells=min_cells)

    sc.pp.normalize_total(adata, target_sum=target_sum)

    sc.pp.log1p(adata)
    if is_hvg == True:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_features, batch_key=batch_key, inplace=False, subset=True)

    print('Processed dataset shape: {}'.format(adata.shape))
    return adata

def serialize_example_batch(x_feature, x_weight, y_label,y_batch,x_id):

    feature = {
        'feature': _int64_feature(x_feature),
        'value': _float_feature(x_weight),
        'label': _int64_feature(y_label),
        'batch': _int64_feature(y_batch),
        'id': _bytes_feature(x_id)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def create_tfrecord(source_file, label_dict, batchs, tfrecord_file, zero_filter=False, norm=False,label_key = 'label',batch_key = 'batch'):
    x_data = source_file.X.toarray()
    y_data = source_file.obs[label_key].tolist()
    batch_data = source_file.obs[batch_key].tolist()
    obs_name_list = source_file.obs_names.tolist()
    label_number = []
    for j in range(len(y_data)):
        cell_type = y_data[j]
        place = label_dict.index(cell_type)
        label_number.append(place)

    batch_number = []
    for j in range(len(batch_data)):
        batch = batch_data[j]
        place = batchs.index(batch)
        batch_number.append(place)

    counter = 0
    batch_examples = {}
    for x, y, batch,k in zip(x_data, label_number, batch_number,obs_name_list):
        if zero_filter is False:
            x = x + 10e-6
            indexes = np.where(x >= 0.0)
        else:
            indexes = np.where(x > 0.0)
        values = x[indexes]

        features = np.array(indexes)
        features = np.reshape(features, (features.shape[1]))
        values = np.array(values, dtype=np.float)
        # values = values / np.linalg.norm(values)

        if len(features) > 50:

            if batch not in batch_examples:
                batch_examples[batch] = []

            y = np.array([int(y)])

            # batchs = np.ones_like(features) * batch

            example = serialize_example_batch(features, values, y, np.array([int(batch)]),k)
            batch_examples[batch].append(example)

            counter += 1
            if counter % 100 == 0:
                print('counter: {} shape: {}, batch: {}'.format(counter, features.shape, batch))

                print(x)
                print(values)
                print("batchs: ", batchs)
                print()

    for item in batch_examples.items():
        batch = item[0]
        examples = item[1]
        if zero_filter is False:
            #file = tfrecord_file
            file = tfrecord_file.replace('.tfrecord', '_{}.tfrecord'.format(batch))
        else:
            if norm is False:
                #file = tfrecord_file
                file = tfrecord_file.replace('.tfrecord', '_{}_no_zero_no_norm.tfrecord'.format(batch))
            else:
                #file = tfrecord_file
                file = tfrecord_file.replace('.tfrecord', '_{}_no_zero.tfrecord'.format(batch))
        with tf.io.TFRecordWriter(file) as writer:
            for example in examples:
                writer.write(example)
    save_dict = {'vocab size': len(features)}
    np.savez_compressed('vocab_size.npz', **save_dict)


if __name__ == '__main__':
    _argparser = argparse.ArgumentParser(
        description='create tfrecord file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _argparser.add_argument(
        '--data', type=str, default='./data/hp_rmSchwann_commonCelltype.loom', metavar='PATH',
        help='A path of reference file.')
    _argparser.add_argument(
        '--output', type=str, default='./tfrecord', metavar='PATH',
        help='A path of saving tfrecord')
    _argparser.add_argument(
        '--hvg', type=bool, default=False,
        help='whether to do hvg ')
    _argparser.add_argument(
        '--hvg-number', type=int, default=2000, metavar='INTEGER',
        help='the number of hvg gene')
    _argparser.add_argument(
        '--batch', type=str, default='tech', metavar='NAME',
        help='batch key name')
    _argparser.add_argument(
        '--label', type=str, default='celltype', metavar='NAME',
        help='label key name')

    _args = _argparser.parse_args()
    path_data = _args.data
    save_path = _args.output
    is_hvg = _args.hvg
    n_top_features = _args.hvg_number
    label_key = _args.label
    batch_key = _args.batch
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('is_hvg:{}'.format(is_hvg))
    pbmc = sc.read(path_data)
    pbmc = preprocessing_rna(pbmc, n_top_features=n_top_features,is_hvg=is_hvg, batch_key =batch_key)
    pbmc_ref = pbmc[pbmc.obs[batch_key] != 'indrop']
    pbmc_query = pbmc[pbmc.obs[batch_key] == 'indrop']
    print('finish preprocessing')
    celltype_list = pbmc_ref.obs[label_key].tolist()
    cc = dict(Counter(celltype_list))
    cc = list(cc.keys())
    save_dict = {'cell type': cc}
    np.savez_compressed('label_dict_ref.npz', **save_dict)
    tech_list = pbmc_ref.obs[batch_key].unique().tolist()
    cc_1 = dict(Counter(tech_list))
    cc_1 = list(cc_1.keys())
    save_dict = {'batch': cc_1}
    np.savez_compressed('batch_dict_ref.npz', **save_dict)
    train_data_file = pbmc_ref.copy()
    obs_list = train_data_file.obs_names.tolist()
    np.random.shuffle(obs_list)
    train_data_file = train_data_file[obs_list]
    save_path_ref = os.path.join(save_path,'ref')
    if not os.path.exists(save_path_ref):
        os.makedirs(save_path_ref)
    tfrecord_oneorder_file_1 = save_path_ref + '/1.tfrecord'
    create_tfrecord(train_data_file, cc, cc_1, tfrecord_oneorder_file_1, zero_filter=False, norm=True,label_key = label_key,batch_key =batch_key)

    celltype_list = pbmc_query.obs[label_key].tolist()
    cc = dict(Counter(celltype_list))
    cc = list(cc.keys())
    save_dict = {'cell type': cc}
    np.savez_compressed('label_dict_query.npz', **save_dict)
    tech_list = pbmc_query.obs[batch_key].unique().tolist()
    cc_1 = dict(Counter(tech_list))
    cc_1 = list(cc_1.keys())
    save_dict = {'batch': cc_1}
    np.savez_compressed('batch_dict_query.npz', **save_dict)
    train_data_file = pbmc_query.copy()
    obs_list = train_data_file.obs_names.tolist()
    np.random.shuffle(obs_list)
    train_data_file = train_data_file[obs_list]
    save_path_query = os.path.join(save_path,'query')
    if not os.path.exists(save_path_query):
        os.makedirs(save_path_query)
    tfrecord_oneorder_file_1 = save_path_query + '/1.tfrecord'
    create_tfrecord(train_data_file, cc, cc_1, tfrecord_oneorder_file_1, zero_filter=False, norm=True,label_key = label_key,batch_key =batch_key)



