import tensorflow as tf
import numpy as np
import os
import scanpy as sc
import argparse
import pandas as pd
import copy
from collections import Counter
from scipy.sparse import issparse
import scipy
import sys
#sys.path.append("./Concerto-main/")
from bgi.utils.data_utils import *
from bgi.models.DeepSingleCell import *
from bgi.metrics.clustering_metrics import *
from bgi.losses.contrastive_loss import simclr_loss
import re
import h5py
import time

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

def set_seeds(seed=10):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

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

def serialize_example_batch(x_feature, x_weight, y_batch,x_id):

    feature = {
        'feature': _int64_feature(x_feature),
        'value': _float_feature(x_weight),
        'batch': _int64_feature(y_batch),
        'id': _bytes_feature(x_id)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def create_tfrecord(source_file,  batch_dict, tfrecord_file, zero_filter=False, norm=False, batch_key = 'batch'):
    if type(source_file.X) != np.ndarray:
        x_data = source_file.X.toarray()
    else:
        x_data = source_file.X
    batch_data = source_file.obs[batch_key].tolist()
    obs_name_list = source_file.obs_names.tolist()
    batch_number = []
    for j in range(len(batch_data)):
        batch = batch_data[j]
        place = batch_dict.index(batch)
        batch_number.append(place)

    counter = 0
    batch_examples = {}
    for x, batch,k in zip(x_data, batch_number,obs_name_list):
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

        if batch not in batch_examples:
            batch_examples[batch] = []

        example = serialize_example_batch(features, values, np.array([int(batch)]),k)
        batch_examples[batch].append(example)

        counter += 1
        if counter % 1000 == 0:
            print('counter: {} shape: {}, batch: {}'.format(counter, features.shape, batch))

            #print(x)
            #print(values)
            #print("batchs: ", batch_dict)

    for item in batch_examples.items():
        batch = item[0]
        examples = item[1]
        if zero_filter is False:
            file = tfrecord_file.replace('.tfrecord', '_{}.tfrecord'.format(batch))
        else:
            if norm is False:
                file = tfrecord_file.replace('.tfrecord', '_{}_no_zero_no_norm.tfrecord'.format(batch))
            else:
                file = tfrecord_file.replace('.tfrecord', '_{}_no_zero.tfrecord'.format(batch))
        with tf.io.TFRecordWriter(file) as writer:
            for example in examples:
                writer.write(example)
    save_dict = {'vocab size': len(features)}
    file = tfrecord_file.replace('tf.tfrecord','vocab_size.npz')
    np.savez_compressed(file, **save_dict)
#     np.savez_compressed('vocab_size.npz', **save_dict)

#  输入模块
def concerto_input(input_file_list):
    query_adata = sc.read(input_file_list)
    return query_adata

# 预处理模块
def concerto_preprocess(query_adata):
    sc.pp.normalize_total(query_adata, target_sum=10000)
    sc.pp.log1p(query_adata)
    return query_adata

# 交集基因
def concerto_intersect_gene(ref_adata, query_adata, parameters=None):
    ref_var_list = ref_adata.var_names.tolist()
    query_var_list = query_adata.var_names.tolist()
    intersect_gene_list = list(set(ref_var_list).intersection(set(query_var_list)))
    intersect_stats_A_B = len(list(set(ref_var_list).difference(set(query_var_list))))# ref中有query中无的个数
    intersect_stats_B_A = len(list(set(query_var_list).difference(set(ref_var_list))))  # ref中有query中无的个数
    intersect_stats = [intersect_stats_A_B,len(intersect_gene_list),intersect_stats_B_A]
    return intersect_gene_list, intersect_stats # list, [int, int, int]([A-B, A交B, B-A])

# HVG
def concerto_HVG(ref_adata,query_adata,n_top_genes=None, min_disp=0.5, min_mean=0.0125, max_mean=3,
	parameters=None):

    sc.pp.highly_variable_genes(query_adata, n_top_genes=n_top_genes, min_disp=0.5,min_mean=0.0125, max_mean=3)
    sc.pp.highly_variable_genes(ref_adata, n_top_genes=n_top_genes, min_disp=0.5, min_mean=0.0125, max_mean=3)
    ref_adata = ref_adata[:,ref_adata.var.highly_variable]
    query_adata = query_adata[:,query_adata.var.highly_variable]
    HVG_list = list(set(ref_adata.var_names.tolist()).intersection(set(query_adata.var_names.tolist())))
    processed_query_adata = query_adata[:,HVG_list]
    processed_ref_adata = ref_adata[:, HVG_list]
    return processed_ref_adata, processed_query_adata, HVG_list

# 如果不训新模型，补全到Ref的基因个数
def concerto_padding(ref_gene_list_path:str, ref_weight_path:str, query_adata):
    # 检验 ref gene list和 weight size 一致
    f = h5py.File(ref_weight_path, 'r')  # 打开h5文件
    if 'RNA-Embedding/embeddings:0' in f['RNA-Embedding']:
        weight_size = f['RNA-Embedding']['RNA-Embedding/embeddings:0'].value.shape[0]
        print('unsup model')
    else:
        weight_size = f['RNA-Embedding']['RNA-Embedding_1/embeddings:0'].value.shape[0]
        print('sup model')
    gene_names = list(pd.read_csv(ref_gene_list_path)['0'].values)
    if weight_size == len(gene_names):
        query_gene_list = query_adata.var_names.tolist()
        gene_inter_list = list(set(gene_names).intersection(set(query_gene_list)))
        empty_matrix = np.zeros([len(query_adata.obs_names),len(gene_names)])
        inter_index = []
        inter_index_query = []
        for i in gene_inter_list:
            inter_index.append(gene_names.index(i))
            inter_index_query.append(query_gene_list.index(i))
        query_X = query_adata.X.toarray()
        query_X_inter = query_X[:, inter_index_query]
        for j in range(query_X_inter.shape[1]):
            empty_matrix[:, inter_index[j]] = query_X_inter[:, j]
        q = sc.AnnData(empty_matrix)
        q.obs = query_adata.obs.copy()
        q.var_names = gene_names
        return q
    else:
        return print('weight size is different from ref gene list')
def concerto_padding2(ref_gene_list_path:str, ref_weight_path:str, query_adata):
    gene_names = list(pd.read_csv(ref_gene_list_path)['0'].values)
    query_gene_list = query_adata.var_names.tolist()
    gene_inter_list = list(set(gene_names).intersection(set(query_gene_list)))
    empty_matrix = np.zeros([len(query_adata.obs_names),len(gene_names)])
    inter_index = []
    inter_index_query = []
    for i in gene_inter_list:
        inter_index.append(gene_names.index(i))
        inter_index_query.append(query_gene_list.index(i))
    query_X = query_adata.X.toarray()
    query_X_inter = query_X[:, inter_index_query]
    for j in range(query_X_inter.shape[1]):
        empty_matrix[:, inter_index[j]] = query_X_inter[:, j]
    q = sc.AnnData(empty_matrix)
    q.obs = query_adata.obs.copy()
    q.var_names = gene_names
    return q


# 造tfrecords
def concerto_make_tfrecord(processed_ref_adata, tf_path, batch_col_name=None):
    # 有输入batch_col_name的时候，用这列作为batchid， 若无假设所有是一个batch
	# 不做乱序,
    if batch_col_name is None:
        batch_col_name = 'batch_'
        sample_num = len(processed_ref_adata.obs_names.tolist())
        processed_ref_adata.obs[batch_col_name]  = ['0']*sample_num
    print(processed_ref_adata)
    batch_list = processed_ref_adata.obs[batch_col_name].unique().tolist()
    cc = dict(Counter(batch_list))
    cc = list(cc.keys())
    tfrecord_file = tf_path + '/tf.tfrecord'
    if not os.path.exists(tf_path):
        os.makedirs(tf_path)
    create_tfrecord(processed_ref_adata, cc, tfrecord_file, zero_filter=False, norm=True, batch_key =batch_col_name)

    return tf_path


# ---------- make supervised tfr --------------
def serialize_example_batch_supervised(x_feature, x_weight, y_label,y_batch,x_id):

    feature = {
        'feature': _int64_feature(x_feature),
        'value': _float_feature(x_weight),
        'label': _int64_feature(y_label),
        'batch': _int64_feature(y_batch),
        'id': _bytes_feature(x_id)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def create_tfrecord_supervised(source_file,  batch_dict,label_dict, tfrecord_file, zero_filter=False, norm=False, batch_key = 'batch',label_key = 'label'):
    if type(source_file.X) != np.ndarray:
        x_data = source_file.X.toarray()
    else:
        x_data = source_file.X
    batch_data = source_file.obs[batch_key].tolist()
    label_data = source_file.obs[label_key].tolist()
    obs_name_list = source_file.obs_names.tolist()
    batch_number = []
    label_number = []
    for j in range(len(batch_data)):
        batch = batch_data[j]
        place = batch_dict.index(batch)
        batch_number.append(place)

    for j in range(len(label_data)):
        cell_type = label_data[j]
        place = label_dict.index(cell_type)
        label_number.append(place)

    counter = 0
    batch_examples = {}
    for x, y,batch,k in zip(x_data, label_number,batch_number,obs_name_list):
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

        if batch not in batch_examples:
            batch_examples[batch] = []
        y = np.array([int(y)])
        example = serialize_example_batch_supervised(features, values, y, np.array([int(batch)]),k)
        batch_examples[batch].append(example)

        counter += 1
        if counter % 100 == 0:
            print('counter: {} shape: {}, batch: {}'.format(counter, features.shape, batch))

            print(x)
            print(values)
            print("batchs: ", batch_dict)

    for item in batch_examples.items():
        batch = item[0]
        examples = item[1]
        if zero_filter is False:
            file = tfrecord_file.replace('.tfrecord', '_{}.tfrecord'.format(batch))
        else:
            if norm is False:
                file = tfrecord_file.replace('.tfrecord', '_{}_no_zero_no_norm.tfrecord'.format(batch))
            else:
                file = tfrecord_file.replace('.tfrecord', '_{}_no_zero.tfrecord'.format(batch))
        with tf.io.TFRecordWriter(file) as writer:
            for example in examples:
                writer.write(example)
    #save_dict = {'vocab size': len(features)}
    save_dict = {'vocab size': len(features),'classes number':len(label_dict),'label_dict':label_dict,'batch_dict':batch_dict}
    file = tfrecord_file.replace('tf.tfrecord','vocab_size.npz')
    np.savez_compressed(file, **save_dict)

# 造tfrecords
def concerto_make_tfrecord_supervised(processed_ref_adata, tf_path,save_dict = None, batch_col_name=None,label_col_name=None):
    # 有输入batch_col_name的时候，用这列作为batchid， 若无假设所有是一个batch
	# 不做乱序,
    tfrecord_file = os.path.join(tf_path, 'tf.tfrecord')
    if not os.path.exists(tf_path):
        os.makedirs(tf_path)
    if batch_col_name is None:
        batch_col_name = 'batch'
        sample_num = len(processed_ref_adata.obs_names.tolist())
        processed_ref_adata.obs[batch_col_name] = ['0'] * sample_num
    if label_col_name is None:
        label_col_name = 'label'
    if save_dict is not None:
        f = np.load(os.path.join(save_dict,'vocab_size.npz')) # load saved dict path
        cc_ = list(f['label_dict'])
        cc = list(f['batch_dict'])
    else:
        batch_list = processed_ref_adata.obs[batch_col_name].unique().tolist()
        cc = dict(Counter(batch_list))
        cc = list(cc.keys())
        label_list = processed_ref_adata.obs[label_col_name].unique().tolist()
        cc_ = dict(Counter(label_list))
        cc_ = list(cc_.keys())
    create_tfrecord_supervised(processed_ref_adata, cc,cc_, tfrecord_file, zero_filter=False, norm=True, batch_key =batch_col_name,label_key=label_col_name)

    return tf_path



def create_tfrecord_supervised_1batch(source_file, batch_dict,label_dict, tfrecord_file, zero_filter=False, norm=False, batch_key = 'batch',label_key = 'label'):
    if type(source_file.X) != np.ndarray:
        x_data = source_file.X.toarray()
    else:
        x_data = source_file.X
    batch_data = source_file.obs[batch_key].tolist()
    label_data = source_file.obs[label_key].tolist()
    obs_name_list = source_file.obs_names.tolist()
    batch_name = batch_dict[0]
    batch_number = []
    label_number = []
    for j in range(len(batch_data)):
        batch = batch_data[j]
        place = batch_dict.index(batch)
        batch_number.append(place)

    for j in range(len(label_data)):
        cell_type = label_data[j]
        place = label_dict.index(cell_type)
        label_number.append(place)

    counter = 0
    batch_examples = {}
    for x, y,batch,k in zip(x_data, label_number,batch_number,obs_name_list):
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

        if batch not in batch_examples:
            batch_examples[batch] = []
        y = np.array([int(y)])
        example = serialize_example_batch_supervised(features, values, y, np.array([int(batch)]),k)
        batch_examples[batch].append(example)

        counter += 1
        if counter % 100 == 0:
            print('counter: {} shape: {}, batch: {}'.format(counter, features.shape, batch))

            print(x)
            print(values)
            print("batchs: ", batch_dict)

    for item in batch_examples.items():
        batch = item[0]
        examples = item[1]
        if zero_filter is False:
            file = tfrecord_file.replace('.tfrecord', '_{}.tfrecord'.format(batch_name))
        else:
            if norm is False:
                file = tfrecord_file.replace('.tfrecord', '_{}_no_zero_no_norm.tfrecord'.format(batch_name))
            else:
                file = tfrecord_file.replace('.tfrecord', '_{}_no_zero.tfrecord'.format(batch_name))
        with tf.io.TFRecordWriter(file) as writer:
            for example in examples:
                writer.write(example)
    #save_dict = {'vocab size': len(features)}
    save_dict = {'vocab size': len(features),'classes number':len(label_dict),'label_dict':label_dict,'batch_dict':batch_dict}
    file = tfrecord_file.replace('tf.tfrecord','vocab_size.npz')
    np.savez_compressed(file, **save_dict)



# 造tfrecords
def concerto_make_tfrecord_supervised_1batch(processed_ref_adata, tf_path, save_dict = None, batch_col_name=None,label_col_name=None):
    # 有输入batch_col_name的时候，用这列作为batchid， 若无假设所有是一个batch
	# 不做乱序,
    tfrecord_file = os.path.join(tf_path, 'tf.tfrecord')
    if not os.path.exists(tf_path):
        os.makedirs(tf_path)
    if batch_col_name is None:
        batch_col_name = 'batch'
        sample_num = len(processed_ref_adata.obs_names.tolist())
        processed_ref_adata.obs[batch_col_name] = ['0'] * sample_num
    if label_col_name is None:
        label_col_name = 'label'
    if save_dict is not None:
        f = np.load(os.path.join(save_dict,'vocab_size.npz')) # load saved dict path
        cc_ = list(f['label_dict'])
        cc = list(f['batch_dict'])

    else:
        batch_list = processed_ref_adata.obs[batch_col_name].unique().tolist()
        cc = dict(Counter(batch_list))
        cc = list(cc.keys())
        label_list = processed_ref_adata.obs[label_col_name].unique().tolist()
        cc_ = dict(Counter(label_list))
        cc_ = list(cc_.keys())

    create_tfrecord_supervised_1batch(processed_ref_adata, cc,cc_, tfrecord_file, zero_filter=False, norm=True, batch_key =batch_col_name,label_key=label_col_name)

    return tf_path

# -----------make supervised tfr end-----------

# -------- multi Threding make tf ---------------
def write_tf_multiThre(batch_i,dir_='./tf_files/',batch_col_name='batch',pp_flag=True,label_col_name=None):
    '''
        adata_i should be announce by global
        use case:
            from multiprocessing import Pool

            p = Pool(10)
            res_l = []
            global adata_ 
            adata_ = ref_p
            tf_save_path = './zzm_tf_pbmc100k_hvg5000/'

            for batch_i in adata_.obs['batch'].unique():
                print('batch=',batch_i)
                res = p.apply_async(write_tf_multiThre, args=(batch_i,))
                res_l.append(res)

            p.close()
            p.join()

            for batch_i in adata_.obs['batch'].unique():
                tf_i = f'{tf_save_path}{batch_i}/tf_{batch_i}.tfrecord'
                shutil.copy2(tf_i, tf_save_path)
            shutil.copy2(f'{tf_save_path}{batch_i}/vocab_size.npz', tf_save_path)
    '''
    if label_col_name is None:
        adata_.obs['fake_label'] = 'fla1'
        label_col_name = 'fake_label'
    
    vs_ = {
    'vocab size': len(adata_.var_names),
    'classes number': adata_.obs[label_col_name].nunique(),
    'label_dict': adata_.obs[label_col_name].unique(),
    'batch_dict': batch_i,
    }
    tf_path = f'{dir_}{batch_i}/'
    if not os.path.exists(tf_path):
        os.makedirs(tf_path)
    np.savez_compressed(f'{dir_}{batch_i}/vocab_size.npz', **vs_)
    ma_i = adata_[adata_.obs[batch_col_name]==batch_i].copy()
    if pp_flag:
        ma_i.obsm['raw'] = ma_i.X
        sc.pp.normalize_total(ma_i, target_sum=10000)
        sc.pp.log1p(ma_i)

    ref_tf_path = conc.concerto_make_tfrecord_supervised_1batch(
        ma_i,
       tf_path=f'{dir_}{batch_i}/',
        batch_col_name=batch_col_name,
        label_col_name=label_col_name)
# -------- multi Threding make tf end ---------------

# train unsupervised
def concerto_train_ref(ref_tf_path:str, weight_path:str, super_parameters=None):
    set_seeds(0)
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size':32,'epoch':3,'lr':1e-5}
#     dirname = os.getcwd()
#     f = np.load(ref_tf_path + './vocab_size.npz')
    f = np.load(os.path.join(ref_tf_path,'vocab_size.npz'))
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
#     tf_list_1 = os.listdir(os.path.join(ref_tf_path))
    tf_list_1 = [f for f in os.listdir(os.path.join(ref_tf_path)) if 'tfrecord' in f]
    train_source_list = []
    for i in tf_list_1:
        train_source_list.append(os.path.join(ref_tf_path, i))

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_cls_accuracy')
    test_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_cls_accuracy')
    total_update_steps = 300 * super_parameters['epoch']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps, super_parameters['lr']*1e-2, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    for epoch in range(super_parameters['epoch']):
        np.random.shuffle(train_source_list)
        for file in train_source_list:
            print(file)
            train_db = create_classifier_dataset_multi([file],
                                                       batch_size=super_parameters['batch_size'],
                                                       is_training=True,
                                                       data_augment=False,
                                                       shuffle_size=10000)

            train_loss.reset_states()
            train_cls_accuracy.reset_states()
            test_cls_accuracy.reset_states()
            for step, (source_features, source_values, source_batch, source_id) in enumerate(train_db):
                # enumerate
                with tf.GradientTape() as tape:
                    z1 = encode_network([source_features, source_values], training=True)
                    z2 = decode_network([source_values], training=True)
                    mu_1 = mu_enc(z1)
                    var_1 = tf.exp(var_enc(z1))
                    ssl_loss = simclr_loss(z1, z2,temperature = 0.1)
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
        encode_network.save_weights(
            weight_path + 'weight_encoder_epoch{}.h5'.format(str(epoch+1)))
        decode_network.save_weights(
            weight_path + 'weight_decoder_epoch{}.h5'.format(str(epoch+1)))

    return weight_path

# train supervised
# train
def concerto_train_ref_supervised(ref_tf_path:str, weight_path:str, super_parameters=None):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size':32,'epoch_pretrain':1,'epoch_classifier':5,'lr':1e-5,}
    # dirname = os.getcwd()
    f = np.load(os.path.join(ref_tf_path, 'vocab_size.npz'))
    
    vocab_size = int(f['vocab size'])
    num_classes = int(f['classes number'])
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
    # tf_list_1 = os.listdir(os.path.join(ref_tf_path))
    tf_list_1 = [f for f in os.listdir(os.path.join(ref_tf_path)) if 'tfrecord' in f]
    train_source_list = []
    for i in tf_list_1:
        train_source_list.append(os.path.join(ref_tf_path, i))

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    cls_loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_cls_accuracy')
    test_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_cls_accuracy')
    total_update_steps = 300 * super_parameters['epoch_pretrain']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps, super_parameters['lr']*1e-2, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    for epoch in range(super_parameters['epoch_pretrain']):
        np.random.shuffle(train_source_list)
        for file in train_source_list:
            print(file)
            train_db = create_classifier_dataset_multi_supervised([file],
                                                       batch_size=super_parameters['batch_size'],
                                                       is_training=True,
                                                       data_augment=False,
                                                       shuffle_size=10000)

            train_loss.reset_states()
            train_cls_accuracy.reset_states()
            test_cls_accuracy.reset_states()
            for step, (source_features, source_values, source_label, source_batch, source_id) in enumerate(train_db):
                # enumerate
                with tf.GradientTape() as tape:
                    z1 = encode_network([source_features, source_values], training=True)
                    z2 = decode_network([source_values], training=True)
                    mu_1 = mu_enc(z1)
                    var_1 = tf.exp(var_enc(z1))
                    ssl_loss = simclr_loss(z1, z2,temperature = 0.1)
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

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    output = encode_network.layers[-1].output
    output = tf.keras.layers.Dense(num_classes, activation='softmax', name='CLS')(output)
    cls_network = tf.keras.Model(encode_network.input, outputs=output)
    for epoch in range(super_parameters['epoch_classifier']):
        np.random.shuffle(train_source_list)
        for file in train_source_list:
            print(file)
            train_db = create_classifier_dataset_multi_supervised([file],
                                                       batch_size=super_parameters['batch_size'],
                                                       is_training=True,
                                                       data_augment=False,
                                                       shuffle_size=10000)

            train_loss.reset_states()
            train_cls_accuracy.reset_states()
            test_cls_accuracy.reset_states()
            for step, (source_features, source_values, source_label, source_batch, source_id) in enumerate(train_db):
                # enumerate
                with tf.GradientTape() as tape:
                    outputs = cls_network([source_features, source_values], training=True)
                    classifer_loss = cls_loss_object(source_label, outputs)
                    source_pred = outputs
                    train_cls_accuracy(source_label, source_pred)
                    train_loss(classifer_loss)

                variables = [cls_network.trainable_variables]
                grads = tape.gradient(classifer_loss, variables)
                for grad, var in zip(grads, variables):
                    opt.apply_gradients(zip(grad, var))

                if step > 0 and step % 5 == 0:
                    template = 'Epoch {}, step {}, train cls loss: {:0.4f}, train acc: {:0.4f}'
                    print(template.format(epoch,
                                          str(step),
                                          train_loss.result(),
                                          train_cls_accuracy.result(),
                                          ))
        encode_network.save_weights(
            os.path.join(weight_path, 'weight_encoder_epoch{}.h5'.format(str(epoch+1))))
        decode_network.save_weights(
            os.path.join(weight_path, 'weight_decoder_epoch{}.h5'.format(str(epoch+1))))

    return weight_path

# train sup 0112
def concerto_train_ref_supervised_yzs(ref_tf_path:str, weight_path:str, super_parameters=None):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size':32,'epoch_pretrain':1,'epoch_classifier':5,'lr':1e-5, 'drop_rate': 0.1}
#     dirname = os.getcwd()
    f = np.load(ref_tf_path + '/vocab_size.npz')
    
    vocab_size = int(f['vocab size'])
    num_classes = int(f['classes number'])
    encode_network = multi_embedding_attention_transfer(multi_max_features=[vocab_size],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    decode_network = multi_embedding_attention_transfer(multi_max_features=[vocab_size],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    mu_enc = EncoderHead()
    var_enc = EncoderHead()
#     tf_list_1 = os.listdir(os.path.join(ref_tf_path))
    tf_list_1 = [f for f in os.listdir(os.path.join(ref_tf_path)) if 'tfrecord' in f]
    train_source_list = []
    for i in tf_list_1:
        train_source_list.append(os.path.join(ref_tf_path, i))

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    cls_loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_cls_accuracy')
    test_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_cls_accuracy')
    total_update_steps = 300 * super_parameters['epoch_pretrain']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps, super_parameters['lr']*1e-2, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    for epoch in range(super_parameters['epoch_pretrain']):
        np.random.shuffle(train_source_list)
        for file in train_source_list:
            print(file)
            train_db = create_classifier_dataset_multi_supervised([file],
                                                       batch_size=super_parameters['batch_size'],
                                                       is_training=True,
                                                       data_augment=False,
                                                       shuffle_size=10000)

            train_loss.reset_states()
            train_cls_accuracy.reset_states()
            test_cls_accuracy.reset_states()
            for step, (source_features, source_values, source_label, source_batch, source_id) in enumerate(train_db):
                # enumerate
                with tf.GradientTape() as tape:
                    z1 = encode_network([source_features, source_values], training=True)
                    z2 = decode_network([source_values], training=True)
                    mu_1 = mu_enc(z1)
                    var_1 = tf.exp(var_enc(z1))
                    ssl_loss = simclr_loss(z1, z2,temperature = 0.1)
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

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    output = encode_network.layers[-1].output
    output = tf.keras.layers.Dense(num_classes, activation='softmax', name='CLS')(output)
    cls_network = tf.keras.Model(encode_network.input, outputs=output)
    for epoch in range(super_parameters['epoch_classifier']):
        np.random.shuffle(train_source_list)
        for file in train_source_list:
            print(file)
            train_db = create_classifier_dataset_multi_supervised([file],
                                                       batch_size=super_parameters['batch_size'],
                                                       is_training=True,
                                                       data_augment=False,
                                                       shuffle_size=10000)

            train_loss.reset_states()
            train_cls_accuracy.reset_states()
            test_cls_accuracy.reset_states()
            for step, (source_features, source_values, source_label, source_batch, source_id) in enumerate(train_db):
                # enumerate
                with tf.GradientTape() as tape:
                    outputs = cls_network([source_features, source_values], training=True)
                    classifer_loss = cls_loss_object(source_label, outputs)
                    source_pred = outputs
                    train_cls_accuracy(source_label, source_pred)
                    train_loss(classifer_loss)

                variables = [cls_network.trainable_variables]
                grads = tape.gradient(classifer_loss, variables)
                for grad, var in zip(grads, variables):
                    opt.apply_gradients(zip(grad, var))

                if step > 0 and step % 5 == 0:
                    template = 'Epoch {}, step {}, train cls loss: {:0.4f}, train acc: {:0.4f}'
                    print(template.format(epoch,
                                          str(step),
                                          train_loss.result(),
                                          train_cls_accuracy.result(),
                                          ))
        encode_network.save_weights(
            weight_path + 'weight_encoder_epoch{}.h5'.format(str(epoch+1)))
        decode_network.save_weights(
            weight_path + 'weight_decoder_epoch{}.h5'.format(str(epoch+1)))
        cls_network.save_weights(os.path.join(weight_path, 'weight_cls_epoch{}.h5'.format(str(epoch+1))))
        

    return weight_path

# test
def concerto_test_1set_attention_supervised(model_path: str, ref_tf_path: str, super_parameters=None, n_cells_for_ref=5000):
    if super_parameters is None:
        super_parameters = {'batch_size': 128, 'epoch': 1, 'lr': 1e-5,'drop_rate': 0.1}

    f = np.load(os.path.join(ref_tf_path, 'vocab_size.npz'))
    vocab_size = int(f['vocab size'])
    num_classes = int(f['classes number'])
    label_dict = f['label_dict']
    batch_dict = f['batch_dict']
    batch_size = super_parameters['batch_size']
    encode_network = multi_embedding_attention_transfer(
        multi_max_features=[vocab_size],
        mult_feature_names=['RNA'],
        embedding_dims=128,
        include_attention=True,
        drop_rate=super_parameters['drop_rate'],
        head_1=128,
        head_2=128,
        head_3=128)
    tf_list_1 = [f for f in os.listdir(os.path.join(ref_tf_path)) if 'tfrecord' in f]
    train_source_list = [os.path.join(ref_tf_path, i) for i in tf_list_1]
    # choose last epoch as test model
    weight_id_list = []
    # weight_list = [f for f in os.listdir(model_path) if (f.endswith('h5') and f.startswith('weight') )]
    weight_list = [f for f in os.listdir(model_path) if (f.endswith('h5') and ('cls' in f))]  # yyyx 1214
    for id in weight_list:
        id_1 = re.findall('.*epoch(.*).h5', id)  # f1
        weight_id_list.append(int(id_1[0]))
    weight_name_ = sorted(list(zip(weight_id_list, weight_list)), key=lambda x: x[0])[-1][1]
    output = encode_network.layers[-1].output
    output = tf.keras.layers.Dense(num_classes, activation='softmax', name='CLS')(output)
    cls_network = tf.keras.Model(encode_network.input, outputs=output)
    cls_network.load_weights(os.path.join(model_path, weight_name_))

    t1 = time.time()
    ref_db = create_classifier_dataset_multi_supervised(
        train_source_list,
        batch_size=batch_size,  # maybe slow
        is_training=False,
        data_augment=False,
        shuffle_size=10000)

    t2 = time.time()
    print('load all tf in memory time(s)', t2 - t1)  # time consumption is huge this step!!!!

    feature_len = n_cells_for_ref // batch_size * batch_size
    print(feature_len, batch_size)
    t2 = time.time()
    source_data_batch_1 = np.zeros((feature_len))
    source_data_label_1 = np.zeros((feature_len))
    source_data_pred_1 = np.zeros((feature_len))
    source_id_1 = []
    source_id_label_1 = []
    source_id_batch_1 = []
    source_id_pred_1 = []
    for step, (target_features, target_values,target_label, target_batch, target_id) in enumerate(ref_db):
        if step * batch_size >= feature_len:
            break
        preds = cls_network([target_features, target_values], training=False)
        preds_1 = np.argmax(preds, axis=1)
        source_data_pred_1[step * batch_size:(step + 1) * batch_size] = preds_1
        source_data_batch_1[step * batch_size:(step + 1) * batch_size] = target_batch
        source_data_label_1[step * batch_size:(step + 1) * batch_size] = target_label
        source_id_1.extend(list(target_id.numpy().astype('U')))

    t3 = time.time()
    print('test time', t3 - t2)
    print('source_id len', len(source_id_1))
    for j in source_data_label_1:
        source_id_label_1.append(label_dict[int(j)])
    for j in source_data_pred_1:
        source_id_pred_1.append(label_dict[int(j)])
    for j in source_data_batch_1:
        source_id_batch_1.append(batch_dict[int(j)])

    acc = accuracy_score(source_data_label_1, source_data_pred_1)
    f1_scores_median = f1_score(source_data_label_1, source_data_pred_1, average=None)
    f1_scores_median = np.median(f1_scores_median)
    f1_scores_macro = f1_score(source_data_label_1, source_data_pred_1, average='macro')
    f1_scores_micro = f1_score(source_data_label_1, source_data_pred_1, average='micro')
    print('acc:', acc, 'f1_scores_median:', f1_scores_median, 'f1_scores_macro:',
          f1_scores_macro, 'f1_scores_micro:', f1_scores_micro)


    return acc,f1_scores_median



# query
def concerto_train_query(ref_model_path:str,ref_tf_path:str,query_tf_path:str, weight_path:str, super_parameters=None):
    set_seeds(0)
    if super_parameters is None:
        super_parameters = {'batch_size': 32, 'epoch': 1, 'lr': 1e-5}
    #dirname = os.getcwd()
#     f = np.load(ref_tf_path + '/vocab_size.npz')
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    f = np.load(os.path.join(ref_tf_path,'vocab_size.npz'))
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
#     tf_list_2 = os.listdir(os.path.join(query_tf_path))
    tf_list_2 = [f for f in os.listdir(os.path.join(query_tf_path)) if 'tfrecord' in f]

    train_target_list = []
    for i in tf_list_2:
        train_target_list.append(os.path.join(query_tf_path, i))

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_cls_accuracy')
    test_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_cls_accuracy')
    total_update_steps = 300 * super_parameters['epoch']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps, super_parameters['lr']*1e-2, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    weight_id_list = []
    weight_list = [f for f in os.listdir(ref_model_path) if f.endswith('h5')]
    for id in weight_list:
        id_1 = re.findall('.*epoch(.*).h.*', id)  # f1
        weight_id_list.append(int(id_1[0]))
    # encode_network.load_weights(ref_model_path + '/weight_encoder_epoch{}.h5'.format(max(weight_id_list))) 
    encode_network.load_weights(ref_model_path + '/weight_encoder_epoch{}.h5'.format(max(weight_id_list)), by_name=True) # yyyx 0126, 支持多模态模型fine tune
    decode_network.load_weights(ref_model_path + '/weight_decoder_epoch{}.h5'.format(max(weight_id_list)))
    for epoch in range(super_parameters['epoch']):
        for file in train_target_list:
            print(file)
            train_db = create_classifier_dataset_multi([file],
                                                       batch_size=super_parameters['batch_size'],
                                                       is_training=True,
                                                       data_augment=False,
                                                       shuffle_size=10000)

            train_loss.reset_states()
            train_cls_accuracy.reset_states()
            test_cls_accuracy.reset_states()
            for step, (source_features, source_values, source_batch, source_id) in enumerate(
                    train_db):
                # enumerate
                with tf.GradientTape() as tape:
                    z1 = encode_network([source_features, source_values], training=True)
                    z2 = decode_network([source_values], training=True)
                    mu_1 = mu_enc(z1)
                    var_1 = tf.exp(var_enc(z1))
                    ssl_loss = simclr_loss(z1, z2, temperature=0.1)
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
        encode_network.save_weights(
            weight_path + '/weight_encoder_epoch{}.h5'.format(str(epoch + 1)))

    return weight_path

# 无监督一起训REF和query，解决读入模型不一致的问题
def concerto_train_ref_query(ref_tf_path: str, query_tf_path: str, weight_path: str, super_parameters=None):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 3, 'epoch_fineturn': 1, 'lr': 1e-4,'drop_rate': 0.1}
    # dirname = os.getcwd()
    # f = np.load(ref_tf_path + './vocab_size.npz')
    f = np.load(os.path.join(ref_tf_path, 'vocab_size.npz'))
    vocab_size = int(f['vocab size'])
    encode_network = multi_embedding_attention_transfer(multi_max_features=[vocab_size],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    decode_network = multi_embedding_attention_transfer(multi_max_features=[vocab_size],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    mu_enc = EncoderHead()
    var_enc = EncoderHead()
    # tf_list_1 = os.listdir(os.path.join(ref_tf_path))
    tf_list_1 = [f for f in os.listdir(os.path.join(ref_tf_path)) if 'tfrecord' in f]
    train_source_list = []
    for i in tf_list_1:
        train_source_list.append(os.path.join(ref_tf_path, i))

    tf_list_2 = [f for f in os.listdir(os.path.join(query_tf_path)) if 'tfrecord' in f]
    train_target_list = []
    for i in tf_list_2:
        train_target_list.append(os.path.join(query_tf_path, i))

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_cls_accuracy')
    test_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_cls_accuracy')
    total_update_steps = 300 * super_parameters['epoch_pretrain']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps,
                                                                super_parameters['lr'] * 1e-2, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    for epoch in range(super_parameters['epoch_pretrain']):
        np.random.shuffle(train_source_list)
        for file in train_source_list:
            print(file)
            train_db = create_classifier_dataset_multi([file],
                                                       batch_size=super_parameters['batch_size'],
                                                       is_training=True,
                                                       data_augment=False,
                                                       shuffle_size=10000)

            train_loss.reset_states()
            train_cls_accuracy.reset_states()
            test_cls_accuracy.reset_states()
            for step, (source_features, source_values, source_batch, source_id) in enumerate(train_db):
                # enumerate
                with tf.GradientTape() as tape:
                    z1 = encode_network([source_features, source_values], training=True)
                    z2 = decode_network([source_values], training=True)
                    mu_1 = mu_enc(z1)
                    var_1 = tf.exp(var_enc(z1))
                    ssl_loss = simclr_loss(z1, z2, temperature=0.1)
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

    for epoch in range(super_parameters['epoch_fineturn']):
        np.random.shuffle(train_target_list)
        for file in train_target_list:
            print(file)
            train_db = create_classifier_dataset_multi([file],
                                                       batch_size=super_parameters['batch_size'],
                                                       is_training=True,
                                                       data_augment=False,
                                                       shuffle_size=10000)

            train_loss.reset_states()
            train_cls_accuracy.reset_states()
            test_cls_accuracy.reset_states()
            for step, (source_features, source_values, source_batch, source_id) in enumerate(train_db):
                # enumerate
                with tf.GradientTape() as tape:
                    z1 = encode_network([source_features, source_values], training=True)
                    z2 = decode_network([source_values], training=True)
                    mu_1 = mu_enc(z1)
                    var_1 = tf.exp(var_enc(z1))
                    ssl_loss = simclr_loss(z1, z2, temperature=0.1)
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
        encode_network.save_weights(
            os.path.join(weight_path, 'weight_encoder_epoch{}.h5'.format(str(epoch + 1))))
        decode_network.save_weights(
            os.path.join(weight_path, 'weight_decoder_epoch{}.h5'.format(str(epoch + 1))))

    return weight_path


def concerto_train_multimodal(RNA_tf_path: str, Protein_tf_path: str, weight_path: str, super_parameters=None):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 3, 'lr': 1e-4,'drop_rate': 0.1}
    # dirname = os.getcwd()
    # f = np.load(ref_tf_path + './vocab_size.npz')
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(Protein_tf_path, 'vocab_size.npz'))
    vocab_size_Protein = int(f['vocab size'])
    encode_network = multi_embedding_attention_transfer(multi_max_features=[vocab_size_RNA,vocab_size_Protein],
                                                        mult_feature_names=['RNA','Protein'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    decode_network = multi_embedding_attention_transfer(multi_max_features=[vocab_size_RNA,vocab_size_Protein],
                                                        mult_feature_names=['RNA','Protein'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    mu_enc = EncoderHead()
    var_enc = EncoderHead()
    # tf_list_1 = os.listdir(os.path.join(ref_tf_path))
    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_Protein = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_Protein.append(os.path.join(Protein_tf_path, i))


    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_cls_accuracy')
    test_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_cls_accuracy')
    total_update_steps = 300 * super_parameters['epoch_pretrain']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps,
                                                                super_parameters['lr'] * 1e-2, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    for epoch in range(super_parameters['epoch_pretrain']):
        for RNA_file, Protein_file in zip(train_source_list_RNA, train_source_list_Protein):
            print(RNA_file)
            print(Protein_file)
            train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                           batch_size=super_parameters['batch_size'],
                                                           is_training=False,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
            train_db_Protein = create_classifier_dataset_multi([Protein_file],
                                                               batch_size=super_parameters['batch_size'],
                                                               is_training=False,
                                                               data_augment=False,
                                                               shuffle_size=10000,
                                                               )
            train_loss.reset_states()
            train_cls_accuracy.reset_states()
            test_cls_accuracy.reset_states()
            step = 0
            for (source_features_RNA, source_values_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_protein, source_values_protein,
                 source_batch_Protein, source_id_Protein) \
                    in (zip(train_db_RNA, train_db_Protein)):
                step += 1

                with tf.GradientTape() as tape:
                    z1 = encode_network([[source_features_RNA, source_features_protein],
                                         [source_values_RNA, source_values_protein]], training=True)
                    z2 = decode_network([source_values_RNA, source_values_protein], training=True)
                    ssl_loss = simclr_loss(z1, z2, temperature=0.1)
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

                if step > 0 and step % 5 == 0:
                    template = 'Epoch {}, step {}, simclr loss: {:0.4f}.'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result()))
        encode_network.save_weights(
            os.path.join(weight_path, 'weight_encoder_epoch{}.h5'.format(str(epoch + 1))))
        decode_network.save_weights(
            os.path.join(weight_path, 'weight_decoder_epoch{}.h5'.format(str(epoch + 1))))

    return print('finished')





def concerto_test_1set_attention(model_path:str, ref_tf_path:str, super_parameters=None, n_cells_for_ref=5000):
    set_seeds(0)
    
    if super_parameters is None:
            super_parameters = {'batch_size': 32, 'epoch': 1, 'lr': 1e-5}
    
    f = np.load(ref_tf_path + 'vocab_size.npz')
    vocab_size = int(f['vocab size'])
    batch_size = super_parameters['batch_size']
    encode_network = multi_embedding_attention_transfer_1(
        multi_max_features=[vocab_size],
        mult_feature_names=['RNA'],
        embedding_dims=128,
        include_attention=True,
        drop_rate=0.1,
        head_1=128,
        head_2=128,
        head_3=128)
    tf_list_1 = [f for f in os.listdir(os.path.join(ref_tf_path)) if 'tfrecord' in f]
    train_source_list = [os.path.join(ref_tf_path, i) for i in tf_list_1]
    # choose last epoch as test model
    weight_id_list = []
#     weight_list = [f for f in os.listdir(model_path) if (f.endswith('h5') and f.startswith('weight') )]
    weight_list = [f for f in os.listdir(model_path) if (f.endswith('h5') and ('encoder' in f) )] # yyyx 1214
    for id in weight_list:
        id_1 = re.findall('.*epoch(.*).h5', id)  # f1
        weight_id_list.append(int(id_1[0]))
    weight_name_ = sorted(list(zip(weight_id_list,weight_list)),key=lambda x:x[0])[-1][1]
    encode_network.load_weights(model_path + weight_name_, by_name=True)
    
    t1 = time.time()
    ref_db = create_classifier_dataset_multi(
        train_source_list,
        batch_size=batch_size, # maybe slow
        is_training=True,
        data_augment=False,
        shuffle_size=10000)
    for step, (target_features, target_values, target_batch, target_id) in enumerate(ref_db):
        output,_ = encode_network([target_features, target_values], training=False)
        break
    t2 = time.time()
    print('load all tf in memory time(s)',t2-t1) # time consumption is huge this step!!!!

    feature_len = n_cells_for_ref//batch_size*batch_size
    print(feature_len, batch_size)
    t2 = time.time()
    dim = output.shape[1]
    source_data_feature_1 = np.zeros((feature_len, dim))
    source_data_batch_1 = np.zeros((feature_len))
    attention_weight = np.zeros((feature_len, vocab_size,1))
    source_id_batch_1 = []
    for step, (target_features, target_values, target_batch, target_id) in enumerate(ref_db):
        if step*batch_size >= feature_len:
            break
        output,attention_output = encode_network([target_features, target_values], training=False)
        output = tf.nn.l2_normalize(output, axis=-1)
        source_data_feature_1[step * batch_size:(step+1) * batch_size, :] = output
        source_data_batch_1[step * batch_size:(step+1) * batch_size] = target_batch
        attention_weight[step * batch_size:(step+1) * batch_size, :,:] = attention_output[-1]
        source_id_batch_1.extend(list(target_id.numpy().astype('U')))

    t3 = time.time()
    print('test time',t3-t2)
    print('source_id_batch_1 len', len(source_id_batch_1))
#     source_id_batch_1 = [i.decode("utf-8") for i in source_id_batch_1]
    return source_data_feature_1, list(source_id_batch_1),attention_weight



def concerto_test_new(model_path:str, ref_tf_path:str, query_tf_path:str, super_parameters=None, n_cells_for_ref=5000):
    if super_parameters is None:
        super_parameters = {'batch_size': 32, 'epoch': 1, 'lr': 1e-5}

    f = np.load(ref_tf_path + 'vocab_size.npz')
    vocab_size = int(f['vocab size'])
    encode_network = multi_embedding_attention_transfer(
        multi_max_features=[vocab_size],
        mult_feature_names=['RNA'],
        embedding_dims=128,
        include_attention=True,
        drop_rate=0.1,
        head_1=128,
        head_2=128,
        head_3=128)

    tf_list_1 = [f for f in os.listdir(os.path.join(ref_tf_path)) if 'tfrecord' in f]
    train_source_list = []
    for i in tf_list_1:
        train_source_list.append(os.path.join(ref_tf_path, i))
    tf_list_2 = [f for f in os.listdir(os.path.join(query_tf_path)) if 'tfrecord' in f]
    train_target_list = []
    for i in tf_list_2:
        train_target_list.append(os.path.join(query_tf_path, i))
    weight_id_list = []
    weight_list = [f for f in os.listdir(model_path) if f.endswith('h5')]

    for id in weight_list:
        id_1 = re.findall('.*epoch(.*).h.*', id)  # f1
        weight_id_list.append(int(id_1[0]))
    encode_network.load_weights(model_path + 'weight_encoder_epoch{}.h5'.format(max(weight_id_list)),by_name=True)
    source_data_batch = []
    source_data_feature = []
    source_data_id = []
    batch_size = super_parameters['batch_size']
    for file in train_source_list:
        print(file)
        feature_len = 0
        ref_db = create_classifier_dataset_multi(
            [file],
            batch_size=1,
            is_training=True,
            data_augment=False,
            shuffle_size=10000)
        for step, (target_features, target_values, target_batch, target_id) in enumerate(ref_db):
            feature_len += 1
            if step == 0:
                output = encode_network([target_features, target_values], training=False)

        dim = output.shape[1]
        source_data_feature_1 = np.zeros((feature_len, dim))
        source_data_batch_1 = np.zeros((feature_len))
        source_id_batch_1 = []
        for step, (target_features, target_values, target_batch, target_id) in enumerate(ref_db):
            output = encode_network([target_features, target_values], training=False)
            output = tf.nn.l2_normalize(output, axis=-1)
            source_data_feature_1[step, :] = output
            source_data_batch_1[step] = target_batch
            source_id_batch_1.append(target_id.numpy()[0])
        source_data_feature.extend(source_data_feature_1)
        source_data_batch.extend(source_data_batch_1)
        source_data_id.extend(source_id_batch_1)

    target_data_batch = []
    target_data_feature = []
    target_data_id = []
    for file in train_target_list:
        print(file)
        feature_len = 0
        query_db = create_classifier_dataset_multi([file],
                                                   batch_size=1,
                                                   is_training=False,
                                                   data_augment=False,
                                                   shuffle_size=10000)
        for step, (target_features, target_values, target_batch, target_id) in enumerate(query_db):
            feature_len += 1
            if step == 0:
                output = encode_network([target_features, target_values], training=False)

        dim = output.shape[1]
        target_data_feature_1 = np.zeros((feature_len, dim))
        target_data_batch_1 = np.zeros((feature_len))
        target_id_batch_1 = []
        for step, (target_features, target_values, target_batch, target_id) in enumerate(query_db):
            output = encode_network([target_features, target_values], training=False)
            output = tf.nn.l2_normalize(output, axis=-1)
            target_data_feature_1[step, :] = output
            target_data_batch_1[step] = target_batch
            target_id_batch_1.append(target_id.numpy()[0])

        target_data_feature.extend(target_data_feature_1)
        target_data_batch.extend(target_data_batch_1)
        target_data_id.extend(target_id_batch_1)

    ref_embedding = np.array(source_data_feature[:n_cells_for_ref])
    query_embedding = np.array(target_data_feature)
    source_data_id_subsample = source_data_id[:n_cells_for_ref]
    source_data_id_subsample = [i.decode("utf-8") for i in source_data_id_subsample]
    target_data_id = [i.decode("utf-8") for i in target_data_id]
    print('query embedding shape', query_embedding.shape)
    print('reference embedding shape', ref_embedding.shape)
    return ref_embedding, query_embedding,source_data_id_subsample,target_data_id # N*dim, 顺序按照adata1， adata2的cell 顺序；

def concerto_test_attention_0117(model_path:str, ref_tf_path:str, query_tf_path:str, super_parameters=None):
    if super_parameters is None:
        super_parameters = {'batch_size': 32, 'epoch': 1, 'lr': 1e-5,'drop_rate': 0.1}

    f = np.load(ref_tf_path + 'vocab_size.npz')
    vocab_size = int(f['vocab size'])
    encode_network = multi_embedding_attention_transfer(
        multi_max_features=[vocab_size],
        mult_feature_names=['RNA'],
        embedding_dims=128,
        include_attention=True,
        drop_rate=super_parameters['drop_rate'],
        head_1=128,
        head_2=128,
        head_3=128)

    tf_list_1 = [f for f in os.listdir(os.path.join(ref_tf_path)) if 'tfrecord' in f]
    train_source_list = []
    for i in tf_list_1:
        train_source_list.append(os.path.join(ref_tf_path, i))
    tf_list_2 = [f for f in os.listdir(os.path.join(query_tf_path)) if 'tfrecord' in f]
    train_target_list = []
    for i in tf_list_2:
        train_target_list.append(os.path.join(query_tf_path, i))
    weight_id_list = []
    weight_list = [f for f in os.listdir(model_path) if f.endswith('h5')]

    for id in weight_list:
        id_1 = re.findall('.*epoch(.*).h.*', id)  # f1
        weight_id_list.append(int(id_1[0]))
    # encode_network.load_weights(model_path + 'weight_encoder_epoch{}.h5'.format(max(weight_id_list)))
    encode_network.load_weights(model_path + 'weight_encoder_epoch{}.h5'.format(max(weight_id_list)), by_name=True) # yyyx0126, 支持多模态模型
    
    source_data_batch = []
    source_data_feature = []
    source_data_id = []
    batch_size = super_parameters['batch_size']
    for file in train_source_list:
        print(file)
        train_size = 0
        ref_db = create_classifier_dataset_multi(
            [file],
            batch_size=batch_size,
            is_training=False,
            data_augment=False,
            shuffle_size=10000)
        for step, (target_features, target_values, target_batch, target_id) in enumerate(ref_db):
            train_size += len(target_id)
            if step == 0:
                output = encode_network([target_features, target_values], training=False)

        dim = output.shape[1]
        source_data_feature_1 = np.zeros((train_size, dim))
        source_data_batch_1 = np.zeros((train_size))
        #source_id_batch_1 = np.zeros((train_size))
        source_id_batch_1 = []
        all_samples = 0
        for step, (target_features, target_values, target_batch, target_id) in enumerate(ref_db):
            output = encode_network([target_features, target_values], training=False)
            output = tf.nn.l2_normalize(output, axis=-1)
            source_data_feature_1[all_samples:all_samples + len(target_id), :] = output
            source_data_batch_1[all_samples:all_samples + len(target_id)] = target_batch
            #source_id_batch_1[all_samples:all_samples + len(target_id)] = target_id.numpy().decode("utf-8")
            source_id_batch_1.extend(list(target_id.numpy().astype('U')))

            all_samples += len(target_id)
        source_data_feature.extend(source_data_feature_1)
        source_data_batch.extend(source_data_batch_1)
        source_data_id.extend(source_id_batch_1)

    target_data_batch = []
    target_data_feature = []
    target_data_id = []
    for file in train_target_list:
        print(file)
        train_size = 0
        query_db = create_classifier_dataset_multi([file],
                                                   batch_size=batch_size,
                                                   is_training=False,
                                                   data_augment=False,
                                                   shuffle_size=10000)
        for step, (target_features, target_values, target_batch, target_id) in enumerate(query_db):
            train_size += len(target_id)
            if step == 0:
                output = encode_network([target_features, target_values], training=False)

        dim = output.shape[1]
        target_data_feature_1 = np.zeros((train_size, dim))
        target_data_batch_1 = np.zeros((train_size))
        #target_id_batch_1 = np.zeros((train_size))
        target_id_batch_1 = []
        all_samples = 0
        for step, (target_features, target_values, target_batch, target_id) in enumerate(query_db):

            output = encode_network([target_features, target_values], training=False)
            output = tf.nn.l2_normalize(output, axis=-1)
            target_data_feature_1[all_samples:all_samples + len(target_id), :] = output
            target_data_batch_1[all_samples:all_samples + len(target_id)] = target_batch
            #target_id_batch_1[all_samples:all_samples + len(target_id)] = target_id.numpy().decode("utf-8")
            target_id_batch_1.extend(list(target_id.numpy().astype('U')))
            all_samples += len(target_id)

        target_data_feature.extend(target_data_feature_1)
        target_data_batch.extend(target_data_batch_1)
        target_data_id.extend(target_id_batch_1)


    ref_embedding = np.array(source_data_feature)
    query_embedding = np.array(target_data_feature)
    source_data_id_subsample = source_data_id
    #source_data_id_subsample = [i.decode("utf-8") for i in source_data_id_subsample]
    #target_data_id = [i.decode("utf-8") for i in target_data_id]
    print('query embedding shape', query_embedding.shape)
    print('reference embedding shape', ref_embedding.shape)
    print('ref id length',len(source_data_id_subsample))
    print('query id length', len(target_data_id))
    return ref_embedding, query_embedding,source_data_id_subsample,target_data_id # N*dim, 顺序按照adata1， adata2的cell 顺序；

def concerto_test_ref_query(model_path:str, ref_tf_path:str, query_tf_path:str, super_parameters=None):
    if super_parameters is None:
        super_parameters = {'batch_size': 32, 'epoch': 1, 'lr': 1e-5,'drop_rate': 0.1}

    f = np.load(ref_tf_path + 'vocab_size.npz')
    vocab_size = int(f['vocab size'])
    encode_network = multi_embedding_attention_transfer(
        multi_max_features=[vocab_size],
        mult_feature_names=['RNA'],
        embedding_dims=128,
        include_attention=True,
        drop_rate=super_parameters['drop_rate'],
        head_1=128,
        head_2=128,
        head_3=128)

    tf_list_1 = [f for f in os.listdir(os.path.join(ref_tf_path)) if 'tfrecord' in f]
    train_source_list = []
    for i in tf_list_1:
        train_source_list.append(os.path.join(ref_tf_path, i))
    tf_list_2 = [f for f in os.listdir(os.path.join(query_tf_path)) if 'tfrecord' in f]
    train_target_list = []
    for i in tf_list_2:
        train_target_list.append(os.path.join(query_tf_path, i))
    weight_id_list = []
    weight_list = [f for f in os.listdir(model_path) if f.endswith('h5')]

    for id in weight_list:
        id_1 = re.findall('.*epoch(.*).h.*', id)  # f1
        weight_id_list.append(int(id_1[0]))
    # encode_network.load_weights(model_path + 'weight_encoder_epoch{}.h5'.format(max(weight_id_list)))
    encode_network.load_weights(model_path + 'weight_encoder_epoch{}.h5'.format(max(weight_id_list)), by_name=True) # yyyx0126, 支持多模态模型
    
    source_data_batch = []
    source_data_feature = []
    source_data_id = []
    batch_size = super_parameters['batch_size']
    for file in train_source_list:
        print(file)
        train_size = 0
        ref_db = create_classifier_dataset_multi(
            [file],
            batch_size=batch_size,
            is_training=False,
            data_augment=False,
            shuffle_size=10000)
        for step, (target_features, target_values, target_batch, target_id) in enumerate(ref_db):
            train_size += len(target_id)
            if step == 0:
                output = encode_network([target_features, target_values], training=False)

        dim = output.shape[1]
        source_data_feature_1 = np.zeros((train_size, dim))
        source_data_batch_1 = np.zeros((train_size))
        #source_id_batch_1 = np.zeros((train_size))
        source_id_batch_1 = []
        all_samples = 0
        for step, (target_features, target_values, target_batch, target_id) in enumerate(ref_db):
            output = encode_network([target_features, target_values], training=False)
            output = tf.nn.l2_normalize(output, axis=-1)
            source_data_feature_1[all_samples:all_samples + len(target_id), :] = output
            source_data_batch_1[all_samples:all_samples + len(target_id)] = target_batch
            #source_id_batch_1[all_samples:all_samples + len(target_id)] = target_id.numpy().decode("utf-8")
            source_id_batch_1.extend(list(target_id.numpy().astype('U')))

            all_samples += len(target_id)
        source_data_feature.extend(source_data_feature_1)
        source_data_batch.extend(source_data_batch_1)
        source_data_id.extend(source_id_batch_1)

    target_data_batch = []
    target_data_feature = []
    target_data_id = []
    for file in train_target_list:
        print(file)
        train_size = 0
        query_db = create_classifier_dataset_multi([file],
                                                   batch_size=batch_size,
                                                   is_training=False,
                                                   data_augment=False,
                                                   shuffle_size=10000)
        for step, (target_features, target_values, target_batch, target_id) in enumerate(query_db):
            train_size += len(target_id)
            if step == 0:
                output = encode_network([target_features, target_values], training=False)

        dim = output.shape[1]
        target_data_feature_1 = np.zeros((train_size, dim))
        target_data_batch_1 = np.zeros((train_size))
        #target_id_batch_1 = np.zeros((train_size))
        target_id_batch_1 = []
        all_samples = 0
        for step, (target_features, target_values, target_batch, target_id) in enumerate(query_db):

            output = encode_network([target_features, target_values], training=False)
            output = tf.nn.l2_normalize(output, axis=-1)
            target_data_feature_1[all_samples:all_samples + len(target_id), :] = output
            target_data_batch_1[all_samples:all_samples + len(target_id)] = target_batch
            #target_id_batch_1[all_samples:all_samples + len(target_id)] = target_id.numpy().decode("utf-8")
            target_id_batch_1.extend(list(target_id.numpy().astype('U')))
            all_samples += len(target_id)

        target_data_feature.extend(target_data_feature_1)
        target_data_batch.extend(target_data_batch_1)
        target_data_id.extend(target_id_batch_1)


    ref_embedding = np.array(source_data_feature)
    query_embedding = np.array(target_data_feature)
    source_data_id_subsample = source_data_id
    #source_data_id_subsample = [i.decode("utf-8") for i in source_data_id_subsample]
    #target_data_id = [i.decode("utf-8") for i in target_data_id]
    print('query embedding shape', query_embedding.shape)
    print('reference embedding shape', ref_embedding.shape)
    print('ref id length',len(source_data_id_subsample))
    print('query id length', len(target_data_id))
    return ref_embedding, query_embedding,source_data_id_subsample,target_data_id # N*dim, 顺序按照adata1， adata2的cell 顺序；



def concerto_test_ref(model_path:str, ref_tf_path:str, super_parameters=None,saved_weight_path=None):
    if super_parameters is None:
        super_parameters = {'batch_size': 128, 'epoch': 1, 'lr': 1e-5,'drop_rate': 0.1}

    f = np.load(ref_tf_path + 'vocab_size.npz')
    vocab_size = int(f['vocab size'])
    encode_network = multi_embedding_attention_transfer(
        multi_max_features=[vocab_size],
        mult_feature_names=['RNA'],
        embedding_dims=128,
        include_attention=True,
        drop_rate=super_parameters['drop_rate'],
        head_1=128,
        head_2=128,
        head_3=128)

    tf_list_1 = [f for f in os.listdir(os.path.join(ref_tf_path)) if 'tfrecord' in f]
    train_source_list = []
    for i in tf_list_1:
        train_source_list.append(os.path.join(ref_tf_path, i))

        
    if saved_weight_path is None:
        weight_id_list = []
        weight_list = [f for f in os.listdir(model_path) if f.endswith('h5')]
        for id in weight_list:
            id_1 = re.findall('.*epoch(.*).h.*', id)  # f1
            weight_id_list.append(int(id_1[0]))
            encode_network.load_weights(model_path + 'weight_encoder_epoch{}.h5'.format(max(weight_id_list)),
                                    by_name=True)

    else:
        encode_network.load_weights(saved_weight_path, by_name=True)
        print('load saved weight')
    
    source_data_batch = []
    source_data_feature = []
    source_data_id = []
    batch_size = super_parameters['batch_size']
    for file in train_source_list:
        print(file)
        train_size = 0
        ref_db = create_classifier_dataset_multi(
            [file],
            batch_size=batch_size,
            is_training=False,
            data_augment=False,
            shuffle_size=10000)
        for step, (target_features, target_values, target_batch, target_id) in enumerate(ref_db):
            train_size += len(target_id)
            if step == 0:
                output = encode_network([target_features, target_values], training=False)

        dim = output.shape[1]
        source_data_feature_1 = np.zeros((train_size, dim))
        source_data_batch_1 = np.zeros((train_size))
        #source_id_batch_1 = np.zeros((train_size))
        source_id_batch_1 = []
        all_samples = 0
        for step, (target_features, target_values, target_batch, target_id) in enumerate(ref_db):
            output = encode_network([target_features, target_values], training=False)
            output = tf.nn.l2_normalize(output, axis=-1)
            source_data_feature_1[all_samples:all_samples + len(target_id), :] = output
            source_data_batch_1[all_samples:all_samples + len(target_id)] = target_batch
            #source_id_batch_1[all_samples:all_samples + len(target_id)] = target_id.numpy().decode("utf-8")
            source_id_batch_1.extend(list(target_id.numpy().astype('U')))

            all_samples += len(target_id)
        source_data_feature.extend(source_data_feature_1)
        source_data_batch.extend(source_data_batch_1)
        source_data_id.extend(source_id_batch_1)

    ref_embedding = np.array(source_data_feature)    
    print('reference embedding shape', ref_embedding.shape)
    print('ref id length',len(source_data_id))    
    return ref_embedding, source_data_id 




def concerto_test_multimodal(model_path: str, RNA_tf_path: str, Protein_tf_path: str, n_cells_for_sample=None,super_parameters=None,
                             saved_weight_path=None):
    if super_parameters is None:
        super_parameters = {'batch_size': 32, 'epoch': 1, 'lr': 1e-5, 'drop_rate': 0.1}
    
    batch_size = super_parameters['batch_size']
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(Protein_tf_path, 'vocab_size.npz'))
    vocab_size_Protein = int(f['vocab size'])
    encode_network = multi_embedding_attention_transfer_explainability(
        multi_max_features=[vocab_size_RNA, vocab_size_Protein],
        mult_feature_names=['RNA', 'Protein'],
        embedding_dims=128,
        include_attention=True,
        drop_rate=super_parameters['drop_rate'],
        head_1=128,
        head_2=128,
        head_3=128)

    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_Protein = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_Protein.append(os.path.join(Protein_tf_path, i))



    if saved_weight_path is None:
        weight_id_list = []
        weight_list = [f for f in os.listdir(model_path) if f.endswith('h5')]
        for id in weight_list:
            id_1 = re.findall('.*epoch(.*).h.*', id)  # f1
            weight_id_list.append(int(id_1[0]))
        encode_network.load_weights(model_path + 'weight_encoder_epoch{}.h5'.format(max(weight_id_list)),
                                    by_name=True)

    else:
        encode_network.load_weights(saved_weight_path, by_name=True)
        print('load saved weight')

    source_data_batch = []
    source_data_feature = []
    RNA_id_all = []
    attention_output_RNA_all = []
    attention_output_Protein_all = []
    for RNA_file, Protein_file in zip(train_source_list_RNA, train_source_list_Protein):
        print(RNA_file)
        print(Protein_file)
        train_size = 0
        train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                       batch_size=batch_size,
                                                       is_training=False,
                                                       data_augment=False,
                                                       shuffle_size=10000,
                                                       )
        train_db_Protein = create_classifier_dataset_multi([Protein_file],
                                                           batch_size=batch_size,
                                                           is_training=False,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
        step = 0
        for (source_features_RNA, source_values_RNA,
             source_batch_RNA, source_id_RNA), \
            (source_features_protein, source_values_protein,
             source_batch_Protein, source_id_Protein) \
                in (zip(train_db_RNA, train_db_Protein)):
            #train_size += len(source_id_RNA)
            if step == 0:
                encode_output, attention_output = encode_network([[source_features_RNA, source_features_protein],
                                                                  [source_values_RNA, source_values_protein]],
                                                                 training=False)
                break

        dim = encode_output.shape[1]
        if n_cells_for_sample is None:            
            feature_len = 1000000
        else:            
            n_cells_for_sample_1 = n_cells_for_sample//8
            feature_len = n_cells_for_sample_1 // batch_size * batch_size
        
        source_data_feature_1 = np.zeros((feature_len, dim))
        source_data_batch_1 = np.zeros((feature_len))
        attention_output_RNA = np.zeros((feature_len, vocab_size_RNA, 1))
        attention_output_Protein = np.zeros((feature_len, vocab_size_Protein, 1))
        print('feature_len:',feature_len)
        RNA_id = []
        all_samples = 0
        for (source_features_RNA, source_values_RNA,
             source_batch_RNA, source_id_RNA), \
            (source_features_protein, source_values_protein,
             source_batch_Protein, source_id_Protein) \
                in (zip(train_db_RNA, train_db_Protein)):
            if all_samples  >= feature_len:
                break
            encode_output, attention_output = encode_network([[source_features_RNA, source_features_protein],
                                                              [source_values_RNA, source_values_protein]],
                                                             training=False)

            encode_output = tf.nn.l2_normalize(encode_output, axis=-1)
            source_data_feature_1[all_samples:all_samples + len(source_id_RNA), :] = encode_output
            source_data_batch_1[all_samples:all_samples + len(source_id_RNA)] = source_batch_RNA
            RNA_id.extend(list(source_id_RNA.numpy().astype('U')))
            attention_output_RNA[all_samples:all_samples + len(source_id_RNA), :, :] = attention_output[0]
            attention_output_Protein[all_samples:all_samples + len(source_id_RNA), :, :] = attention_output[1]
            all_samples += len(source_id_RNA)
            print('all_samples num:{}'.format(all_samples))

        source_data_feature.extend(source_data_feature_1[:all_samples])
        source_data_batch.extend(source_data_batch_1[:all_samples])
        RNA_id_all.extend(RNA_id[:all_samples])
        attention_output_RNA_all.extend(attention_output_RNA[:all_samples])
        attention_output_Protein_all.extend(attention_output_Protein[:all_samples])

    source_data_feature = np.array(source_data_feature).astype('float32')
    source_data_batch = np.array(source_data_batch).astype('int32')
    attention_weight = {'attention_output_RNA': attention_output_RNA_all,
                        'attention_output_Protein': attention_output_Protein_all}
    #np.savez_compressed('./multi_attention.npz', **attention_weight)
    return source_data_feature, source_data_batch, RNA_id_all, attention_weight


def concerto_test_multimodal_project(model_path: str, RNA_tf_path: str, Protein_tf_path: str, super_parameters=None,saved_weight_path = None):
    if super_parameters is None:
        super_parameters = {'batch_size': 32, 'epoch': 1, 'lr': 1e-5, 'drop_rate': 0.1}

    batch_size = super_parameters['batch_size']
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(Protein_tf_path, 'vocab_size.npz'))
    vocab_size_Protein = int(f['vocab size'])
    encode_network = multi_embedding_attention_transfer_explainability(
        multi_max_features=[vocab_size_RNA,vocab_size_Protein],
        mult_feature_names=['RNA','Protein'],
        embedding_dims=128,
        include_attention=True,
        drop_rate=super_parameters['drop_rate'],
        head_1=128,
        head_2=128,
        head_3=128)
    encode_network_RNA = multi_embedding_attention_transfer_explainability(
        multi_max_features=[vocab_size_RNA],
        mult_feature_names=['RNA'],
        embedding_dims=128,
        include_attention=True,
        drop_rate=super_parameters['drop_rate'],
        head_1=128,
        head_2=128,
        head_3=128)


    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_Protein = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_Protein.append(os.path.join(Protein_tf_path, i))

    weight_id_list = []
    weight_list = [f for f in os.listdir(model_path) if f.endswith('h5')]

    for id in weight_list:
        id_1 = re.findall('.*epoch(.*).h.*', id)  # f1
        weight_id_list.append(int(id_1[0]))
    
    if  saved_weight_path is None:
        encode_network.load_weights(model_path + 'weight_encoder_epoch{}.h5'.format(max(weight_id_list)),
                                    by_name=True)
        encode_network_RNA.load_weights(model_path + 'weight_encoder_epoch{}.h5'.format(max(weight_id_list)),
                                    by_name=True)

    else:
        encode_network.load_weights(saved_weight_path,by_name=True)
        encode_network_RNA.load_weights(saved_weight_path, by_name=True)
        

    source_data_batch = []
    source_data_feature = []
    source_data_feature_RNA = []    
    RNA_id_all = []

    for RNA_file, Protein_file in zip(train_source_list_RNA, train_source_list_Protein):
        print(RNA_file)
        print(Protein_file)
        train_size = 0
        train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                       batch_size=batch_size,
                                                       is_training=False,
                                                       data_augment=False,
                                                       shuffle_size=10000,
                                                       )
        train_db_Protein = create_classifier_dataset_multi([Protein_file],
                                                           batch_size=batch_size,
                                                           is_training=False,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
        step = 0
        for (source_features_RNA, source_values_RNA,
             source_batch_RNA, source_id_RNA), \
            (source_features_protein, source_values_protein,
             source_batch_Protein, source_id_Protein) \
                in (zip(train_db_RNA, train_db_Protein)):
            train_size += len(source_id_RNA)
            if step == 0:
                encode_output, attention_output = encode_network([[source_features_RNA, source_features_protein],
                                                                        [source_values_RNA, source_values_protein]],
                                                                       training=False)

        dim = encode_output.shape[1]
        source_data_feature_1 = np.zeros((train_size, dim))
        source_data_feature_RNA_1 = np.zeros((train_size, dim))        
        source_data_batch_1 = np.zeros((train_size))
        RNA_id = []
        all_samples = 0
        for (source_features_RNA, source_values_RNA,
             source_batch_RNA, source_id_RNA), \
            (source_features_protein, source_values_protein,
             source_batch_Protein, source_id_Protein) \
                in (zip(train_db_RNA, train_db_Protein)):
            encode_output, attention_output = encode_network([[source_features_RNA, source_features_protein],
                                                              [source_values_RNA, source_values_protein]],
                                                             training=False)
            encode_output_RNA, attention_output_ = encode_network_RNA([[source_features_RNA],
                                                              [source_values_RNA]],
                                                             training=False)


            encode_output = tf.nn.l2_normalize(encode_output, axis=-1)
            source_data_feature_1[all_samples:all_samples + len(source_id_RNA), :] = encode_output
            source_data_feature_RNA_1[all_samples:all_samples + len(source_id_RNA), :] = encode_output_RNA            
            source_data_batch_1[all_samples:all_samples + len(source_id_RNA)] = source_batch_RNA
            RNA_id.extend(list(source_id_RNA.numpy().astype('U')))
            all_samples += len(source_id_RNA)
            print('all_samples num:{}'.format(all_samples))

        source_data_feature.extend(source_data_feature_1)
        source_data_feature_RNA.extend(source_data_feature_RNA_1)        
        source_data_batch.extend(source_data_batch_1)
        RNA_id_all.extend(RNA_id)

    source_data_feature = np.array(source_data_feature).astype('float32')
    source_data_feature_RNA = np.array(source_data_feature_RNA).astype('float32')    
    source_data_batch = np.array(source_data_batch).astype('int32')

    return source_data_feature,source_data_feature_RNA, source_data_batch, RNA_id_all




def knn_classifier(ref_embedding, query_embedding, ref_anndata, source_data_id, column_name,k, num_chunks=100):
    '''
    return :
        target_neighbor: predicted label
        traget_prob: confidence score
    '''
    train_features = tf.transpose(ref_embedding)
    num_test_images = int(query_embedding.shape[0])
    imgs_per_chunk = num_test_images // num_chunks
    if imgs_per_chunk == 0:
        imgs_per_chunk = 10

    print(num_test_images, imgs_per_chunk)
    ref_anndata = ref_anndata[source_data_id]
    train_labels = ref_anndata.obs[column_name].tolist()
    target_pred_labels = []
    target_pred_prob = []
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = query_embedding[
                   idx: min((idx + imgs_per_chunk), num_test_images), :
                   ]
        # targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        similarity = tf.matmul(features, train_features)
        target_distances, target_indices = tf.math.top_k(similarity, k, sorted=True)

        for distances, indices in zip(target_distances, target_indices):
            selected_label = {}
            selected_count = {}
            count = 0
            for distance, index in zip(distances, indices):
                label = train_labels[index]
                weight = distance
                if label not in selected_label:
                    selected_label[label] = 0
                    selected_count[label] = 0
                selected_label[label] += weight
                selected_count[label] += 1
                count += 1

            filter_label_list = sorted(selected_label.items(), key=lambda x: x[1], reverse=True)
            target_pred_labels.append(filter_label_list[0][0])

            prob = selected_label[filter_label_list[0][0]] / selected_count[filter_label_list[0][0]]
            target_pred_prob.append(prob)

    target_neighbor = np.array(target_pred_labels)
    target_prob = np.array(target_pred_prob)

    return target_neighbor, target_prob #返回预测的label和置信度

def knn_classifier_faiss(ref_embedding, query_embedding, ref_anndata, source_data_id, column_name,k, num_chunks=100):
    '''
    return :
        target_neighbor: predicted label
        traget_prob: confidence score
    '''
    import faiss
    
    ref_embedding = ref_embedding.astype('float32')
    query_embedding = query_embedding.astype('float32')
    n, dim = ref_embedding.shape[0], ref_embedding.shape[1]
    index = faiss.IndexFlatIP(dim)
    #index = faiss.index_cpu_to_all_gpus(index)
    index.add(ref_embedding)
    ref_anndata = ref_anndata[source_data_id]
    train_labels = ref_anndata.obs[column_name].tolist()
    target_pred_labels = []
    target_pred_prob = []

    target_distances, target_indices= index.search(query_embedding, k)  # Sample itself is included

    for distances, indices in zip(target_distances, target_indices):
        selected_label = {}
        selected_count = {}
        count = 0
        for distance, index in zip(distances, indices):
            label = train_labels[index]
            weight = distance
            if label not in selected_label:
                selected_label[label] = 0
                selected_count[label] = 0
            selected_label[label] += weight
            selected_count[label] += 1
            count += 1

        filter_label_list = sorted(selected_label.items(), key=lambda x: x[1], reverse=True)
        target_pred_labels.append(filter_label_list[0][0])

        prob = selected_label[filter_label_list[0][0]] / selected_count[filter_label_list[0][0]]
        target_pred_prob.append(prob)

    target_neighbor = np.array(target_pred_labels)
    target_prob = np.array(target_pred_prob)

    return target_neighbor, target_prob #返回预测的label和置信度

# GPU control
def concerto_GPU():
    return


if __name__ == '__main__':
    ref_path = ''
    query_path = ''