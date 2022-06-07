import argparse
import os
import random as rn

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import sys
import tensorflow as tf
import umap
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from sklearn import metrics
from sklearn import mixture
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
#from sklearn.utils.linear_assignment_ import linear_assignment
from time import time

try:
    # from MulticoreTSNE import MulticoreTSNE as TSNE
    from openTSNE import TSNE as TSNE
except BaseException:
    print("Missing MulticoreTSNE package.. Only important if evaluating other manifold learners.")


def cluster_manifold_in_embedding(hidden_layer,
                                  y_labels,
                                  label_names=None,
                                  manifold_learner='UMAP',
                                  umap_min_distance=0.0,
                                  umap_metric='euclidean',
                                  umap_dim=2,
                                  umap_neighbors=10,
                                  cluster='GMM',
                                  n_clusters=10,
                                  dataset_name='Single Cell'
                                  ):
    """
    Clustering on new manifold of embeddings
    :param hidden_layer:
    :param y_labels:
    :param label_names:
    :param manifold_learner:
    :param umap_min_distance:
    :param umap_metric:
    :param umap_dim:
    :param umap_neighbors:
    :param cluster:
    :param n_clusters:
    :param dataset_name:
    :return:
    """

    # N2D: (Not Too) Deep Clustering via Clustering the Local Manifold of an Autoencoded Embedding
    # Ryan McConville, Raul Santos-Rodriguez, Robert J Piechocki, Ian Craddock
    # https://arxiv.org/abs/1908.05968
    rn.seed(0)
    tf.random.set_seed(0)
    np.random.seed(0)
    np.set_printoptions(threshold=sys.maxsize)

    # find manifold on autoencoded embedding
    if manifold_learner:
        if manifold_learner == 'UMAP':
            md = float(umap_min_distance)
            hle = umap.UMAP(
                random_state=0,
                metric=umap_metric,
                n_components=umap_dim,
                n_neighbors=umap_neighbors,
                min_dist=md).fit_transform(hidden_layer)
        elif manifold_learner == 'LLE':
            hle = LocallyLinearEmbedding(
                n_components=umap_dim,
                n_neighbors=umap_neighbors).fit_transform(hidden_layer)
        elif manifold_learner == 'tSNE':
            hle = TSNE(
                n_components=umap_dim,
                n_jobs=16,
                random_state=0,
                verbose=0).fit_transform(hidden_layer)
        elif manifold_learner == 'isomap':
            hle = Isomap(
                n_components=umap_dim,
                n_neighbors=5,
            ).fit_transform(hidden_layer)
        elif manifold_learner == 'None':
            hle = hidden_layer


    print("Hidden layer: ", hidden_layer.shape)
    #print("Hidden layer e: ", hle.shape)
    print("y_labels: ", y_labels.shape)

    # clustering on new manifold of autoencoded embedding
    if cluster == 'GMM':
        gmm = mixture.GaussianMixture(
            covariance_type='full',
            n_components=n_clusters,
            random_state=0)
        gmm.fit(hle)
        y_pred_prob = gmm.predict_proba(hle)
        y_pred = y_pred_prob.argmax(1)
    elif cluster == 'KM':
        km = KMeans(
            init='k-means++',
            n_clusters=n_clusters,
            random_state=0,
            n_init=20)
        y_pred = km.fit_predict(hle)
    elif cluster == 'SC':
        sc = SpectralClustering(
            n_clusters=n_clusters,
            random_state=0,
            affinity='nearest_neighbors')
        y_pred = sc.fit_predict(hle)

    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_labels)

    acc = np.round(cluster_acc(y_true, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y_true, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y_true, y_pred), 5)
    if manifold_learner is None:
        manifold_learner = ''
    # print(dataset_name + " | " + manifold_learner +
    #       " on autoencoded embedding with " + cluster + ".")
    # print('=' * 80)
    # print(acc)
    # print(nmi)
    # print(ari)
    # print('=' * 80)

    # if args.visualize:
    #     plot(hle, y, 'n2d', label_names)
    #     y_pred_viz, _, _ = best_cluster_fit(y, y_pred)
    #     plot(hle, y_pred_viz, 'n2d-predicted', label_names)

    return y_pred, acc, nmi, ari


# def best_cluster_fit(y_true, y_pred):
#     y_true = y_true.astype(np.int64)
#     D = max(y_pred.max(), y_true.max()) + 1
#     w = np.zeros((D, D), dtype=np.int64)
#     for i in range(y_pred.size):
#         w[y_pred[i], y_true[i]] += 1
#
#     ind = linear_assignment(w.max() - w)
#     best_fit = []
#     for i in range(y_pred.size):
#         for j in range(len(ind)):
#             if ind[j][0] == y_pred[i]:
#                 best_fit.append(ind[j][1])
#     return best_fit, ind, w


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    #assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[int(y_pred[i]), int(y_true[i])] += 1
    #from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment
    ind = linear_sum_assignment(w.max() - w)
    ind_1 = np.concatenate((np.array([ind[0]]),np.array([ind[1]])),axis=0)
    ind_1 = np.transpose(ind_1)
    ac = sum([w[i, j] for i, j in ind_1]) * 1.0 / y_pred.size
    return ac


# def cluster_acc(y_true, y_pred):
#     _, ind, w = best_cluster_fit(y_true, y_pred)
#     return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

# def plot(x, y, plot_id, names=None):
#     viz_df = pd.DataFrame(data=x[:5000])
#     viz_df['Label'] = y[:5000]
#     if names is not None:
#         viz_df['Label'] = viz_df['Label'].map(names)
#
#     viz_df.to_csv(args.save_dir + '/' + args.dataset + '.csv')
#     plt.subplots(figsize=(8, 5))
#     sns.scatterplot(x=0, y=1, hue='Label', legend='full', hue_order=sorted(viz_df['Label'].unique()),
#                     palette=sns.color_palette("hls", n_colors=args.n_clusters),
#                     alpha=.5,
#                     data=viz_df)
#     l = plt.legend(bbox_to_anchor=(-.1, 1.00, 1.1, .5), loc="lower left", markerfirst=True,
#                    mode="expand", borderaxespad=0, ncol=args.n_clusters + 1, handletextpad=0.01, )
#
#     l.texts[0].set_text("")
#     plt.ylabel("")
#     plt.xlabel("")
#     plt.tight_layout()
#     plt.show()
#     # plt.savefig(args.save_dir + '/' + args.dataset +
#     #             '-' + plot_id + '.png', dpi=300)
#     plt.clf()


# def autoencoder(dims, act='relu'):
#     n_stacks = len(dims) - 1
#     x = Input(shape=(dims[0],), name='input')
#     h = x
#     for i in range(n_stacks - 1):
#         h = Dense(dims[i + 1], activation=act, name='encoder_%d' % i)(h)
#     h = Dense(dims[-1], name='encoder_%d' % (n_stacks - 1))(h)
#     for i in range(n_stacks - 1, 0, -1):
#         h = Dense(dims[i], activation=act, name='decoder_%d' % i)(h)
#     h = Dense(dims[0], name='decoder_0')(h)
#
#     return Model(inputs=x, outputs=h)

#
# if __name__ == "__main__":
#
#     parser = argparse.ArgumentParser(
#         description='(Not Too) Deep',
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('dataset', default='mnist', )
#     parser.add_argument('gpu', default=0, )
#     parser.add_argument('--n_clusters', default=10, type=int)
#     parser.add_argument('--batch_size', default=256, type=int)
#     parser.add_argument('--pretrain_epochs', default=1000, type=int)
#     parser.add_argument('--ae_weights', default=None)
#     parser.add_argument('--save_dir', default='results/n2d')
#     parser.add_argument('--umap_dim', default=2, type=int)
#     parser.add_argument('--umap_neighbors', default=10, type=int)
#     parser.add_argument('--umap_min_dist', default="0.00", type=str)
#     parser.add_argument('--umap_metric', default='euclidean', type=str)
#     parser.add_argument('--cluster', default='GMM', type=str)
#     parser.add_argument('--eval_all', default=False, action='store_true')
#     parser.add_argument('--manifold_learner', default='UMAP', type=str)
#     parser.add_argument('--visualize', default=False, action='store_true')
#     args = parser.parse_args()
#     print(args)
#
#     optimizer = 'adam'
#     from datasets import load_mnist, load_mnist_test, load_usps, load_pendigits, load_fashion, load_har
#
#     label_names = None
#     if args.dataset == 'mnist':
#         x, y = load_mnist()
#     elif args.dataset == 'mnist-test':
#         x, y = load_mnist_test()
#     elif args.dataset == 'usps':
#         x, y = load_usps()
#     elif args.dataset == 'pendigits':
#         x, y = load_pendigits()
#     elif args.dataset == 'fashion':
#         x, y, label_names = load_fashion()
#     elif args.dataset == 'har':
#         x, y, label_names = load_har()
#
#     shape = [x.shape[-1], 500, 500, 2000, args.n_clusters]
#     autoencoder = autoencoder(shape)
#
#     hidden = autoencoder.get_layer(name='encoder_%d' % (len(shape) - 2)).output
#     encoder = Model(inputs=autoencoder.input, outputs=hidden)
#
#     pretrain_time = time()
#
#     # Pretrain autoencoders before clustering
#     if args.ae_weights is None:
#         autoencoder.compile(loss='mse', optimizer=optimizer)
#         autoencoder.fit(
#             x,
#             x,
#             batch_size=args.batch_size,
#             epochs=args.pretrain_epochs,
#             verbose=0)
#         pretrain_time = time() - pretrain_time
#         autoencoder.save_weights('weights/' +
#                                  args.dataset +
#                                  "-" +
#                                  str(args.pretrain_epochs) +
#                                  '-ae_weights.h5')
#         print("Time to train the autoencoder: " + str(pretrain_time))
#     else:
#         autoencoder.load_weights('weights/' + args.ae_weights)
#
#     if not os.path.exists(args.save_dir):
#         os.makedirs(args.save_dir)
#     with open(args.save_dir + '/args.txt', 'w') as f:
#         f.write("\n".join(sys.argv))
#
#     hl = encoder.predict(x)
#     if args.eval_all:
#         eval_other_methods(x, y, label_names)
#     clusters, t_acc, t_nmi, t_ari = cluster_manifold_in_embedding(
#         hl, y, label_names)
#     np.savetxt(args.save_dir + "/" + args.dataset + '-clusters.txt', clusters, fmt='%i', delimiter=',')
