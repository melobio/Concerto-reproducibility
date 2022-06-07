import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics import f1_score, accuracy_score
import umap.umap_ as umap # yyyx 0112
import tensorflow as tf
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import Isomap
from sklearn import mixture
from openTSNE import TSNE as TSNE

# Adjusted Rand index 调整兰德系数
ARI = adjusted_rand_score

# Mutual Information based scores 互信息
NMI = normalized_mutual_info_score

#Silhouette Coefficient 轮廓系数
SS = silhouette_score



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
    # assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[int(y_pred[i]), int(y_true[i])] += 1
    # from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_sum_assignment(w.max() - w)
    ind_1 = np.concatenate((np.array([ind[0]]), np.array([ind[1]])), axis=0)
    ind_1 = np.transpose(ind_1)
    ac = sum([w[i, j] for i, j in ind_1]) * 1.0 / y_pred.size
    return ac

def cal_F1(true, pred):
    #acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average=None)
    #print("F1: ", list(f1))
    f1_median = np.median(f1)
    f1_macro = f1_score(true, pred, average='macro')
    f1_micro = f1_score(true, pred, average='micro')
    return f1_median, f1_macro, f1_micro


def knn_classifier(train_features, train_labels, test_features, test_labels, k, num_chunks=100):
    train_features = tf.transpose(train_features)

    num_test_images = int(tf.shape(test_labels))
    imgs_per_chunk = num_test_images // num_chunks
    if imgs_per_chunk == 0:
        imgs_per_chunk = 10

    print(num_test_images, imgs_per_chunk)

    target_pred_labels = []
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
                   idx: min((idx + imgs_per_chunk), num_test_images), :
                   ]
        # targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        similarity = tf.matmul(features, train_features)
        target_distances, target_indices = tf.math.top_k(similarity, k, sorted=True)

        for distances, indices in zip(target_distances, target_indices):
            selected_label = {}
            count = 0
            for distance, index in zip(distances, indices):
                label = train_labels[index]
                weight = distance
                if label not in selected_label:
                    selected_label[label] = 0
                selected_label[label] += weight
                count += 1

            filter_label_list = sorted(selected_label.items(), key=lambda x: x[1], reverse=True)
            target_pred_labels.append(filter_label_list[0][0])

    target_labels = np.array(test_labels, dtype=np.int)
    target_neighbor = np.array(target_pred_labels, dtype=np.int)

    return target_labels, target_neighbor



def cluster_plot(features,
                 labels,
                 cluster='SC',
                 n_clusters=8,
                 k = 10,
                 manifold_learner=None,
                 umap_min_distance=0.0,
                 umap_metric='euclidean',
                 umap_dim=2,
                 umap_neighbors=10,
                 ):
    hidden_layer = features
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
    else:
        hle = hidden_layer

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

    target_labels, target_preds = knn_classifier(features, y_pred, features, y_pred, k=k, num_chunks=100)
    acc = np.round(cluster_acc(labels, target_preds), 5)
    nmi = np.round(normalized_mutual_info_score(labels, target_preds), 5)
    ari = np.round(adjusted_rand_score(labels, target_preds), 5)
    print('knn(K= %.1f, n_cluster: %.1f): acc = %.5f , ari = %.5f , nmi = %.5f' % (k, n_clusters, acc, ari, nmi), '.')

    return nmi,ari




