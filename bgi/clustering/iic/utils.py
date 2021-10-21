import os
import pickle
import numpy as np
from scipy.optimize import linear_sum_assignment


def save_performance(perf, epoch, save_path):
    """
    :param perf: performance dictionary
    :param epoch: epoch number
    :param save_path: path to save plot to. if None, plot will be drawn
    :return: none
    """
    # return if save path is None
    if save_path is None:
        return

    # loop over the metrics
    for metric in perf.keys():

        # loop over the data splits
        for split in perf[metric].keys():

            # trim data to utilized epochs
            perf[metric][split] = perf[metric][split][:epoch]
            assert len(perf[metric][split]) == epoch

    # create the file name
    f_name = os.path.join(save_path, 'perf.pkl')

    # pickle it
    with open(f_name, 'wb') as f:
        pickle.dump(perf, f, pickle.HIGHEST_PROTOCOL)

    # make sure it worked
    with open(f_name, 'rb') as f:
        perf_load = pickle.load(f)
    assert str(perf) == str(perf_load), 'performance saving failed'


def unsupervised_labels(y, y_hat, num_classes, num_clusters):
    """
    :param y: true label
    :param y_hat: concentration parameter
    :param num_classes: number of classes (determined by data)
    :param num_clusters: number of clusters (determined by model)
    :return: classification error rate
    """
    assert num_classes == num_clusters

    # initialize count matrix
    cnt_mtx = np.zeros([num_classes, num_classes])

    # fill in matrix
    for i in range(len(y)):
        cnt_mtx[int(y_hat[i]), int(y[i])] += 1

    # find optimal permutation
    row_ind, col_ind = linear_sum_assignment(-cnt_mtx)

    # compute error
    error = 1 - cnt_mtx[row_ind, col_ind].sum() / cnt_mtx.sum()

    # print results
    print('Classification error = {:.4f}'.format(error))

    return error
