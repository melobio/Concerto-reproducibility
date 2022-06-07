import numpy as np
import tensorflow as tf

def tf_index_select(input_, dim, indices):
    """
    input_(tensor): input tensor
    dim(int): dimension
    indices(list): selected indices list
    """
    shape = input_.get_shape().as_list()
    if dim == -1:
        dim = len(shape)-1
    shape[dim] = 1

    tmp = []
    for idx in indices:
        begin = [0]*len(shape)
        begin[dim] = idx
        tmp.append(tf.slice(input_, begin, shape))
    res = tf.concat(tmp, axis=dim)

    return res


class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        self.n = n
        self.dim = dim
        self.features = np.zeros((self.n, self.dim), dtype=np.float) #torch.FloatTensor(self.n, self.dim)
        self.targets = np.zeros((self.n), dtype=np.float)            #torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def weighted_knn(self, predictions):
        # perform weighted knn
        retrieval_one_hot = tf.zeros([self.K, self.C])
        batchSize = tf.shape(predictions)[0]

        correlation = tf.matmul(predictions, self.features, transpose_b=True)
        yd, yi = tf.nn.top_k(correlation, self.K, sorted=True)

        #candidates = self.targets.expand_dims(batchSize, -1)

        candidates = tf.expand_dims(self.targets, 2)
        retrieval = tf.gather(candidates, 1, yi)

        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)

        yd_transform = yd.clone().div_(self.temperature).exp_()
        probs = tf.reduce_sum(tf.matmul(retrieval_one_hot.view(batchSize, -1, self.C),
                                    yd_transform.view(batchSize, -1, 1)), 1)

        _, class_preds = probs.sort(1, True)

        class_pred = class_preds[:, 0]

        return class_pred

    def knn(self, predictions):
        # perform knn
        correlation = np.matmul(predictions, self.features)
        sample_pred = np.argmax(correlation, dim=1)
        class_pred = tf_index_select(self.targets, 0, sample_pred)
        return class_pred

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        # mine the topk nearest neighbors for every sample
        import faiss
        features = self.features
        features = np.array(features).astype('float32')

        n, dim = features.shape[0], features.shape[1]
        index = faiss.IndexFlatIP(dim)
        #index = faiss.IndexFlatL2(dim)
        #index = faiss.index_cpu_to_all_gpus(index)

        index.add(features)
        distances, indices = index.search(features, topk + 1)  # Sample itself is included

        # evaluate
        if calculate_accuracy:
            targets = self.targets
            neighbor_targets = np.take(targets, indices[:, 1:], axis=0)  # Exclude sample itself for eval
            anchor_targets = np.repeat(targets.reshape(-1, 1), topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)
            return indices, accuracy,neighbor_targets[:,0]

        else:
            return indices

    def reset(self):
        self.ptr = 0

    def update(self, features:np.ndarray, targets:np.ndarray):
        b = features.shape[0]

        assert (b + self.ptr <= self.n)

        self.features[self.ptr:self.ptr + b] = features
        self.targets[self.ptr:self.ptr + b] = targets

        self.ptr += b

