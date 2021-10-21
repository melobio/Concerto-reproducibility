import tensorflow as tf

EPS = 1e-8
MAX = 1e9


def entropy(x, input_as_probabilities):
    """ 
    Helper function to compute the entropy over the batch 

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ = tf.clip_by_value(x, clip_value_min=EPS, clip_value_max=MAX)
        entropy = x_ * tf.math.log(x_)
    else:
        entropy = tf.nn.softmax(x, dim=1) * tf.nn.log_softmax(x, dim=1)

    if len(tf.shape(entropy)) == 2:  # Sample-wise entropy
        return -tf.reduce_mean(tf.reduce_sum(entropy, axis=1))
    elif len(tf.shape(entropy)) == 1:  # Distribution-wise entropy
        return - tf.reduce_sum(entropy)
    else:
        raise ValueError('Input tensor is %d-Dimensional')


def sequence_mask(input_lengths, max_len=None, expand=True):
    if max_len is None:
        max_len = tf.reduce_max(input_lengths)

    if expand:
        return tf.expand_dims(tf.sequence_mask(input_lengths, max_len, dtype=tf.float32), axis=-1)
    return tf.sequence_mask(input_lengths, max_len, dtype=tf.float32)


def MaskedCrossEntropyLoss(outputs, targets, lengths=None, mask=None, max_len=None):
    if lengths is None and mask is None:
        raise RuntimeError('Please provide either lengths or mask')

    # [batch_size, time_length]
    if mask is None:
        mask = sequence_mask(lengths, max_len, False)

    # One hot encode targets (outputs.shape[-1] = hparams.quantize_channels)
    targets_ = tf.one_hot(targets, depth=tf.shape(outputs)[-1])

    with tf.control_dependencies([tf.assert_equal(tf.shape(outputs), tf.shape(targets_))]):
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs, labels=targets_)

    # print("mask: ", mask)
    # print("losses: ", losses)

    # with tf.control_dependencies([tf.assert_equal(tf.shape(mask), tf.shape(losses))]):
    #     masked_loss = losses * mask

    masked_loss = tf.boolean_mask(losses, mask)

    # print("masked_loss: ", masked_loss)
    # if tf.shape(masked_loss)[0] > 0:
    #     return tf.reduce_sum(masked_loss) / tf.math.count_nonzero(masked_loss, dtype=tf.float32)
    # else:
    #     return tf.zeros((lengths))

    return tf.reduce_sum(masked_loss) / tf.math.count_nonzero(masked_loss, dtype=tf.float32)


def SCAN_Loss(anchors, neighbors, entropy_weight=1.0):
    """
    SCAN loss
    :param anchors:
    :param neighbors:
    :param entropy_weight:
    :return:
    """
    batch_size, n = tf.shape(anchors)

    anchors_prob = tf.nn.softmax(anchors) # shape[batch,classes]
    positives_prob = tf.nn.softmax(neighbors) # shape[batch,classes]

    # Similarity in output space
    anchors_prob = tf.reshape(anchors_prob, (batch_size, 1, n)) # shape[batch,1,classes]
    positives_prob = tf.reshape(positives_prob, (batch_size, n, 1)) # shape[batch,classes,1]
    similarity = tf.matmul(anchors_prob, positives_prob) # shape[batch,1,1]

    ones = tf.ones_like(similarity)
    consistency_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = similarity,labels = ones)

    # Entropy loss
    entropy_loss = entropy(tf.reduce_mean(anchors_prob, 0), input_as_probabilities=True)

    # Total loss
    total_loss = consistency_loss - entropy_weight * entropy_loss

    return total_loss, consistency_loss, entropy_loss


class ConfidenceBasedCE(object):
    def __init__(self, threshold=0.90, apply_class_balancing=False):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = None #MaskedCrossEntropyLoss()
        self.threshold = threshold
        self.apply_class_balancing = apply_class_balancing

    def __call__(self, anchors_weak, anchors_strong):
        """
        Loss function during self-labeling

        input: logits for original samples and for its strong augmentations
        output: cross entropy
        """
        # Retrieve target and mask based on weakly augmentated anchors
        anchors_weak = tf.cast(anchors_weak, dtype=tf.float32)
        anchors_strong = tf.cast(anchors_strong, dtype=tf.float32)

        weak_anchors_prob = tf.nn.softmax(anchors_weak)

        target = tf.argmax(weak_anchors_prob, axis=1)
        target = tf.cast(target, dtype=tf.int32)

        batch_size = tf.shape(target)[0]

        prob_rows = tf.range(batch_size)
        prob_rows = tf.reshape(prob_rows, (batch_size, 1))
        prob_values = tf.reshape(target, (batch_size, 1))
        prob_indices = tf.concat([prob_rows, prob_values], axis=1)

        max_prob = tf.gather_nd(weak_anchors_prob, prob_indices)
        mask = max_prob > self.threshold

        b, c = tf.shape(weak_anchors_prob)
        target_masked = tf.boolean_mask(target, mask)

        n = tf.shape(target_masked)[0]

        # Inputs are strongly augmented anchors
        input_ = anchors_strong

        # Loss
        # if n > 0:
        #     loss = MaskedCrossEntropyLoss(input_, target, mask=mask)
        # else:
        #     loss = tf.zeros((n))

        loss = MaskedCrossEntropyLoss(input_, target, mask=mask)

        return loss


# class SimCLRLoss(nn.Module):
#     # Based on the implementation of SupContrast
#     def __init__(self, temperature):
#         super(SimCLRLoss, self).__init__()
#         self.temperature = temperature
#
#     def forward(self, features):
#         """
#         input:
#             - features: hidden feature representation of shape [b, 2, dim]
#
#         output:
#             - loss: loss computed according to SimCLR
#         """
#
#         b, n, dim = features.size()
#         assert (n == 2)
#         mask = torch.eye(b, dtype=torch.float32).cuda()
#
#         contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
#         anchor = features[:, 0]
#
#         # Dot product
#         dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature
#
#         # Log-sum trick for numerical stability
#         logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
#         logits = dot_product - logits_max.detach()
#
#         mask = mask.repeat(1, 2)
#         logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0)
#         mask = mask * logits_mask
#
#         # Log-softmax
#         exp_logits = torch.exp(logits) * logits_mask
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
#
#         # Mean log-likelihood for positive
#         loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()
#
#         return loss


def crossentropy(args):
    def _loss(y_true, y_pred):
        if args.classes == 1:
            return tf.keras.losses.binary_crossentropy(y_true, y_pred)
        else:
            return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return _loss


def supervised_contrastive(args, batch_size_per_replica):
    def _loss(labels, logits):
        labels = tf.reshape(labels, (-1, 1))
        # indicator for yi=yj
        mask = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)

        # (zi dot zj) / temperature
        anchor_dot_contrast = tf.math.divide(
            tf.linalg.matmul(logits, tf.transpose(logits)),
            tf.constant(args.temperature, dtype=tf.float32))

        # for numerical stability
        logits_max = tf.math.reduce_max(anchor_dot_contrast, axis=-1, keepdims=True)
        anchor_dot_contrast = anchor_dot_contrast - logits_max

        # tile mask for 2N images
        mask = tf.tile(mask, (2, 1))

        # indicator for i \neq j
        logits_mask = tf.ones_like(mask)-tf.eye(batch_size_per_replica*2)
        mask *= logits_mask

        # compute log_prob
        # log(\exp(z_i \cdot z_j / temperature) / (\sum^{2N}_{k=1} \exp{z_i \cdot z_k / temperature}))
        # = (z_i \cdot z_j / temperature) - log(\sum^{2N}_{k=1} \exp{z_i \cdot z_k / temperature})
        # apply indicator for i \neq k in denominator
        exp_logits = tf.math.exp(anchor_dot_contrast) * logits_mask
        log_prob = anchor_dot_contrast - tf.math.log(tf.math.reduce_sum(exp_logits, axis=-1, keepdims=True))
        mean_log_prob = tf.reduce_sum(mask * log_prob, axis=-1) / tf.reduce_sum(mask, axis=-1)
        loss = -tf.reduce_mean(tf.reshape(mean_log_prob, (2, batch_size_per_replica)), axis=0)
        return loss
    return _loss