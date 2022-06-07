import tensorflow as tf


def base_contrastive_loss(y_pred):
    return 0.5 * y_pred  # K.sum(y_pred, axis=0)


def add_contrastive_loss(hidden,
                         hidden_norm=True,
                         temperature=0.1,
                         weights=1.0,
                         LARGE_NUM=10e9):
    """Compute loss for model.
    Args:
      hidden: hidden vector (`Tensor`) of shape (2 * bsz, dim).
      hidden_norm: whether or not to use normalization on the hidden vector.
      temperature: a `floating` number for temperature scaling.
      tpu_context: context information for tpu.
      weights: a weighting number or vector.
    Returns:
      A loss scalar.
      The logits for contrastive prediction task.
      The labels for contrastive prediction task.
    """
    # Get (normalized) hidden1 and hidden2.
    if hidden_norm:
        hidden = tf.math.l2_normalize(hidden, -1)

    hidden1, hidden2 = tf.split(hidden, 2, 0)
    batch_size = tf.shape(hidden1)[0]

    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
    logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature

    loss_a = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_ab, logits_aa], 1))
    loss_b = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_ba, logits_bb], 1))
    loss = loss_a + loss_b

    return loss, logits_ab, labels




def byol_loss(p, z):
    p = tf.math.l2_normalize(p, axis=1)  # (2*bs, 128)
    z = tf.math.l2_normalize(z, axis=1)  # (2*bs, 128)

    similarities = tf.reduce_sum(tf.multiply(p, z), axis=1)
    return 2 - 2 * tf.reduce_mean(similarities)


def simclr_loss(z1, z2, temperature=0.1, LARGE_NUM=10e9):
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)

    z1_large = z1
    z2_large = z2

    step_batch_size = tf.shape(z1)[0]

    labels = tf.one_hot(tf.range(step_batch_size), step_batch_size * 2)
    masks = tf.one_hot(tf.range(step_batch_size), step_batch_size)

    logits_aa = tf.matmul(z1, z1_large, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(z2, z2_large, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(z1, z2_large, transpose_b=True) / temperature
    logits_ba = tf.matmul(z2, z1_large, transpose_b=True) / temperature

    loss_a = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_ab, logits_aa], 1))
    loss_b = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_ba, logits_bb], 1))
    loss = tf.reduce_mean(loss_a + loss_b)
    return loss

def distributed_simclr_loss(z1, z2, temperature=0.1, LARGE_NUM=10e9, GLOBAL_BATCH_SIZE=64):
    """
    需要为分布式训练适配simlr loss
    1、建议不要使用tf.reduce_mean。 这样做会将损失除以实际的每个副本中每一步都会改变的批次大小。
    2、GLOBAL_BATCH_SIZE ： scale_loss = tf.reduce_sum(loss) * (1. / GLOBAL_BATCH_SIZE)
    3、或者你可以使用tf.nn.compute_average_loss 来获取每个示例的损失，
    :param z1:
    :param z2:
    :param temperature:
    :param LARGE_NUM:
    :return:
    """
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)

    z1_large = z1
    z2_large = z2

    step_batch_size = tf.shape(z1)[0]

    labels = tf.one_hot(tf.range(step_batch_size), step_batch_size * 2)
    masks = tf.one_hot(tf.range(step_batch_size), step_batch_size)

    logits_aa = tf.matmul(z1, z1_large, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(z2, z2_large, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(z1, z2_large, transpose_b=True) / temperature
    logits_ba = tf.matmul(z2, z1_large, transpose_b=True) / temperature

    loss_a = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_ab, logits_aa], 1))
    loss_b = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_ba, logits_bb], 1))

    # loss = tf.reduce_mean(loss_a + loss_b)

    # Todo: Using tf.reduce_mean is not recommended.
    #  Doing so divides the loss by actual per replica batch size which may vary step to step.
    loss = loss_a + loss_b

    # compute_average_loss
    # per_example_loss = loss_a + loss_b
    # loss = tf.reduce_sum(per_example_loss) * (1. / GLOBAL_BATCH_SIZE)
    # print("-1: sim per_example_loss", per_example_loss)
    # loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

    return loss