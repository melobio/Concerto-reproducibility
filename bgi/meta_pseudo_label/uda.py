import tensorflow as tf
import collections

from bgi.meta_pseudo_label import common_utils

MODEL_SCOPE = 'model'


def build_uda_cross_entropy_1(model: tf.keras.models.Model,
                            all_data_source,
                            all_data_target,
                            all_data_target_aug,
                            l_labels,
                            train_batch_size: int,
                            num_train_steps: int,
                            num_classes: int,
                            num_replicas: int = 1,
                            label_smoothing: float = 0.1,
                            uda_data: int = 3,
                            uda_temp: float = 0.5,
                            uda_threshold: float = 0.6,
                            global_step=1):
    """Compute the UDA loss."""
    batch_size = train_batch_size // num_replicas
    labels = {}

    # l_labels is sparse. turn into one_hot
    if l_labels.dtype == tf.int32:
        labels['l'] = tf.one_hot(l_labels, num_classes, dtype=tf.float32)
    else:
        labels['l'] = l_labels

    masks = {}
    logits = {}
    cross_entropy = {}

    # 进行预测

    logits['l'] = model(all_data_source, training=True)
    logits['u_ori'] = model(all_data_target, training=True)
    logits['u_aug'] = model(all_data_target_aug, training=True)

    # sup loss
    cross_entropy['l'] = tf.nn.softmax_cross_entropy_with_logits(labels=labels['l'], logits=logits['l'])
    probs = tf.nn.softmax(logits['l'], axis=-1)

    correct_probs = tf.reduce_sum(labels['l'] * probs, axis=-1)

    # Label mask
    r = tf.cast(global_step, tf.float32) / float(num_train_steps)
    l_threshold = r * (1. - 1. / num_classes) + 1. / num_classes
    masks['l'] = tf.less_equal(correct_probs, l_threshold)
    masks['l'] = tf.cast(masks['l'], tf.float32)
    masks['l'] = tf.stop_gradient(masks['l'])

    cross_entropy['l'] = tf.reduce_sum(cross_entropy['l']) / float(train_batch_size)

    # unsup loss
    labels['u_ori'] = tf.nn.softmax(logits['u_ori'] / uda_temp, axis=-1)
    labels['u_ori'] = tf.stop_gradient(labels['u_ori'])

    cross_entropy['u'] = (labels['u_ori'] * tf.nn.log_softmax(logits['u_aug'], axis=-1))
    largest_probs = tf.reduce_max(labels['u_ori'], axis=-1, keepdims=True)
    masks['u'] = tf.greater_equal(largest_probs, uda_threshold)
    masks['u'] = tf.cast(masks['u'], tf.float32)
    masks['u'] = tf.stop_gradient(masks['u'])
    cross_entropy['u'] = tf.reduce_sum(-cross_entropy['u'] * masks['u']) / float(train_batch_size * uda_data)

    return logits, labels, masks, cross_entropy


def build_uda_cross_entropy_2(model: tf.keras.models.Model,
                            all_data_source,
                            all_data_target,
                            all_data_target_aug,
                            l_labels,
                            train_batch_size: int,
                            num_train_steps: int,
                            num_classes: int,
                            num_replicas: int = 1,
                            label_smoothing: float = 0.1,
                            uda_data: int = 3,
                            uda_temp: float = 0.5,
                            uda_threshold: float = 0.6,
                            global_step=1):
    """Compute the UDA loss."""
    batch_size = train_batch_size // num_replicas
    labels = {}

    # l_labels is sparse. turn into one_hot
    # if l_labels.dtype == tf.int32:
    #     labels['l'] = tf.one_hot(l_labels, num_classes, dtype=tf.float32)
    # else:
    #     labels['l'] = l_labels
    labels['l'] = tf.cast(l_labels,tf.float32)
    masks = {}
    logits = {}
    cross_entropy = {}

    # 进行预测

    logits['l'] = model(all_data_source, training=True)
    logits['u_ori'] = model(all_data_target, training=True)
    logits['u_aug'] = model(all_data_target_aug, training=True)

    # sup loss

    cross_entropy['l'] = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels['l'], logits=logits['l'])
    probs = tf.nn.sigmoid(logits['l'])

    correct_probs = tf.reduce_sum(labels['l'] * probs, axis=-1)

    # Label mask
    r = tf.cast(global_step, tf.float32) / float(num_train_steps)
    l_threshold = r * (1. - 1. / num_classes) + 1. / num_classes
    masks['l'] = tf.less_equal(correct_probs, l_threshold)
    masks['l'] = tf.cast(masks['l'], tf.float32)
    masks['l'] = tf.stop_gradient(masks['l'])

    cross_entropy['l'] = tf.reduce_sum(cross_entropy['l']) / float(train_batch_size)

    # unsup loss
    #labels['u_ori'] = tf.nn.softmax(logits['u_ori'] / uda_temp, axis=-1)
    labels['u_ori'] = tf.nn.sigmoid(logits['u_ori'] / uda_temp)
    labels['u_ori'] = tf.stop_gradient(labels['u_ori'])

    cross_entropy['u'] = (labels['u_ori'] * tf.nn.sigmoid(logits['u_aug']))
    largest_probs = tf.reduce_max(labels['u_ori'], axis=-1, keepdims=True)
    masks['u'] = tf.greater_equal(largest_probs, uda_threshold)
    masks['u'] = tf.cast(masks['u'], tf.float32)
    masks['u'] = tf.stop_gradient(masks['u'])
    cross_entropy['u'] = tf.reduce_sum(-cross_entropy['u'] * masks['u']) / float(train_batch_size * uda_data)

    return logits, labels, masks, cross_entropy








def build_uda_cross_entropy(model: tf.keras.models.Model,
                            all_data,
                            l_labels,
                            train_batch_size: int,
                            num_train_steps: int,
                            num_classes: int,
                            num_replicas: int = 1,
                            label_smoothing: float = 0.1,
                            uda_data: int = 3,
                            uda_temp: float = 0.5,
                            uda_threshold: float = 0.6,
                            global_step=1):
    """Compute the UDA loss."""
    batch_size = train_batch_size // num_replicas
    labels = {}

    # l_labels is sparse. turn into one_hot
    if l_labels.dtype == tf.int32:
        labels['l'] = tf.one_hot(l_labels, num_classes, dtype=tf.float32)
    else:
        labels['l'] = l_labels

    masks = {}
    logits = {}
    cross_entropy = {}

    # 进行预测
    all_logits = model(all_data, training=True)

    logits['l'], logits['u_ori'], logits['u_aug'] = tf.split(
        all_logits, [batch_size, batch_size * uda_data, batch_size * uda_data], 0)

    # sup loss
    cross_entropy['l'] = tf.nn.softmax_cross_entropy_with_logits(labels=labels['l'], logits=logits['l'])
    probs = tf.nn.softmax(logits['l'], axis=-1)

    correct_probs = tf.reduce_sum(labels['l'] * probs, axis=-1)

    # Label mask
    r = tf.cast(global_step, tf.float32) / float(num_train_steps)
    l_threshold = r * (1. - 1. / num_classes) + 1. / num_classes
    masks['l'] = tf.less_equal(correct_probs, l_threshold)
    masks['l'] = tf.cast(masks['l'], tf.float32)
    masks['l'] = tf.stop_gradient(masks['l'])

    cross_entropy['l'] = tf.reduce_sum(cross_entropy['l']) / float(train_batch_size)

    # unsup loss
    labels['u_ori'] = tf.nn.softmax(logits['u_ori'] / uda_temp, axis=-1)
    labels['u_ori'] = tf.stop_gradient(labels['u_ori'])

    cross_entropy['u'] = (labels['u_ori'] * tf.nn.log_softmax(logits['u_aug'], axis=-1))
    largest_probs = tf.reduce_max(labels['u_ori'], axis=-1, keepdims=True)
    masks['u'] = tf.greater_equal(largest_probs, uda_threshold)
    masks['u'] = tf.cast(masks['u'], tf.float32)
    masks['u'] = tf.stop_gradient(masks['u'])
    cross_entropy['u'] = tf.reduce_sum(-cross_entropy['u'] * masks['u']) / float(train_batch_size * uda_data)

    return logits, labels, masks, cross_entropy


def step_fn(model: tf.keras.models.Model,
            all_data,
            l_labels,
            optimizer: tf.keras.optimizers.Optimizer,
            learning_rate: float,
            train_batch_size: int,
            num_train_steps: int,
            num_classes: int,
            num_replicas: int = 1,
            label_smoothing: float = 0.1,
            uda_data: int = 3,
            uda_temp: float = 0.5,
            uda_threshold: float = 0.6,
            uda_weight: float = 1.0,
            weight_decay: float = 1e-4,
            uda_steps=1000,
            grad_bound: float = 5.0,
            global_step: int = 1,
            ema_decay: float = 0.999,
            ema_start: int = 0,
            num_cores_per_replica: int = 1,
            ):
    """Separate implementation."""

    # with tf.variable_scope(MODEL_SCOPE, reuse=tf.AUTO_REUSE):
    with tf.GradientTape(persistent=True) as tape:
        logits, _, masks, cross_entropy = build_uda_cross_entropy(model,
                                                                  all_data,
                                                                  l_labels,
                                                                  train_batch_size=train_batch_size,
                                                                  num_train_steps=num_train_steps,
                                                                  num_classes=num_classes,
                                                                  num_replicas=num_replicas,
                                                                  label_smoothing=label_smoothing,
                                                                  uda_data=uda_data,
                                                                  uda_temp=uda_temp,
                                                                  uda_threshold=uda_threshold,
                                                                  global_step=global_step
                                                                  )

        l2_reg_rate = tf.cast(weight_decay / num_replicas, tf.float32)
        weight_dec = common_utils.get_l2_loss(model.trainable_variables)
        uda_weight = uda_weight * tf.minimum(1., tf.cast(global_step, tf.float32) / float(uda_steps))

        total_loss = cross_entropy['l'] #+ cross_entropy['u'] * uda_weight + weight_dec * l2_reg_rate

    gradients = tape.gradient(total_loss, model.trainable_variables)
    gradients, grad_norm = tf.clip_by_global_norm(gradients, grad_bound)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # EMA exponential moving average，指数加权平均
    common_utils.setup_ema(model.trainable_variables,
                           ema_decay=ema_decay,
                           ema_start=ema_start,
                           global_step=global_step,
                           num_cores_per_replica=num_cores_per_replica,
                           name_scope=f'{MODEL_SCOPE}/{model.name}')

    logs = collections.OrderedDict()
    logs['global_step'] = tf.cast(global_step, tf.float32)
    logs['loss/total'] = total_loss
    logs['loss/cross_entropy'] = cross_entropy['l']
    logs['loss/lr'] = tf.identity(learning_rate) / num_replicas
    logs['loss/grad_norm'] = tf.identity(grad_norm) / num_replicas
    logs['loss/weight_dec'] = weight_dec / num_replicas

    logs['uda/cross_entropy'] = cross_entropy['u']
    logs['uda/u_ratio'] = tf.reduce_mean(masks['u']) / num_replicas
    logs['uda/l_ratio'] = tf.reduce_mean(masks['l']) / num_replicas
    logs['uda/weight'] = uda_weight / num_replicas

    return logs


if __name__ == '__main__':
    import numpy as np

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=10, activation='relu'),
    ])

    batch_size = 32
    l_data = np.random.random((batch_size, 100))
    l_labels = np.random.random((batch_size)) * 10
    l_labels = np.array(l_labels, dtype=np.int)
    u_data_ori = np.random.random((batch_size * 5, 100))
    u_data_aug = np.random.random((batch_size * 5, 100))

    l_data = tf.convert_to_tensor(l_data)
    u_data_ori = tf.convert_to_tensor(u_data_ori)
    u_data_aug = tf.convert_to_tensor(u_data_aug)
    l_labels = tf.convert_to_tensor(l_labels)

    opt = tf.keras.optimizers.Adam()
    logs = step_fn(model,
                   l_data,
                   l_labels,
                   u_data_ori,
                   u_data_aug,
                   optimizer=opt,
                   learning_rate=1e-4,
                   train_batch_size=32,
                   num_train_steps=1000,
                   num_classes=10,
                   num_replicas=1,
                   uda_data=5
                   )

    print("logs: ", logs)
    for k, v in logs.items():
        print(k, ': ', np.array(v))
