import tensorflow as tf
import collections

from bgi.meta_pseudo_label import uda
from bgi.meta_pseudo_label import common_utils

MODEL_SCOPE = 'model'


def step_fn(teacher_model: tf.keras.models.Model,
            student_model: tf.keras.models.Model,
            all_data,
            l_data,
            u_aug_and_l_data,
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
            # params
            ):
    """Separate implementation."""
    train_batch_size = train_batch_size
    num_replicas = num_replicas
    uda_data = uda_data
    batch_size = train_batch_size // num_replicas

    # student_model.set_weights(teacher_model.get_weights())

    # all calls to teacher
    with tf.GradientTape(persistent=True) as teacher_tape:
        logits, labels, masks, cross_entropy = uda.build_uda_cross_entropy(teacher_model,
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
                                                                           global_step=global_step)

        # ===============================================================
        # 1st call to student
        # ===============================================================
        with tf.GradientTape(persistent=True) as student_tape:
            logits['s_on_u_aug_and_l'] = student_model(u_aug_and_l_data, training=True)
            logits['s_on_u'], logits['s_on_l_old'] = tf.split(
                logits['s_on_u_aug_and_l'],
                [batch_size * uda_data, batch_size], axis=0)

            # for backprop
            # Student模型预测与Teacher模型预测比较
            cross_entropy['s_on_u'] = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.stop_gradient(tf.nn.softmax(logits['u_aug'], -1)), logits=logits['s_on_u'])
            cross_entropy['s_on_u'] = tf.reduce_sum(cross_entropy['s_on_u']) / float(train_batch_size * uda_data)

            # for Taylor
            cross_entropy['s_on_l_old'] = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.stop_gradient(labels['l']), logits=logits['s_on_l_old'])
            cross_entropy['s_on_l_old'] = tf.reduce_sum(cross_entropy['s_on_l_old']) / float(train_batch_size)

            student_loss = cross_entropy['s_on_u'] #+ cross_entropy['s_on_l_old']

        # 创建浅更新变量
        shadow = tf.compat.v1.get_variable(
            name='cross_entropy_old', shape=[], trainable=False, dtype=tf.float32)
        shadow_update = tf.compat.v1.assign(shadow, cross_entropy['s_on_l_old'])



        # 更新模型权重
        w_s = {}
        g_s = {}
        g_n = {}
        w_s['s'] = student_model.trainable_variables
        g_s['s_on_u'] = student_tape.gradient(student_loss, w_s['s'])
        # g_s['s_on_u'] = common_utils.add_weight_decay(w_s['s'], g_s['s_on_u'], weight_decay=weight_decay)
        g_s['s_on_u'], g_n['s_on_u'] = tf.clip_by_global_norm(g_s['s_on_u'], grad_bound)
        optimizer.apply_gradients(zip(g_s['s_on_u'], w_s['s']))

        with tf.control_dependencies([shadow_update]):
            # EMA exponential moving average，指数加权平均
            common_utils.setup_ema(student_model.trainable_variables,
                                   ema_decay=ema_decay,
                                   ema_start=ema_start,
                                   global_step=global_step,
                                   num_cores_per_replica=num_cores_per_replica,
                                   name_scope=f'{MODEL_SCOPE}/{student_model.name}')
        del student_tape

        # ===============================================================
        # 2nd call to student
        # ===============================================================
        with tf.GradientTape(persistent=True) as student_tape:
            logits['s_on_l_new'] = student_model(l_data, training=True)

            cross_entropy['s_on_l_new'] = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels['l'], logits=logits['s_on_l_new'])
            cross_entropy['s_on_l_new'] = tf.reduce_sum(cross_entropy['s_on_l_new']) / float(train_batch_size)

            dot_product = cross_entropy['s_on_l_new'] - shadow

            moving_dot_product = tf.compat.v1.get_variable(
                'moving_dot_product', shape=[], trainable=False, dtype=tf.float32)
            moving_dot_product_update = tf.compat.v1.assign_sub(
                moving_dot_product, 0.01 * (moving_dot_product - dot_product))

            with tf.control_dependencies([moving_dot_product_update]):
                dot_product = dot_product - moving_dot_product
                dot_product = tf.stop_gradient(dot_product)

            cross_entropy['mpl'] = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.stop_gradient(tf.nn.softmax(logits['u_aug'], axis=-1)), logits=logits['u_aug'])
            cross_entropy['mpl'] = tf.reduce_sum(cross_entropy['mpl']) / float(train_batch_size * uda_data)

        # ===============================================================
        # teacher train op
        # ===============================================================
        l2_reg_rate = tf.cast(weight_decay / num_replicas, tf.float32)
        weight_dec = common_utils.get_l2_loss(teacher_model.trainable_variables)

        uda_weight = uda_weight * tf.minimum(1., tf.cast(global_step, tf.float32) / float(uda_steps))
        teacher_loss = (cross_entropy['u'] * uda_weight
                        + cross_entropy['l']
                        + cross_entropy['mpl'] * dot_product
                        + weight_dec * l2_reg_rate
                        )

    w_s['t'] = teacher_model.trainable_variables
    g_s['t'] = teacher_tape.gradient(teacher_loss, w_s['t'])
    # g_s['t'] = common_utils.add_weight_decay(w_s['t'], g_s['t'], weight_decay=weight_decay)
    g_s['t'], g_n['t'] = tf.clip_by_global_norm(g_s['t'], grad_bound)

    optimizer.apply_gradients(zip(g_s['t'], w_s['t']))

    # EMA exponential moving average，指数加权平均
    common_utils.setup_ema(teacher_model.trainable_variables,
                           ema_decay=ema_decay,
                           ema_start=ema_start,
                           global_step=global_step,
                           num_cores_per_replica=num_cores_per_replica,
                           name_scope=f'{MODEL_SCOPE}/{teacher_model.name}')


    logs = collections.OrderedDict()
    logs['global_step'] = tf.cast(global_step, tf.float32)
    logs['loss/total'] = teacher_loss
    logs['cross_entropy/student_on_u'] = cross_entropy['s_on_u']
    logs['cross_entropy/student_on_l'] = (cross_entropy['s_on_l_new'] / num_replicas)
    logs['cross_entropy/teacher_on_u'] = cross_entropy['u']
    logs['cross_entropy/teacher_on_l'] = cross_entropy['l']
    # logs['lr/student'] = lr['s'] / num_replicas
    # logs['lr/teacher'] = lr['t'] / num_replicas
    logs['mpl/loss'] = cross_entropy['mpl']
    logs['mpl/dot_product'] = dot_product / num_replicas
    # logs['mpl/moving_dot_product'] = moving_dot_product / num_replicas
    logs['uda/u_ratio'] = tf.reduce_mean(masks['u']) / num_replicas
    logs['uda/l_ratio'] = tf.reduce_mean(masks['l']) / num_replicas
    logs['uda/weight'] = uda_weight / num_replicas

    return logs
