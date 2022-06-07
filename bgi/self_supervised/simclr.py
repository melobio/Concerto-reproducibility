import random
import time
import os
import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers, losses, metrics, datasets, models

from bgi.losses.contrastive_loss import base_contrastive_loss


def self_supervised_gradient_training(network: models.Model,
                                      train_db: tf.data.Dataset,
                                      valid_db: tf.data.Dataset = None,
                                      test_db: tf.data.Dataset = None,
                                      train_data_len: int = 0,
                                      valid_data_len: int = 0,
                                      test_data_len: int = 0,
                                      batch_size: int = 32,
                                      epochs: int = 10,
                                      optimizer=optimizers.Adam(),
                                      weight_path: str = None,
                                      temperature: float = 0.1,
                                      LARGE_NUM: int = 1e9,
                                      model_file: str = 'simclr_self_supervised_model.h5'
                                      ):
    if network is None:
        return

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    # 加载权重
    if weight_path and os.path.exists(weight_path):
        network.load_weights(weight_path, by_name=True)

    train_steps_per_epoch = int(train_data_len // batch_size) - 1

    last_train_loss = 1.0 * 10e8
    for epoch in range(epochs):
        train_loss.reset_states()

        start = time.process_time()
        for step, (x_feature_ids, x_values, aug_feature_ids, aug_values, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                x1 = {
                    'Input-Feature': x_feature_ids,
                    'Input-Value': x_values,
                }
                x2 = {
                    'Input-Feature': aug_feature_ids,
                    'Input-Value': aug_values,
                }
                z1 = network(x1)
                z2 = network(x2)

                # Normalize
                z1 = tf.math.l2_normalize(z1, axis=1)
                z2 = tf.math.l2_normalize(z2, axis=1)

                z1_large = z1
                z2_large = z2

                batch_size = tf.shape(z1)[0]

                labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
                masks = tf.one_hot(tf.range(batch_size), batch_size)

                # =======================================================================
                # A Simple Framework for Contrastive Learning of Visual Representations
                # Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
                # https://arxiv.org/abs/2002.05709
                #
                # Big Self-Supervised Models are Strong Semi-Supervised Learners
                # Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, Geoffrey Hinton
                # https://arxiv.org/abs/2006.10029
                # ========================================================================
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
                loss = loss_a + loss_b
                loss = base_contrastive_loss(loss)

            grads = tape.gradient(loss, network.trainable_variables)
            optimizer.apply_gradients(zip(grads, network.trainable_variables))
            train_loss(loss)

            if step % 100 == 0:
                template = 'Epoch {}, Step {}, Loss: {}.'
                print(template.format(epoch + 1,
                                      str(step),
                                      train_loss.result()))

            if step >= train_steps_per_epoch:
                break

        end = time.process_time()

        template = 'Epoch {}, Loss: {}, Times: {}.'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              str(end - start)), )

        if train_loss.result() < last_train_loss:
            last_train_loss = train_loss.result()
            print("Exporting saved model..")
            network.save(model_file)
            print("Complete export saved model.")
