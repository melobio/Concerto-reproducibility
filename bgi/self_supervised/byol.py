import random
import time
import os
import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers, losses, metrics, datasets, models

from bgi.losses.contrastive_loss import byol_loss


def self_supervised_gradient_training(online_network: models.Model,
                                      target_network: models.Model,
                                      train_db: tf.data.Dataset,
                                      valid_db: tf.data.Dataset = None,
                                      test_db: tf.data.Dataset = None,
                                      train_data_len: int = 0,
                                      valid_data_len: int = 0,
                                      test_data_len: int = 0,
                                      batch_size: int = 32,
                                      epochs: int = 10,
                                      optimizer=optimizers.Adam(),
                                      online_weight_path: str = None,
                                      target_weight_path: str = None,
                                      model_file: str = 'byol_self_supervised_model.h5'
                                      ):
    if online_network is None or target_network is None:
        return

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    # 加载权重
    if online_weight_path and os.path.exists(online_weight_path):
        online_network.load_weights(online_weight_path, by_name=True)

    if target_weight_path and os.path.exists(target_weight_path):
        target_network.load_weights(target_weight_path, by_name=True)

    train_steps_per_epoch = int(train_data_len // batch_size) - 1

    last_train_loss = 1.0 * 10e8
    for epoch in range(epochs):
        train_loss.reset_states()

        start = time.process_time()
        for step, (x_feature_ids, x_values, aug_feature_ids, aug_values, y) in enumerate(train_db):
            x1 = {
                'Input-Feature': x_feature_ids,
                'Input-Value': x_values,
            }
            x2 = {
                'Input-Feature': aug_feature_ids,
                'Input-Value': aug_values,
            }
            z_target_1 = target_network(x1)
            z_target_2 = target_network(x2)

            with tf.GradientTape() as tape:
                z_online_1 = online_network(x1)
                z_online_2 = online_network(x2)

                z_online = tf.concat([z_online_1, z_online_2], axis=0)
                z_target = tf.concat([z_target_2, z_target_1], axis=0)
                loss = byol_loss(z_online, z_target)

            grads = tape.gradient(loss, online_network.trainable_variables)
            optimizer.apply_gradients(zip(grads, online_network.trainable_variables))
            train_loss(loss)

            if step % 100 == 0:
                template = 'Epoch {}, Step {}, Loss: {}.'
                print(template.format(epoch + 1,
                                      str(step),
                                      train_loss.result()))

            if step >= train_steps_per_epoch:
                break

        # Update target networks (exponential moving average of online networks)
        beta = 0.99
        f_target_weights = target_network.get_weights()
        f_online_weights = online_network.get_weights()
        for i in range(len(f_online_weights)):
            f_target_weights[i] = beta * f_target_weights[i] + (1 - beta) * f_online_weights[i]
        target_network.set_weights(f_target_weights)

        end = time.process_time()

        template = 'Epoch {}, Loss: {}, Times: {}.'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              str(end - start)), )

        if train_loss.result() < last_train_loss:
            last_train_loss = train_loss.result()
            print("Exporting saved model..")
            online_network.save(model_file)
            print("Complete export saved model.")
