import random
import time

import tensorflow as tf
from tensorflow.keras import optimizers, metrics, models


def supervised_gradient_training(network: models.Model,
                                 train_db: tf.data.Dataset,
                                 valid_db: tf.data.Dataset = None,
                                 test_db: tf.data.Dataset = None,
                                 train_data_len: int = 0,
                                 valid_data_len: int = 0,
                                 test_data_len: int = 0,
                                 batch_size: int = 32,
                                 epochs: int = 10,
                                 optimizer=optimizers.Adam(),
                                 model_file: str = 'supervised_model.h5'
                                 ):
    # 训练集
    train_loss = metrics.Mean(name='train_loss')
    train_accuracy = metrics.SparseCategoricalAccuracy(name='train_accuracy')
    # 验证集
    valid_loss = metrics.Mean(name='valid_loss')
    valid_accuracy = metrics.SparseCategoricalAccuracy(name='valid_accuracy')
    # 测试集
    test_loss = metrics.Mean(name='test_loss')
    test_accuracy = metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # Loss函数
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    # @tf.function
    def train_step(x_feature_ids, x_values, labels):
        with tf.GradientTape() as tape:
            x = {
                'Input-Feature': x_feature_ids,
                'Input-Value': x_values,
            }
            predictions = network(x)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, network.trainable_variables))
        train_loss(loss)
        train_accuracy(labels, predictions)

    # @tf.function
    def valid_step(x_feature_ids, x_values, labels):
        x = {
            'Input-Feature': x_feature_ids,
            'Input-Value': x_values,
        }
        predictions = network(x)
        t_loss = loss_object(labels, predictions)

        valid_loss(t_loss)
        valid_accuracy(labels, predictions)

    def test_step(x_feature_ids, x_values, labels):
        x = {
            'Input-Feature': x_feature_ids,
            'Input-Value': x_values,
        }
        predictions = network(x)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    train_steps_per_epoch = int(train_data_len // batch_size)
    valid_steps_per_epoch = int(valid_data_len // batch_size)
    test_steps_per_epoch = int(test_data_len // batch_size)

    last_valid_accuracy = 0
    for epoch in range(epochs):

        # 重置度量指标
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        start = time.perf_counter()
        for step, (x_feature_ids, x_values, labels) in enumerate(train_db):
            train_step(x_feature_ids, x_values, labels)

            if step % 100 == 0:
                template = 'Epoch {}, Step {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
                print(template.format(epoch + 1,
                                      str(step),
                                      train_loss.result(),
                                      train_accuracy.result() * 100,
                                      test_loss.result(),
                                      test_accuracy.result() * 100))

            if step >= train_steps_per_epoch:
                break

        if valid_db is not None:
            for step, (x_feature_ids, x_values, labels) in enumerate(valid_db):
                valid_step(x_feature_ids, x_values, labels)
                if step >= valid_steps_per_epoch:
                    break

        end = time.perf_counter()

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Valid Loss: {}, Valid Accuracy: {}, Times: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              valid_loss.result(),
                              valid_accuracy.result() * 100,
                              str(end - start)))

        if valid_accuracy.result() > last_valid_accuracy:
            last_valid_accuracy = valid_accuracy.result()
            print("Exporting saved model..")
            network.save(model_file)
            print("Complete export saved model.")
