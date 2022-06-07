import numpy as np
import pandas as pd
import os

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, History

from bgi.metrics.clustering_metrics import ACC, ARI, NMI


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


def compile(model: tf.keras.models.Model, optimizer='sgd', loss='kld'):
    model.compile(optimizer=optimizer, loss=loss)


def fit_on_batch(model: tf.keras.models.Model,
                 x_data: np.ndarray,
                 y_label: np.ndarray,
                 init_predictions: np.ndarray,
                 maxiter=1e4,
                 update_interval=200,
                 save_encoder_step=4,
                 save_dir='./',
                 tol=0.005,
                 batch_size=32,
                 epochs=50,
                 use_earlyStop=True,
                 num_classes=10,
                 ):
    # step1 initial weights by louvain, or Kmeans, or N2D, 并训练一轮
    if use_earlyStop:
        callbacks = [EarlyStopping(monitor='loss', min_delta=1e-4, patience=5, verbose=1, mode='auto')]
        y_pred = tf.keras.utils.to_categorical(init_predictions, num_classes=num_classes)
        loss = model.fit(x=x_data,
                         y=y_pred,
                         batch_size=batch_size,
                         epochs=epochs,
                         callbacks=callbacks,
                         verbose=1)
    else:
        y_pred = tf.keras.utils.to_categorical(init_predictions, num_classes=num_classes)
        loss = model.fit(x=x_data,
                         y=y_pred,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=1)

    # Step 2: deep clustering
    y_pred_last = np.copy(init_predictions)
    index_array = np.arange(x_data.shape[0])
    index = 0
    last_acc = 0
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            q = model.predict(x_data, verbose=0)
            p = target_distribution(q)
            y_pred = q.argmax(1)

            if y_label is not None:
                acc = np.round(ACC(y_label, y_pred), 5)
                nmi = np.round(NMI(y_label, y_pred), 5)
                ari = np.round(ARI(y_label, y_pred), 5)
                print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), '.')

                # if acc > last_acc:
                #     last_acc = acc
                # elif last_acc > 0 and acc < last_acc:
                #     break

            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            print("The value of delta_label of current", str(ite + 1), "th iteration is", delta_label, ">= tol", tol)
            if ite > 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print('Reached tolerance threshold. Stop training.')
                break
        idx = index_array[index * batch_size: min((index + 1) * batch_size, x_data.shape[0])]
        loss = model.train_on_batch(x=x_data[idx], y=p[idx])
        index = index + 1 if (index + 1) * batch_size <= x_data.shape[0] else 0

    # save cluster model
    model.save(os.path.join(save_dir, "self_supervised_self_cluster_model.h5"))
    print("Save model.")


def fit_on_all(model: tf.keras.models.Model,
               x_data: np.ndarray,
               y_label: np.ndarray,
               init_predictions: np.ndarray,
               maxiter=1e4,
               update_interval=200,
               save_encoder_step=4,
               save_dir='./',
               tol=0.005,
               batch_size=32,
               epochs=50,
               use_earlyStop=True,
               ):  # unsupervised

    if use_earlyStop:
        callbacks = [EarlyStopping(monitor='loss', min_delta=1e-4, patience=5, verbose=1, mode='auto')]
        y_pred = tf.keras.utils.to_categorical(init_predictions, num_classes=10)
        loss = model.fit(x=x_data,
                         y=y_pred,
                         batch_size=batch_size,
                         epochs=epochs,
                         callbacks=callbacks,
                         verbose=1)
    else:
        y_pred = tf.keras.utils.to_categorical(init_predictions, num_classes=10)
        loss = model.fit(x=x_data,
                         y=y_pred,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=1)

    # Step 2: deep clustering
    last_acc = 0
    y_pred_last = np.copy(init_predictions)
    for ite in range(int(maxiter)):
        # if self.save_encoder_weights and ite % save_encoder_step == 0:  # save ae_weights for every 5 iterations
        #     self.encoder.save_weights(os.path.join(self.save_dir,
        #                                            'encoder_weights_resolution_' + str(self.resolution) + "_" + str(
        #                                                ite) + '.h5'))
        #     print('Fine tuning encoder weights are saved to %s/encoder_weights.h5' % self.save_dir)
        q = model.predict(x_data, verbose=0)

        # update the auxiliary target distribution p
        p = target_distribution(q)

        # evaluate the clustering performance
        y_pred = q.argmax(1)

        if y_label is not None:
            acc = np.round(ACC(y_label, y_pred), 5)
            nmi = np.round(NMI(y_label, y_pred), 5)
            ari = np.round(ARI(y_label, y_pred), 5)
            # loss = np.round(loss, 5)
            print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), '.')

            if acc > last_acc:
                last_acc = acc
            elif last_acc > 0 and acc < last_acc:
                break

        # check stop criterion
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stop training.')
            break
        print("The value of delta_label of current", str(ite + 1), "th iteration is", delta_label, ">= tol", tol)
        # train on whole dataset on prespecified batch_size
        if use_earlyStop:
            callbacks = [EarlyStopping(monitor='loss', min_delta=1e-4, patience=5, verbose=1, mode='auto')]
            model.fit(x=x_data, y=p, epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                      shuffle=True, verbose=True)
        else:
            model.fit(x=x_data, y=p, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=True)

    # save cluster model
    model.save(os.path.join(save_dir, "self_supervised_self_cluster_model.h5"))
    print("Save model.")
