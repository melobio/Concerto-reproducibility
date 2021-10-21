from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os
import re
import threading

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf


def strip_var_name(var_name):
    """Strips variable name of sub-strings blocking variable name matching.
    Removes sub-strings that should be ignored when matching checkpointed variable
    names to variable names in the training graph, namely:
    - trailing colon + number, e.g. "W:0" --> "W"
    - partitioning info., e.g. "/a/part_12/b" --> "a/b".
    (Note that checkpointed variables do not have partitioning info in their name,
    while model variables do).
    Args:
      var_name: str, variable name.
    Returns:
      stripped variable name.
    """
    # Strip trailing number, e.g. convert "lstm/W_0:0" to "lstm/W_0".
    var_name = re.sub(r':\d+$', '', var_name)
    # Strip partitioning info, e.g. convert "W_0/part_3/Adagrad" to "W_0/Adagrad".
    var_name = re.sub(r'/part_\d+', '', var_name)
    return var_name


def shard_weight(w, num_cores):
    """Apply XLA sharding to a weight `w`."""
    del num_cores
    return w


def shard_tensor(x, num_cores):
    """Apply XLA sharding to a tensor `x`."""
    del num_cores
    return x


def get_l2_loss(variables, excluded_keywords=None):
    """Traverse `tf.trainable_variables` compute L2 reg. Ignore `batch_norm`."""

    def _is_excluded(v):
        """Guess whether a variable belongs to `batch_norm`."""
        keywords = ['batchnorm', 'batch_norm', 'bn',
                    'layernorm', 'layer_norm']
        if excluded_keywords is not None:
            keywords += excluded_keywords
        return any([k in v.name.lower() for k in keywords])

    l2_losses = [tf.nn.l2_loss(v) for v in variables if not _is_excluded(v)]
    return tf.add_n(l2_losses)


def setup_ema(
        variables,
        ema_decay: float = 0.999,
        ema_start: int = 0,
        global_step: int = 1,
        num_cores_per_replica: int = 1,
        name_scope=None):
    """Create exponential moving average for all variables under `name_scope`."""
    logging.info(f'ema_decay with rate {ema_decay}')
    all_vars = variables  # tf.global_variables()
    ema_ops = []
    step = tf.cast(global_step - ema_start, tf.float32)
    decay = 1. - tf.minimum(ema_decay, (step + 1.) / (step + 10.))
    decay = tf.cond(global_step < ema_start, lambda: tf.constant(1, tf.float32), lambda: decay)

    def should_skip(v):
        key_words = ['momentum', 'rms', 'global_step', 'debug', 'adam', 'lars']
        conditions = [k in v.name.lower() for k in key_words]
        if name_scope is not None:
            conditions += [not v.name.lower().startswith(name_scope)]
        return any(conditions)

    def get_init(v_name):
        key_words = ['variance', 'beta']
        if any([k in v_name for k in key_words]):
            return tf.initializers.ones()
        return tf.initializers.zeros()

    with tf.variable_scope('ema'):
        for v in all_vars:
            if not should_skip(v):
                v_name = strip_var_name(v.name)
                with tf.device(v.device):
                    ema_var = tf.get_variable(
                        name=v_name,
                        shape=v.shape.as_list(),
                        initializer=get_init(v_name),
                        trainable=False)
                    v = shard_weight(v, num_cores_per_replica)
                    ema = shard_weight(ema_var, num_cores_per_replica)
                    ema_op = tf.assign_sub(ema_var, decay * (ema - v), use_locking=True)
                ema_ops.append(ema_op)
    ema_op = tf.group(*ema_ops)
    return ema_op


def add_weight_decay(variables, gradients, weight_decay:float=1e-4):
    """Add the gradients of `weight_decay` to existing `gradients`."""

    def should_skip_(v):
        """Guess whether a variable belongs to `batch_norm`."""
        keywords = ['batchnorm', 'batch_norm', 'bn', 'layer_norm', 'layernorm']
        return any([k in v.name.lower() for k in keywords])

    reg_gradients = []
    for v, g in zip(variables, gradients):
        with tf.device(v.device):
            # if g is not None:
            #     g = tf.tpu.cross_replica_sum(g)
            if should_skip_(v):
                reg_gradients.append(g)
            else:
                # if params.use_xla_sharding:
                #     v = shard_weight(v, params.num_cores_per_replica)
                if g is None:
                    reg_gradients.append(tf.stop_gradient(v) * weight_decay)
                else:
                    reg_gradients.append(g + tf.stop_gradient(v) * weight_decay)
    return reg_gradients
