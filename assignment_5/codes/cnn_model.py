import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import os, re, sys
import numpy as np
# TOWER_NAME = 'tower'
# FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_integer('batch_size', 32,
#                             """Number of images to process in a batch.""")
# # tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
# #                            """Path to the CIFAR-10 data directory.""")
# tf.app.flags.DEFINE_boolean('use_fp16', False,
#                             """Train the model using fp16.""")

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def _variable_with_weight_decay(name, shape, wd):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var  


# def _variable_on_cpu(name, shape, initializer):
#     with tf.device('/cpu:0'):
#         dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
#         var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
#     return var


# def _activation_summary(x):
#     tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
#     tf.summary.histogram(tensor_name + '/activations', x)
#     tf.summary.scalar(tensor_name + '/sparsity',tf.nn.zero_fraction(x))


def parametric_relu(input_, name):
    alpha = tf.get_variable(name=name + '_alpha',
                            shape=input_.get_shape()[-1],
                            initializer=tf.random_uniform_initializer(minval=0.1, maxval=0.3),
                            dtype=tf.float32)
    pos = tf.nn.relu(input_)
    tf.summary.histogram(name, pos)
    neg = alpha * (input_ - abs(input_)) * 0.5
    return pos + neg


def conv(input_, name, k, n_o, wd, is_tr, s=1, is_act=True, padding='SAME'):
    n_i = input_.get_shape()[-1].value
    with tf.variable_scope(name):
        weights = _variable_with_weight_decay('weights', shape=[k, k, n_i, n_o], wd=wd)
        # weights = tf.get_variable(name + "weights", kernel, tf.float32, xavier_initializer())
        biases = tf.get_variable(name +"bias", [n_o], tf.float32, tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_, weights, (1, s, s, 1), padding=padding)
        bn = tf.layers.batch_normalization(conv, axis=-1, training=is_tr, name='bn')
        activation = parametric_relu(tf.nn.bias_add(bn, biases), name + "activation") if is_act else tf.nn.bias_add(bn, biases)
        variable_summaries(weights)
        variable_summaries(biases)
    return activation


def pool(input_, name, k, s=2):
    return tf.nn.max_pool(input_, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='VALID', name=name)


def flatten(input_):
    return tf.reshape(input_, [-1, np.prod(input_.shape.as_list()[1:])])


# def fully_conn(input_, n_o):
#     W = tf.get_variable(tf.truncated_normal([int(input_.shape[1]), n_o], stddev=.05))
#     b = tf.get_variable(tf.zeros([n_o]))
#     x = tf.add(tf.matmul(input_, W), b)  
#     return tf.nn.relu(x)


# def output(input_, n_o):
#     W = tf.Variable(tf.truncated_normal([int(input_.shape[1]), n_o], stddev=.05))
#     b = tf.Variable(tf.zeros([n_o]))
#     return tf.add(tf.matmul(input_, W), b)


def fc(input_, name, n_o, wd, is_tr, is_act=True):
    n_i = input_.get_shape()[-1].value
    with tf.variable_scope(name):
        # weights = tf.get_variable(name + "weights", [n_i, n_o], tf.float32, xavier_initializer(
        # ),  regularizer=tf.contrib.layers.l2_regularizer(reg_fac))
        weights = _variable_with_weight_decay('weights', shape=[n_i, n_o], wd=wd)
        biases = tf.get_variable(name + "bias", [n_o], tf.float32, tf.constant_initializer(0.0))
        bn = tf.nn.bias_add(tf.matmul(input_, weights), biases)
        activation = tf.layers.batch_normalization(bn, axis=-1, training=is_tr, name='bn')
        # logits = parametric_relu(activation, name + "activation") if is_act else activation
        logits = tf.nn.relu(activation)
        
        variable_summaries(weights)
        variable_summaries(biases)

    return logits


def inference(X, phase=False, dropout_rate=0.8, n_classes=10, weight_decay=1e-4):
    # logits should be of dimension (batch_size, n_classes)
    n_layers = 4
    conv_size = 5
    pool_size = 3
    for i in range(n_layers):
        X = conv(X, name="conv_" + str(i), k=conv_size, n_o=2**(5+i), wd=weight_decay, is_tr=phase)
        X = pool(X, name="pool_" + str(i), k=pool_size)
        # X = tf.cond(is_tr, lambda: tf.nn.dropout(X, rate=dropout_rate, training=True), lambda:X)
        X = tf.layers.dropout(X, rate=dropout_rate, training=True)

    X = flatten(X)
    X = fc(X, name="fc", n_o=n_classes, wd=weight_decay, is_tr=phase)
    # X = tf.layers.dropout(X, rate=dropout_rate, training=phase)
    logits = tf.layers.dense(inputs=X, units=10)
    return X

