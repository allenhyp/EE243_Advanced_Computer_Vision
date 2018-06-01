import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import os, re, sys
import numpy as np

def variable_summaries(var):
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


# def parametric_relu(input_, name):
#     alpha = tf.get_variable(name=name + '_alpha',
#                             shape=input_.get_shape()[-1],
#                             initializer=tf.random_uniform_initializer(minval=0.1, maxval=0.3),
#                             dtype=tf.float32)
#     pos = tf.nn.relu(input_)
#     tf.summary.histogram(name, pos)
#     neg = alpha * (input_ - abs(input_)) * 0.5
#     return pos + neg


def conv(input_, name, k, n_o, wd, is_tr, s=1, is_act=True, padding='SAME'):
    n_i = input_.get_shape()[-1].value
    with tf.variable_scope(name):
        weights = _variable_with_weight_decay(name+'_weights', shape=[k, k, n_i, n_o], wd=wd)
        # weights = tf.get_variable(name + "weights", kernel, tf.float32, xavier_initializer())
        biases = tf.get_variable(name + "_bias", [n_o], tf.float32, tf.constant_initializer(0.0))
        # biases = _variable_with_weight_decay(name + "_bias", shape=[n_o], wd=wd)
        conv = tf.nn.conv2d(input_, weights, (1, s, s, 1), padding=padding)
        bn = tf.layers.batch_normalization(conv, axis=-1, training=is_tr, name='bn')
        activation = tf.nn.relu(tf.nn.bias_add(bn, biases))

        # activation = parametric_relu(tf.nn.bias_add(bn, biases), name + "activation") if is_act else tf.nn.bias_add(bn, biases)
        variable_summaries(weights)
        variable_summaries(biases)
    return activation


def pool(input_, name, k, s=2):
    return tf.nn.max_pool(input_, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME', name=name)


def flatten(input_):
    return tf.reshape(input_, [-1, np.prod(input_.shape.as_list()[1:])])


# def fully_conn(input_, n_o):
#     W = tf.get_variable(tf.truncated_normal([int(input_.shape[1]), n_o], stddev=.05))
#     b = tf.get_variable(tf.zeros([n_o]))
#     x = tf.add(tf.matmul(input_, W), b)  
#     return tf.nn.relu(x)


def output(input_, n_o, wd):
    n_i = input_.get_shape()[-1].value
    weights = _variable_with_weight_decay("output_weights", [n_i, n_o], wd=wd)
    biases = _variable_with_weight_decay("output_biases", [n_o], wd=wd)
    
    return tf.matmul(input_, weights), biases


def fc(input_, name, n_o, wd, is_tr, is_act=True):
    n_i = input_.get_shape()[-1].value
    with tf.variable_scope(name):
        # weights = tf.get_variable(name + "weights", [n_i, n_o], tf.float32, xavier_initializer(
        # ),  regularizer=tf.contrib.layers.l2_regularizer(reg_fac))
        weights = _variable_with_weight_decay(name+'_weights', shape=[n_i, n_o], wd=wd)
        biases = tf.get_variable(name + "_bias", [n_o], tf.float32, tf.constant_initializer(0.0))
        # biases = _variable_with_weight_decay(name + "_bias", shape=[n_o], wd=wd)
        conv = tf.nn.bias_add(tf.matmul(input_, weights), biases)
        bn = tf.layers.batch_normalization(conv, axis=-1, training=is_tr, name='bn')
        # logits = parametric_relu(activation, name + "activation") if is_act else activation
        activation = tf.nn.relu(tf.nn.bias_add(bn, biases))
        
        variable_summaries(weights)
        variable_summaries(biases)

    return activation


def inference(X, phase=False, dropout_rate=0.8, n_classes=10, weight_decay=1e-4):
    # logits should be of dimension (batch_size, n_classes)
    # X = conv(X, name="conv_1", k=3, n_o=32, wd=weight_decay, is_tr=phase)
    X = pool(X, name="pool_1", k=2)
    X = tf.layers.dropout(X, rate=dropout_rate, training=phase)

    X = conv(X, name="conv_2", k=3, n_o=64, wd=weight_decay, is_tr=phase)
    X = pool(X, name="pool_2", k=2)
    X = tf.layers.dropout(X, rate=dropout_rate, training=phase)

    X = conv(X, name="conv_3", k=3, n_o=64, wd=weight_decay, is_tr=phase)
    X = pool(X, name="pool_3", k=2)
    X = tf.layers.dropout(X, rate=dropout_rate, training=phase)

    X = conv(X, name="conv_4", k=2, n_o=128, wd=weight_decay, is_tr=phase)
    X = pool(X, name="pool_4", k=2)
    X = tf.layers.dropout(X, rate=dropout_rate, training=phase)

    X = flatten(X)
    X = fc(X, name="fc", n_o=1024, wd=weight_decay, is_tr=phase)
    X = tf.nn.dropout(X, dropout_rate)

    # X = tf.layers.dropout(X, rate=dropout_rate, training=phase)
    logits = output(X, n_classes, weight_decay)
    return X

