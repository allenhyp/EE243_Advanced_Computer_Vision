import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import os, re, sys
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


# def batch_norm(input_, name, n_out, phase_train):
#     with tf.variable_scope(name + 'bn'):
#         beta = tf.get_variable(tf.constant(0.0, shape=[n_out]), name=name + 'beta', trainable=True)
#         gamma = tf.get_variable(tf.constant(1.0, shape=[n_out]), name=name + 'gamma', trainable=True)
#         if len(input_.get_shape().as_list()) > 3:
#             batch_mean, batch_var = tf.nn.moments(input_, [0, 1, 2], name=name + 'moments')
#         else:
#             batch_mean, batch_var = tf.nn.moments(input_, [0, 1], name=name + 'moments')
#         ema = tf.train.ExponentialMovingAverage(decay=0.5)

#         def mean_var_with_update():
#             ema_apply_op = ema.apply([batch_mean, batch_var])
#             with tf.control_dependencies([ema_apply_op]):
#                 return tf.identity(batch_mean), tf.identity(batch_var)

#         mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (
#             ema.average(batch_mean), ema.average(batch_var)))
#         normed = tf.nn.batch_normalization(input_, mean, var, beta, gamma, 1e-3)

#     variable_summaries(beta)
#     variable_summaries(gamma)
#     return normed


def parametric_relu(input_, name):
    alpha = tf.get_variable(name=name + '_alpha',
                            shape=input_.get_shape()[-1],
                            initializer=tf.random_uniform_initializer(minval=0.1, maxval=0.3),
                            dtype=tf.float32)
    pos = tf.nn.relu(input_)
    tf.summary.histogram(name, pos)
    neg = alpha * (input_ - abs(input_)) * 0.5
    return pos + neg


def conv(input_, name, k, n_o, wd, is_tr, s=1, is_act=True, is_bn=True, padding='SAME'):

    n_i = input_.get_shape()[-1].value
    with tf.variable_scope(name):
        kernel = _variable_with_weight_decay('weights', shape=[k, k, n_i, n_o], wd=wd)
        print(kernel)
        weights = tf.get_variable(name + "weights", kernel, tf.float32, xavier_initializer())
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


def fully_conn(input_, n_o):
    W = tf.get_variable(tf.truncated_normal([int(input_.shape[1]), n_o], stddev=.05))
    b = tf.get_variable(tf.zeros([n_o]))
    x = tf.add(tf.matmul(input_, W), b)  
    return tf.nn.relu(x)


def output(input_, n_o):
    W = tf.Variable(tf.truncated_normal([int(input_.shape[1]), n_o], stddev=.05))
    b = tf.Variable(tf.zeros([n_o]))
    return tf.add(tf.matmul(input_, W), b)


def inference(X, phase=False, dropout_rate=0.8, n_classes=10, weight_decay=1e-4):
    # logits should be of dimension (batch_size, n_classes)
    n_layers = 2
    conv_size = 5
    n_o = 64
    is_tr = True
    pool_size = 3
    n_i = X.get_shape()[-1].value
    for i in range(n_layers):
        X = conv(X, name="conv_" + str(i), k=conv_size, n_o=n_o, wd=weight_decay, is_tr=is_tr, is_bn=False)
        X = pool(X, name="pool_" + str(i), k=pool_size)
        # X = tf.cond(is_tr, lambda: tf.nn.dropout(X, rate=dropout_rate, training=True), lambda:X)
        X = tf.layers.dropout(X, rate=dropout_rate, training=True)

    X = flatten(X)
    X = fully_conn(X, 128)
    X = tf.layers.dropout(X, rate=dropout_rate, training=True)
    return output(X, n_classes)
    # logits = tf.layers.dense(X, units=n_classes)
    # return logits
    '''
    # 1st conv layer with max_pool, normalization, and dropout
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64], wd=weight_decay)
        conv = tf.nn.conv2d(X, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')
    dropout1 = tf.layers.dropout(inputs=norm1, rate=dropout_rate, training=True)

    # 2nd conv layer with max_pool, normalization, and dropout
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64], wd=weight_decay)
        conv = tf.nn.conv2d(X, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    dropout2 = tf.layers.dropout(inputs=norm2, rate=dropout_rate, training=True)

    # logits layer
    logits = tf.layers.dense(inputs=dropout2, units=10)

    return logits
    '''
