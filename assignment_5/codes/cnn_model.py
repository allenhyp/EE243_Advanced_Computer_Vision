import tensorflow as tf

def _variable_with_weight_decay(name, shape, wd):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var  

def inference(X,phase=False,dropout_rate=0.8,n_classes=10,weight_decay=1e-4):
    # logits should be of dimension (batch_size, n_classes)
    
    return logits
