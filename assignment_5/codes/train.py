import os
import numpy as np
import tensorflow as tf
import time
import random
import cnn_model
from random import shuffle
import cv2
import scipy.io as sio
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

batch_size = 128                     # YOU MAY MODIFY THIS
max_epoch = 15                      # YOU MAY MODIFY THIS
init_lr = 5e-3                      # YOU MAY MODIFY THIS
summary_ckpt = 50                   # YOU MAY MODIFY THIS
model_ckpt = 500                    # YOU MAY MODIFY THIS
model_save_path = './model'         
tensorboard_path = './Tensorboard'
n_class = 10
image_height = 32
image_width = 32
num_channels = 3
use_pretrained_model = False
gpu_number = 1

if not os.path.exists(model_save_path):
   os.mkdir(model_save_path)

def get_loss(logits, labels):
   # FILL IN; cross entropy loss between logits and labels
   ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
   ce_loss = tf.reduce_mean(ce, name='cross_entropy')
   tf.add_to_collection('losses', ce_loss)
   losses = tf.get_collection('losses')
   total_loss = tf.add_n(losses, name='total_loss')
   return total_loss

def read_data(L, readpos=None):
   image = []
   label = []
   if readpos is None:
      readpos = random.sample(range(len(L)), batch_size)
   for i in range(len(readpos)):
      # FILL IN. Read images and label. image should be of dimension (batch_size,32,32,3) and label of dimension (batch_size,)
      f, l = L[i].split(' ')
      image.append(cv2.resize(cv2.imread(f), dsize=(image_height,image_width), interpolation=cv2.INTER_CUBIC))
      label.append(int(l))
   return np.array(image).astype('float32')/128 - 1, np.array(label).astype('int64')

def main():

   # Placeholders
   learning_rate = tf.placeholder(tf.float32)
   keep_prob = tf.placeholder(tf.float32)
   images = tf.placeholder(tf.float32, [None, image_height, image_width, num_channels])
   labels = tf.placeholder(tf.int64, [None])
   phase = tf.placeholder(tf.bool, [])

   with tf.device('/gpu:%d' %gpu_number):
      logits = cnn_model.inference(images, phase=phase, dropout_rate=keep_prob)
      var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      loss = get_loss(logits, labels)
      # FILL IN. Obtain accuracy of given batch of data.
      with tf.name_scope('prediction'):
         correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      tf.summary.scalar('train_accuracy', accuracy)

   apply_gradient_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss) # YOU MAY MODIFY THE OPTIMIZER

   # Summary list
   tf.summary.scalar('Total Loss',loss)
   tf.summary.image('Input', images, max_outputs=batch_size)
   for var in var_list:
      tf.summary.histogram(var.op.name, var)

   # Initialize
   init = tf.global_variables_initializer()
   config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
   config.gpu_options.allow_growth = True
   sess = tf.Session(config=config)
   sess.run(init)
   merged = tf.summary.merge_all()
   train_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
   saver = tf.train.Saver()

   # Start from scratch or load model
   if use_pretrained_model==False:
      lr = init_lr
      epoch_num = 0
   else: 
      lr = np.load('learning_rate.npy')
      epoch_num = np.load('epoch_num.npy')
      saver.restore(sess,model_save_path + '/model')

   trlist = list(open('train.list','r'))
   testlist = list(open('test.list','r'))

   test_accuracy = []
   train_accuracy = []
   train_loss = []
   # Start training
   print "Start training..."
   for i in range(epoch_num, max_epoch):

      # Update learning rate if required

      shuffle(trlist)
      _acc = 0
      epoch_start_time = time.time()
      for pos in range(0,len(trlist),batch_size):

         # Load batch data
         t = time.time()
         batch_images, batch_labels = read_data(trlist, range(pos,min(pos+batch_size,len(trlist))))
         dt = time.time()-t

         # Train with batch
         t = time.time()
         _, cost, acc = sess.run([apply_gradient_op, loss, accuracy], feed_dict={images:batch_images, labels: batch_labels, learning_rate: lr, phase: True, keep_prob: 0.8})
         # print('Epoch: %d, Item : %d, Loss: %.5f, Train Accuracy: %.2f, Data Time: %.2f, Network Time: %.2f' %(i, pos, cost, acc, dt, time.time()-t))
         train_loss.append(cost)
         train_accuracy.append(acc)
         _acc += acc * len(batch_labels)

      # lr = 0.95 * lr
      # Test, Save model
      # FILL IN. Obtain test_accuracy on the entire test set and append it to variable test_accuracy. 
      num_test_examples = len(testlist)
      total_accuracy = 0
      for offset in range(0, num_test_examples, batch_size):
         test_images, test_labels = read_data(testlist, range(offset, min(offset+batch_size, len(testlist))))
         _, _, acc = sess.run([apply_gradient_op, loss, accuracy], feed_dict={images: test_images, labels: test_labels, learning_rate: lr, phase: False, keep_prob: 0.8})
         total_accuracy += (acc * len(test_labels))
      test_acc = total_accuracy / num_test_examples
      test_accuracy.append(test_acc)
      tf.summary.scalar('test_accuracy', test_accuracy)
      print('Epoch: %d, Train Accuracy: %.2f, Test Accuracy: %.2f, Epoch Time: %.2f' %(i, _acc/len(trlist), test_acc, time.time()-epoch_start_time))

      np.save('test_accuracy.npy',test_accuracy); sio.savemat('test_accuracy.mat', mdict={'test_accuracy': test_accuracy})
      np.save('train_accuracy.npy',train_accuracy); sio.savemat('train_accuracy.mat', mdict={'train_accuracy': train_accuracy})
      np.save('train_loss.npy',train_loss); sio.savemat('train_loss.mat', mdict={'train_loss': train_loss})
      np.save('learning_rate.npy', lr)
      np.save('epoch_num.npy', i)
      saver.save(sess,model_save_path + '/model')
        
   print('Training done.')

if __name__ == "__main__":
   main()
