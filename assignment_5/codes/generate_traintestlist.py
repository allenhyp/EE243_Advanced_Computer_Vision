import numpy as np
import glob
import scipy.io as sio 

data_folder = '/data/ece243/cifar10'

train_path = os.path.join(data_folder,'train_images')
test_path = os.path.join(data_folder,'test_images')
train_annotationpath = os.path.join(data_folder,'cifar10_train_labels.mat')
test_annotationpath = os.path.join(data_folder,'cifar10_test_labels.mat')

train_images = sorted(glob.glob(train_path + '/*.jpg'))
test_images = sorted(glob.glob(test_path + '/*.jpg'))

train_annotations = sio.loadmat(train_annotationpath)['L']
test_annotations = sio.loadmat(test_annotationpath)['L']

fid = open('train.list','w')
for i,p in enumerate(train_images):
  fid.write(p + ' ' + str(train_annotations[i][0]) + '\n') 
fid.close()

fid = open('test.list','w')
for i,p in enumerate(test_images):
  fid.write(p + ' ' + str(test_annotations[i][0]) + '\n')
fid.close()






