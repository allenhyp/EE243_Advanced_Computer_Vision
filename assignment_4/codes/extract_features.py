import os
import numpy as np
import tensorflow as tf
import cv2
import scipy.io as sio
import glob
import numpy as np

''' Works with tensorflow >= 1.5 '''

# SPECIFY PATH TO THE DATASET
path_to_dataset = '/data/ece243/tiny-UCF101'

def main():

  feature = []
  label = []
  categories = sorted(os.listdir(path_to_dataset))

  # FILL IN TO LOAD THE ResNet50 MODEL
  resnet = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    pooling=max)

  for i,c in enumerate(categories):
    path_to_images = sorted(glob.glob(os.path.join(path_to_dataset,c) + '/*.jpg'))
    for p in path_to_images:
      # FILL IN TO LOAD IMAGE, PREPROCESS, EXTRACT FEATURES. 
      # OUTPUT VARIABLE F EXPECTED TO BE THE FEATURE OF THE IMAGE OF DIMENSION (2048,)
      img = tf.keras.preprocessing.image.load_img(p, target_size=(227, 227))
      x = tf.keras.preprocessing.image.img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = tf.keras.applications.resnet50.preprocess_input(x)

      f = resnet.predict(x)
      F = f.squeeze()
      feature.append(F)
      label.append(categories.index(c))
    print(np.shape(feature))

  sio.savemat('ucf101dataset.mat', mdict={'feature': feature, 'label': label})


if __name__ == "__main__":
   main()
