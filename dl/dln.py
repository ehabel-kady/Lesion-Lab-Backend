
import random
import pickle
import os
# from skimage.transform import resize
# from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import *
import tensorflow as tf
from keras.layers.merge import concatenate
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
# from matplotlib import pyplot as plt
import cv2
from shutil import copyfile
from keras.utils import plot_model
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array



# Helper Functions

# # Plotter
# def plotter(img,mask):
#   plt.figure(figsize=(7,7))
#   plt.subplot(121)
#   plt.imshow(np.squeeze(img), 'gray')
#   plt.subplot(122)
#   plt.imshow(np.squeeze(mask), 'gray')

image_width = 256
image_height = 256

# U-NET Architecture
def mean_iou(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))

def GDL(y_true,y_pred):
  intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
  pred_sum = tf.reduce_sum( y_pred, axis=[1, 2, 3])
  true_sum = tf.reduce_sum( y_true, axis=[1, 2, 3])

  return 1-((2*intersection)/(pred_sum+true_sum))


  
def UNet():
    inputs = Input((256, 256, 1))
    
    # Usage of Drouout to prevent overfitting - REF: https://keras.io/layers/core/

    #   Downsampling Layer Number 1
    conv1 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    

    #   Downsampling Layer Number 2
    conv2 = Conv2D(128, (3,3), padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    #   Downsampling Layer Number 3
    conv3 = Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    

    #   Downsampling Layer Number 4
    conv4 = Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)

    #   Bottleneck Layer
    conv5 = Conv2D(1024, (3,3), padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(1024, (3,3), padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    

    # Expansive Section
    up1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    up1 = concatenate([up1, conv4])
    conv6 = Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal')(up1)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)


    up2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    up2 = concatenate([up2, conv3])
    conv7 = Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal')(up2)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)


    up3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    up3 = concatenate([up3, conv2])
    conv8 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(up3)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)


    # up4 = concatenate([UpSampling2D((2,2))(conv8), conv1]) # 128

    up4 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv8)
    up4 = concatenate([up4, conv1])
    conv9 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(up4)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    

    conv10 = Conv2D(1, (1,1), padding='same', activation = 'sigmoid')(conv9)
    model = Model(inputs = [inputs], outputs = [conv10])
    model.compile(optimizer = Adam(lr=1e-6), loss = GDL, metrics = [mean_iou])
    
    # model.compile(optimizer = Adam(lr=1e-5), loss = "mean_squared_error", metrics = [dice_coef])
    
    return model



# Prediction Generation

def generatePredictions(model, images, imageNum):
  for image in range(len(images)):
    if images[image].shape != (256,256):
      newImg = cv2.resize(images[image], dsize=(256,256), interpolation=cv2.INTER_CUBIC)
    else:
      newImg = images[image]
    original_image = np.expand_dims(np.expand_dims(newImg/np.max(newImg), axis=0),-1)
    prediction = model.predict(original_image)

    # Saving to pngs for prediction contour mapping
    naming_original = naming_original = str(imageNum)+'_original.png'
    naming = str(imageNum)+'.png'
    prediction_to_save = img_to_array(np.squeeze(prediction))
    original_to_save = img_to_array(np.squeeze(original_image))
    save_img(naming, prediction_to_save)
    save_img(naming_original, original_to_save)

    # Reading saved files for final mapping
    # Loading images
    img = cv2.imread(naming)
    img_original = cv2.imread(naming_original)

    # Grayscale filters
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgray_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    # Drawing contours on original images
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_original,contours, -1, (255,255,255), 2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Saving the final predictions
    os.remove(naming)
    os.remove(naming_original)
    cv2.imwrite(naming_original, img_original)