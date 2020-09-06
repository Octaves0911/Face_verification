import tensorflow as tf 
import numpy as np
import os
from keras.layers import Conv2D, Activation, AveragePooling2D, MaxPooling2D, ZeroPadding2D, Input, concatenate
from keras.layers.core import Lambda, Dense, Flatten
from numpy import genfromtxt
import cv2
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers import *
from keras.models import Model
from sklearn.preprocessing import normalize
K.set_image_data_format('channels_first')
import random
import keras

import argparse
import sys


def triplet_loss(y_true, y_pred, alpha = 0.3):

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)

    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss

 


#To loacaloze the face and resize the image
def image_resizing(image,path_haar='haarcascade_frontalface_default.xml'):
    #image=cv2.imread(path_image)
    #image=image.astype('float32')/255.0
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    classifier=cv2.CascadeClassifier(path_haar)
    faces=classifier.detectMultiScale(gray,1.1,6)
    #print(type(faces))
    if len(faces)!=1:
        print('More than one Image in the selfie')
        sys.exit(0)
    x,y,w,h=faces.squeeze()
    crop=image[y:y+h,x:x+w]
    image=cv2.resize(crop,(96,96))
    #image=cv2.resize(image,(96,96))
    #image=np.transpose(image,(2,0,1))
    #image=image.astype('float32')/255.0
    return image

#function to encode the given image
def encode_img(img1,model):
    #img1=cv2.imread(path,1)
    img=img1[...,::-1]
    img=np.around(np.transpose(img,(2,0,1))/255,decimals=12)
    x_train=np.array([img])
    emb=model.predict_on_batch(x_train)
    return emb


threshold=0.65
interval=0.3
def confidence_value(ref_encode,img_encode,thres=threshold):
    #diff=np.max(img_encode-ref_encode)
    dist=np.linalg.norm((img_encode-ref_encode))
    #confidence=(1-K.eval(tf.minimum(dist,1)))
    confidence=(threshold-max([dist,interval]))/(threshold-interval)
    return dist,confidence


arg=argparse.ArgumentParser()
arg.add_argument("-r","--rimage",required=True,help="The reference image path")
arg.add_argument("-i","--image",required=True,help="The image file path")
args=vars(arg.parse_args())


#rimage=cv2.imread(rimage)
#image=cv2.imread(image)

rimage=cv2.imread(args["rimage"])
image=cv2.imread(args["image"])
#print(args["rimage"])
#print(args["image"])
#print(rimage.shape)
#print(image.shape)

rimg=image_resizing(rimage)
img=image_resizing(image)


model=tf.keras.models.load_model('One_Shot_model.h5',custom_objects={'triplet_loss': triplet_loss})
r_encode=encode_img(rimg,model)
img_encode=encode_img(img,model)




dist,conf=confidence_value(r_encode,img_encode)
#print(dist,"  ",conf)

if dist<threshold:
    print("Match with a confidence of ",conf*100)
    #print("Distance ",dist)
else:
    print("No Match with a confidence of ",abs(conf*100))
