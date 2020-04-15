import os
from alexnet import AlexNet
import numpy as np 
import cv2
import tensorflow as tf


def read_images(path, folders):
    images = []
    labels = []
    idx = 0
    for folder in folders:
        for filename in os.listdir(path+folder):
            image = cv2.imread(os.path.join(path+folder, filename))
            if image is not None:
                images.append(image)
                labels.append(idx)
                
        idx += 1
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def getDescriptors(images) : 
    descriptors = []
    
    for image in images : 
        print (image.shape)
        # Re-sizing the image
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        des = AlexNet(image)
        if des is not None : 
            descriptors.append(des)
            
    descriptors = np.concatenate(descriptors, axis=0)
    descriptors = np.asarray(descriptors)
        
    return descriptors

root = "/home/slr/Desktop/vr/Assignments/Feature-Learning/101_ObjectCategories/"

set_images = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]

images, labels = read_images("./../101_ObjectCategories/", set_images)
alexnet_des = getDescriptors(images)


# x = tf.placeholder(tf.float32, (None, 32, 32, 3))
# resized = tf.image.resize_images(x, (227, 227))
