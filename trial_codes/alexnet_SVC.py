import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable

import copy

import torchvision
from torchvision import datasets, models, transforms

from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import StratifiedKFold  

data_transforms = {
   'train': transforms.Compose([
       transforms.ToPILImage(),
       transforms.RandomResizedCrop(224),
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ]),
   'val': transforms.Compose([
       transforms.ToPILImage(),
       transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ])
}

def read_images(path, folders):
    images = []
    labels = []
    idx = 0
    for folder in folders:
        for filename in os.listdir(path+folder):
            image = os.path.join(path+folder, filename)
            if image is not None:
                images.append(image)
                labels.append(idx)
                
        idx += 1
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


train_data_dir = '/home/swasti/Documents/sem6/VR/Assignment3/Feature-Learning/trial_codes/101_ObjectCategories/101_ObjectCategories/train/'
val_data_dir = '/home/swasti/Documents/sem6/VR/Assignment3/Feature-Learning/trial_codes/101_ObjectCategories/101_ObjectCategories/val/'

def getDescriptors(images, labels, model, phase) :
    features = []
    image_labels = []
    i = 0

    for image in images:

        img = Image.open(image)

        try:
            if phase == 'train':
                trans2 = transforms.RandomResizedCrop(224)
                trans3 = transforms.RandomHorizontalFlip()
                trans4 = transforms.ToTensor()
                trans5 = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                img = trans5(trans4(trans3(trans2(img))))
            else:
                trans2 = transforms.Resize(256)
                trans3 = transforms.CenterCrop(224)
                trans4 = transforms.ToTensor()
                trans5 = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                img = trans5(trans4(trans3(trans2(img))))

            img = img.unsqueeze(0)
            feature = model(img)
            feature = feature.data.numpy().reshape(4096)
            if feature is not None :
                features.append(feature)
                image_labels.append(labels[i])
            i += 1
        except:
            continue

    return features, image_labels

folders = [ item for item in os.listdir(train_data_dir) if os.path.isdir(os.path.join(train_data_dir, item)) ]

train_images, train_labels = read_images(train_data_dir, folders)
val_images, val_labels = read_images(val_data_dir, folders)


alexnet = torchvision.models.alexnet(pretrained=True)
classifier_list = list(alexnet.classifier)

alexnet.classifier = nn.Sequential(classifier_list[0],
                                   classifier_list[1],
                                   classifier_list[2],
                                   classifier_list[3],
                                   classifier_list[4],
                                   classifier_list[5])


alexnet.eval()

train_features, train_labels = getDescriptors(train_images, train_labels, alexnet, 'train')
val_features, val_labels = getDescriptors(val_images, val_labels, alexnet, 'val')

pca = PCA(n_components = 500)
train_features = pca.fit_transform(train_features)
val_features = pca.transform(val_features)

svc = svm.SVC(kernel='linear')
svc.fit(train_features, train_labels)
score = svc.score(val_features, val_labels)
print("Score: {}".format(score))

# Features Score for Ants and Bees

# Without PCA : Score: 0.8758169934640523
# PCA n_components = 244 : Score: 0.8627450980392157
# PCA n_components = 220 : Score: 0.9019607843137255
# PCA n_components = 210 : Score: 0.8823529411764706
# PCA n_components = 200 : Score: 0.9019607843137255
# PCA n_components = 180 :Score: 0.8954248366013072
# PCA n_components = 100 : Score: 0.8954248366013072


# Features Score for 101_ObjectCategories
# PCA n_components = 210 : Score: 0.46727748691099474
# Without PCA : Score: 0.49476439790575916
# PCA n_components = 300 : Score: 0.4725130890052356
# Without last two layer n PCA : Score: 0.4620418848167539
# Without last layer n PCA : Score: 0.5039267015706806
# Without last layer n PCA n_components = 300 : Score: 0.4620418848167539
# Without last layer n PCA n_components = 500 : Score: 0.5013089005235603
