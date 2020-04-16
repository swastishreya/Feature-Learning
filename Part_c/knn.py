import os
import numpy as np 
import cv2
import torch
import torch.nn as nn
from torchvision import models




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

def getDescriptors(images, labels, model) : 
    features = []
    image_labels = []
    i = 0

    for image in images : 
        print (image.shape)
        # Re-sizing the image
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        feature = model(image)
        if des is not None : 
            features.append(feature)
            image_labels.append(labels[i])
        i += 1

        
    return features, image_labels


model = models.alexnet(pretrained=True)
new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model.classifier = new_classifier
model.eval()


root = "/home/slr/Desktop/vr/Assignments/Feature-Learning/101_ObjectCategories/"

set_images = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]

images, labels = read_images("./../101_ObjectCategories/", set_images)
features, image_labels = getDescriptors(images, labels, model)

# k = 50
# visDic = MiniBatchKMeans(init='k-means++',
#                             n_clusters=50,
#                             max_iter=1000,
#                             batch_size=1000, 
#                             n_init=10, 
#                             max_no_improvement=10, 
#                             verbose=0).fit(alexnet_des)

x_train, x_test, y_train, y_test = train_test_split(features, 
                                                    image_labels, 
                                                    test_size=0.1, 
                                                    random_state=4)

clf = cv2.ml.KNearest_create()
clf.train(X_train, cv2.ml.ROW_SAMPLE, np.asarray(y_train, dtype=np.float32))

ret, results, neighbours ,dist = clf.findNearest(X_test, k=10)

pred_label = []
for var in results:
    label = var
    pred_label.append(int(label))

print (y_test)
print (pred_label)
    
# Measuring the accuracies
metrics.accuracy_score(y_test, pred_label)