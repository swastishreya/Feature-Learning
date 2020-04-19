import os
import shutil

path = "/home/swasti/Documents/sem6/VR/Assignment3/Feature-Learning/trial_codes/101_ObjectCategories/101_ObjectCategories/"
folders = [ item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item)) ]

for folder in folders:
    images = []
    for filename in os.listdir(path+folder):
        source = os.path.join(path+folder, filename)
        images.append(source)

    num_images = len(images)
    train_images = images[ :(num_images - int(num_images/10))]
    test_images = images[(num_images - int(num_images/10)):]
    
    train_path = "/home/swasti/Documents/sem6/VR/Assignment3/Feature-Learning/trial_codes/101_ObjectCategories/101_ObjectCategories/train"
    test_path = "/home/swasti/Documents/sem6/VR/Assignment3/Feature-Learning/trial_codes/101_ObjectCategories/101_ObjectCategories/val"

    train_des = os.path.join(train_path, folder)
    os.makedirs(train_des)
    for filename in train_images:
        shutil.move(filename, train_des)

    test_des = os.path.join(test_path, folder)
    os.makedirs(test_des)
    for filename in test_images:
        shutil.move(filename, os.path.join(test_path, test_des))