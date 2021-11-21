# Importing the required libraries

import numpy as np
import cv2
import os
import pickle
import sys
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

n_classes=36
clustering_factor=6

import tensorflow as tf

# Implementing Bag of Features Model

def surf_features(images):
  surf_descriptors_class_by_class={}
  surf_descriptors_list=[]
  surf=cv2.xfeatures2d.SURF_create()
  for key,value in images.items():
    print(key, "Started")
    features=[]
    for img in value:
      kp,desc=surf.detectAndCompute(img,None)
      surf_descriptors_list.extend(desc)
      features.append(desc)
    surf_descriptors_class_by_class[key]=features
    print(key," Completed!")
  return [surf_descriptors_list,surf_descriptors_class_by_class]

# Creating a visual dictionary using only the train dataset 
# K-means clustering alogo takes only 2 parameters which are number of clusters (k) and descrpitors list
# It reurn an array which holds central points

def minibatchkmeans(k, descriptors_list):
  kmeans=MiniBatchKMeans(n_clusters=k)
  print("MiniBatchKMeans Initialized!")
  kmeans.fit(descriptors_list)
  print("Clusters Created!")
  visual_words=kmeans.cluster_centers_
  return visual_words, kmeans

# Loading train images into dictionaries which holds all images category by category

def load_images_by_category(folder):
  images={}
  for label in os.listdir(folder):
    print(label," started")
    category=[]
    path=folder+'/'+label
    for image in os.listdir(path):
      img=cv2.imread(path+'/'+image)
      #new_img=cv2.resize(img,(128,128))
      if img is not None:
        category.append(img)
    images[label]=category
    print(label, "ended")
  return images

# Creating histograms for train images

# Function takes 2 parameters. The first one is a dictionary that holds the descriptors that are separated class by class 
# And the second parameter is the clustered model
# Returns a dictionary that holds the histograms for each images that are separated class by class. 

def create_histogram(all_descs,kmeans):
  features_dict={}
  for key,value in all_descs.items():
    print(key," Started!")
    category=[]
    for desc in value:
      raw_words=kmeans.predict(desc)
      hist = np.array(np.bincount(raw_words,minlength=n_classes*clustering_factor))
      category.append(hist)
    features_dict[key]=category
    print(key," Completed!")
  return features_dict

train_folder='D:/Indian Sign Language/Code/Train-Test/Train'

# Load train images
train_images=load_images_by_category(train_folder)
#print(len(train_images))

#print(len(train_images['a'][0][0]))

#Extracting surf features from each image stored in train_images list

surfs=surf_features(train_images)
all_train_descriptors=surfs[0]
train_descriptors_by_class=surfs[1]

#print(len(surfs[0]))
#print(len(surfs[1]['0'][1]))

# Calling MiniBatchkmeans function and getting central points
visual_words,kmeans=minibatchkmeans(n_classes*clustering_factor,all_train_descriptors)


# Calling create_histogram and getting histogram for each image
bows_train=create_histogram(train_descriptors_by_class,kmeans)

#print((bows_train['a'][0][1]))

# Saving .csv file
import csv
loc='D:/Indian Sign Language/Code/Code/Classification/csv files/train.csv'
with open(loc,'w',newline='') as file:
  writer=csv.writer(file)
  header=[]
  for i in range (1,n_classes*clustering_factor+1):
    header.append(str('pixel')+str(i))
  header.append('Label')
  writer.writerow(header)
  count=0
  for label in bows_train:
     # print(len(bows_train[label]))
    for i in range(len(bows_train[label])):
      list=[]
      for j in range(216):
        list.append(bows_train[label][i][j])
      list.append(label)
      writer.writerow(list)

