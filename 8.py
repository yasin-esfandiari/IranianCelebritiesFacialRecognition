# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 15:39:37 2018

@author: yasin esfandiari
"""

import tensorflow as tf
import cv2
import re
import detect_face
import FaceToolKit as ftk
from scipy import misc
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier

verification_threshhold = 1.1
image_tensor_size = 160

# Initialize required variables
base_dir = './face_data/data_croped_mtcnnpy_160'
IDs = os.listdir(base_dir)
images = []
labels = []
image_paths = []
#-------
default_color = (0, 255, 0) #BGR
default_thickness = 2
margin = 20


# Class instantiations
v = ftk.Verification()

# Pre-load model for Verification
v.load_model("./models/20180204-160909/")
v.initial_input_output_tensors()

#step 6
ID2label = {
        168:0,
        220:1,
        357:2,
        439:3,
        451:4,
        455:5,
        461:6,
        471:7,
        473:8,
        476:9
        }
# Traversing Directories
for ID in IDs:
    sub_dir = os.path.join(base_dir, ID)
    photos = os.listdir(sub_dir)
    results = []
    for photo in photos:
        results.append(os.path.join(sub_dir, photo))
        image_paths.append(os.path.join(sub_dir, photo))
    sub_dir_len = len(results)
    for i in range(sub_dir_len):
        images.append(v.img_to_encoding(misc.imread(results[i]), image_tensor_size))
        labels.append(ID2label[int(ID)])

Label_name = {
        0:"Hassan Joharchi",
        1:"Daanial Hakimi",
        2:"Ali Saadeqi(Actor)",
        3:"Kaambiz Dirbaaz",
        4:"Gloriaa Haardy",
        5:"Lindaa Kiaani",
        6:"La'yaa Zangane",
        7:"Majid Majidi",
        8:"Mohsen Afshaani",
        9:"Mahtaab Keraamati",
        }

#step 7
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(images, labels)

#step 8
image_base_dir = './input/'
image_paths = sorted([f for f in os.listdir(image_base_dir) if re.match(r'.+\.jpg', f)])
print(image_paths)

with tf.Graph().as_default():
    sess = tf.Session()
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

for image_path in image_paths:

    print(image_base_dir + image_path)

    img_orig = cv2.imread(image_base_dir + image_path)
    img = np.copy(img_orig)
    bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    for bounding_box in bounding_boxes:
        pts = bounding_box[:4].astype(np.int32)
        pts_margined = [pts[0]-margin, pts[1]-margin, pts[2]+margin, pts[3]+margin]
        
        img_test = cv2.imread(image_base_dir + image_path)
        img_test = img_test[pts_margined[1]:pts_margined[3], pts_margined[0]:pts_margined[2]]
        img_test = cv2.resize(img_test, (160,160), interpolation = cv2.INTER_AREA)
        img_test_encoded = v.img_to_encoding(img_test, image_tensor_size)
        img_listed = [list(img_test_encoded)]
#        img_listed = img_listed.reshap
        
        predicted_class = neigh.predict(img_listed)
        nearest_index = labels.index(predicted_class[0])
        nearest = images[nearest_index]
        
        #distance
        diff = np.subtract(nearest, img_test_encoded)
        dist = np.sum(np.square(diff))
        
        if (dist>verification_threshhold):
            cv2.putText(img, "Unknown", (pts[0]-5, pts[0]-15), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 255, 255), 1)
        else:
            cv2.putText(img, Label_name[predicted_class[0]], (pts[0]-5, pts[0]-15), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 255, 255), 1)            
        
        pt1 = (pts_margined[0], pts_margined[1])
        pt2 = (pts_margined[2], pts_margined[3])
        cv2.rectangle(img, pt1, pt2, color=default_color, thickness=default_thickness)

    for i in range(points.shape[1]):
        pts = points[:, i].astype(np.int32)
        for j in range(pts.size // 2):
            pt = (pts[j], pts[5 + j])
            cv2.circle(img, center=pt, radius=1, color=default_color, thickness=default_thickness)

    separator = np.zeros((img_orig.shape[0], 20, 3), np.uint8)
    cv2.imwrite('./output/' + image_path, np.hstack((img_orig, separator, img)))
    #cv2.moveWindow(image_path, 50, 50)
    #cv2.waitKey(0)
    #cv2.destroyWindow(image_path)