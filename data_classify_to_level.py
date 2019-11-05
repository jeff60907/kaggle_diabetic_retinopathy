# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 20:23:21 2019

@author: jeff
"""


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt # showing and rendering figures

IMG_SIZE = 512


def crop_image1(img,tol=7):
    # img is image data
    # tol  is tolerance
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)

        return img

def load_ben_color(path, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
    return image



base_image_dir = os.path.join('input', 'resized_train_cropped')
retina_df = pd.read_csv(os.path.join(base_image_dir, 'trainLabels_cropped.csv'))

retina_df['PatientId'] = retina_df['image'].map(lambda x: x.split('_')[0])
retina_df['path'] = retina_df['image'].map(lambda x: os.path.join(base_image_dir,
                                                         '{}.jpeg'.format(x)))

retina_df['exists'] = retina_df['path'].map(os.path.exists)
print(retina_df['exists'].sum(), 'images found of', retina_df.shape[0], 'total')


# Check directory different level
level_classes = [0,1,2,3,4]
for i in level_classes:
    dst_path = "./input/train/" + str(i)
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

# Copy pre-process image to different level
for i in range(retina_df['exists'].sum()):
    dst = 'input/train/' + str(retina_df['level'][i]) + '/' + str(retina_df['image'][i]) + '.jpeg'
    print(dst)
    path = retina_df['path'][i]
    image = load_ben_color(path,sigmaX=30)
    cv2.imwrite(dst, image)
