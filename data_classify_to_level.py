# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 20:23:21 2019

@author: jeff
"""


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from shutil import copyfile

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
    dst_path = "./input./train/" + str(i)
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

# Copy file to different level
for i in range(len(retina_df)):
    dst = './input/train/'+ str(retina_df['level'][i])+'/' + str(retina_df['image'][i]) + '.jpeg'
    copyfile(retina_df['path'][i], dst)
