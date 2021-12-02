import efficientnet.tfkeras as efn

import random
import cv2
from keras import backend as K
from keras.preprocessing import image
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.compat.v1.logging import INFO, set_verbosity


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
from glob import glob

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.models import load_model

# from tensorflow.keras.applications import DenseNet121
import tensorflow as tf
import tensorflow.keras.layers as L
# import tensorflow.keras.layers as Layers

random.seed(a=None, version=2)

set_verbosity(INFO)


# Read CSV
test_df = pd.read_csv('./archive/test_df.csv', index_col=0)
#train_df_main.drop(['No Finding'], axis = 1, inplace = True)
#labels = train_df_main.columns[2:-1]
labels = test_df.columns[2:0]

#Add ImageFile column
all_image_paths = {os.path.basename(x): x for x in 
                   glob('./archive/images_*/images/*.png')}

print('Scans found:', len(all_image_paths), ', Total Headers', test_df.shape[0])
test_df['FilePath'] = test_df['Image Index'].map(all_image_paths.get)

test_df = test_df.dropna(axis=0)

test_df.to_csv('./archive/test_df.csv')
print('Scans found:', len(all_image_paths), ', Total Headers', test_df.shape[0])