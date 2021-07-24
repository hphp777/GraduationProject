# 이미 학습된 가중치를 이용해서 질병을 진단하기

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

from . import classifier_model as cm

cm.model.load_weights('../input/nih-chest-xray-training-weights/efficent_net_b1_trained_weights.h5')

result = cm.model.predict(image)

disease1 = np.argmax(result)