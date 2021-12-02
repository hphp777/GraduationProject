import torch
from IPython.display import Image, clear_output  # to display images

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
from glob import glob

all_data = pd.read_csv('./Detection/BBox_List.csv',  index_col=0)

CLASS_NAMES = all_data['Finding Label'].unique().tolist()

print(CLASS_NAMES)

## move image

import shutil

def move_img(df, tgt_images_dir):
  for index, row in df.iterrows():
    src_image_path = row['FilePath']
    # 이미지 1개당 단 1개의 오브젝트만 존재하므로 class_name을 object_name으로 설정.  
    # yolo format으로 annotation할 txt 파일의 절대 경로명을 지정. 
    tgt = tgt_images_dir + row['Image Index']
    # image의 경우 target images 디렉토리로 단순 copy
    shutil.copy(src_image_path, tgt)


# move_img(all_data, './Detection/data/images/')


## Creating Annotation
def Creating_annotation(df, tgt_labels_dir):
    for index, row in df.iterrows():
        image_name = row['Image Index'].split('.')[0]
        output_txt_file = tgt_labels_dir + image_name +'.txt'
        with open(output_txt_file, 'w') as output_fpointer:
            class_id = CLASS_NAMES.index(row['Finding Label'])
            x_norm = round(row['x_mid'], 7)
            y_norm = round(row['y_mid'], 7)
            w_norm = round(row['w_n'], 7)
            h_norm = round(row['h_n'], 7)

            if (x_norm < 0) or (y_norm < 0) or (w_norm < 0) or (h_norm < 0):
                break

            value_str = ('{0} {1} {2} {3} {4}').format(class_id, x_norm, y_norm, w_norm, h_norm)
            output_fpointer.write(value_str+'\n')

# Creating_annotation(all_data, './Detection/data/labels/')

## yaml file
