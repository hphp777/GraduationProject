import torch
from IPython.display import Image, clear_output  # to display images

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
from glob import glob

clear_output()
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

all_data = pd.read_csv('./archive/BBox_List_2017.csv')
all_data.head()

size = 1024

all_data.rename(columns = {'Bbox [x' : 'x_min'}, inplace = True)
all_data.rename(columns = {'y' : 'y_min'}, inplace = True)
all_data.rename(columns = {'h]' : 'h'}, inplace = True)

all_data = all_data.drop(['Unnamed: 6','Unnamed: 7','Unnamed: 8'], axis = 1)

all_data['x_max'] = all_data.apply(lambda row: row.x_min + row.w, axis =1)
all_data['y_max'] = all_data.apply(lambda row: row.y_min + row.h, axis =1)

# calculation x-mid, y-mid, width and hight of the bounding box for yolo
all_data['x_mid'] = all_data.apply(lambda row: (row.x_max+row.x_min)/2, axis =1)
all_data['y_mid'] = all_data.apply(lambda row: (row.y_max+row.y_min)/2, axis =1)

all_data['x_mid'] /= float(size)
all_data['y_mid'] /= float(size)

all_data['w_n'] =  all_data.apply(lambda row: (row.w)/size, axis =1)
all_data['h_n'] =  all_data.apply(lambda row: (row.h)/size, axis =1)

# 이미지이름: 이미지경로 를 만들어주는 딕셔너리
all_image_paths = {os.path.basename(x): x for x in 
                   glob('./archive/images_*/images/*.png')}
all_data['FilePath'] = all_data['Image Index'].map(all_image_paths.get)

all_data.head()

all_data.to_csv('./Detection/BBox_List.csv')