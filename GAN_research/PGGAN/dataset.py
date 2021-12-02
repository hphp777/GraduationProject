import pandas as pd

from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from PIL import Image

from IPython import display

import shutil
import os

# Read CSV
all_xray_df = pd.read_csv('./updated_data_entry.csv')

#Add ImageFile column
# all_image_paths = {os.path.basename(x): x for x in 
#                    glob('./archive/images_*/images/*.png')}

# print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
# all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)

pd.set_option('display.max_columns', None)
# print(all_xray_df.head())

# all_xray_df.to_csv('./updated_data_entry.csv')

diseases =  {
 'Atelectasis':1, 'Cardiomegaly':2, 'Consolidation':3, 'Edema':4, 'Effusion':5, 'Emphysema':6, 'Fibrosis':7, 'Hernia':8, 'Infiltration':9, 'Mass':10, 'Nodule':11, 'Pleural_Thickening':12, 'Pneumonia':13, 'Pneumothorax':14   
}

# continent 컬럼을 선택합니다.
# 컬럼의 값과 조건을 비교합니다.
# 그 결과를 새로운 변수에 할당합니다.
disease = all_xray_df['Finding Labels'] == 'Consolidation'

# lifeExp 컬럼을 선택합니다.
# 컬럼의 값과 조건을 비교합니다.
# 그 결과를 새로운 변수에 할당합니다.
female = all_xray_df['Patient Gender'] == 'F'
male = all_xray_df['Patient Gender'] == 'M'

# 두가지 조건를 동시에 충족하는 데이터를 필터링하여 새로운 변수에 저장합니다. (AND)
augF = all_xray_df[disease & female]
augM = all_xray_df[disease & male]

cnt = 0

for idx, row in augF.iterrows():
    cnt += 1
    pathFrom = row['path']
    pathTo = './x-ray/Consolidation_F'
    shutil.move(pathFrom, pathTo + '/' + str(idx) + '.png')

for idx, row in augM.iterrows():
    cnt += 1
    pathFrom = row['path']
    pathTo = './x-ray/Consolidation_M'
    shutil.move(pathFrom, pathTo + '/' + str(idx) + '.png')

print(cnt, "images moved")