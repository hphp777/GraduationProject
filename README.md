:crystal_ball:Chest X-ray Disease Diagnosis
=============

Due to the global pandemic, the number of emergency patients continues to increase. More than
half (56%) of patients visiting the emergency room are undergoing x-ray imaging. However, the time required for chest x-ray diagnosis in the emergency room takes about an hour, which is very long for emergency patients who need proper emergency treatment within golden time. In the case of system in this paper, x-ray imaging, which omits additional tests, is shortened to two stages of diagnosis, and the diagnosis speed of diagnosis is drastically shortened to around 12 seconds. To implement this, Classification and Detection-based Web Service was implemented, and Synthetic Medical X-ray Data were created using PGGAN to improve Classification performance. It is expected that these system will be used to improve the efficiency of medical process such as specifying problematic area, reducing x-ray processing time, and accurate disease classification in the emergency case.

<img src="https://user-images.githubusercontent.com/71118045/144365814-59484285-4c11-48ce-8e2a-cd99ccc134ad.PNG" width="600" height="300"/>

It is an explanation of the system structure of the program. Starting from the right, uploading the x-ray image from the website requests classification and detection diagnosis of the image at the backend. Then, the uploaded image is classified with the learned model. In the case of Detection, learning is conducted using the yolov5 model. Then, the suspected disease area is detected with the uploaded image and the resulting image is stored. Through this process, a screen is finally output as a result of diagnosing the patient's disease on the website screen and detecting the suspected disease occurrence site. Three possible diseases are presented to help doctors diagnose diseases quickly.

<img src="https://user-images.githubusercontent.com/71118045/144364448-1fdd71ac-c470-46ed-8b56-d014affe6594.PNG" width="500" height="300"/>

## :sparkles:0. Dataset

Chest X-ray exams are one of the most frequent and cost-effective medical imaging examinations available. However, clinical diagnosis of a chest X-ray can be challenging and sometimes more difficult than diagnosis via chest CT imaging. The lack of large publicly available datasets with annotations means it is still very difficult, if not impossible, to achieve clinically relevant computer-aided detection and diagnosis (CAD) in real world medical sites with chest X-rays. One major hurdle in creating large X-ray image datasets is the lack resources for labeling so many images. Prior to the release of this dataset, Openi was the largest publicly available source of chest X-ray images with 4,143 images available.

This NIH Chest X-ray Dataset is comprised of 112,120 X-ray images with disease labels from 30,805 unique patients. To create these labels, the authors used Natural Language Processing to text-mine disease classifications from the associated radiological reports. The labels are expected to be >90% accurate and suitable for weakly-supervised learning. The original radiology reports are not publicly available but you can find more details on the labeling process in this Open Access paper: "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases." (Wang et al.)

###### https://www.kaggle.com/nih-chest-xrays/data

## :sparkles:1. Classification

<img src="https://user-images.githubusercontent.com/71118045/144366234-16d81ff2-3980-4499-96c6-4c4218b2dd28.PNG"  width="900" height="300"/>

###### Weights: [Google Drive Link](https://drive.google.com/drive/folders/1-uo9GchtOoAFvXmE0zpPi0eaFgKNOrk6?usp=sharing)

## :sparkles:2. Detection

###### Training Data: [Google Drive Link](https://drive.google.com/drive/folders/11CUJGctnzHQcsq9O3WCSTRhgRjkMOOUN?usp=sharing)
###### Ultralytics Yolov5 : https://github.com/ultralytics/yolov5

<img src="./Result_Image/Detection_NIH_200.jpg"  width="400" height="400"/>

###### Result of NIH Data: [Google Drive Link](https://drive.google.com/drive/folders/1qo_5ICzeMUrHQ_-s0Z9d3KYSLCrNzqRl?usp=sharing)
###### Result of ChestX Data : [Google Drive Link](https://drive.google.com/drive/folders/1NBvWFz3Fto6ZqeLrqopEMlbUZnNpxodN?usp=sharing)
###### Result of NIH & ChestX Data: [Google Frive Link](https://drive.google.com/drive/folders/1Koryg3pxeUs7oJ0ulO7FEjrq0EMPB6of?usp=sharing)

## :sparkles:3. GAN Research

<img src="https://user-images.githubusercontent.com/85219925/144364517-22de0573-d468-433f-95d4-b4eaac10f902.png"  width="400" height="400"/>

###### Generated Image(PGGAN1): [Google Drive Link](https://drive.google.com/drive/folders/1qJj4dn9ap-fPbrHuP2OR9f7_tTKUm58L?usp=sharing)
###### Generated Image(PGGAN2): [Google Drive Link](https://drive.google.com/drive/folders/1IWavLvJQTNJ_Ui-s0R7is2MTI1Q3naOe?usp=sharing)
###### Generated Image(PGGAN3): [Google Drive Link](https://drive.google.com/drive/folders/1q1PmqqxZPPGEzazGkzOXv4WF1G5zFNO1?usp=sharing)
###### PGGAN Weights: [Google Drive Link](https://drive.google.com/drive/folders/1Y9l7wqjt-cKR-gJRIe8DqZwbG91nyXEy?usp=sharing)

###### Generated Image(DCGAN): [Google Drive Link](https://drive.google.com/drive/folders/18MekMJsuhZS6Shu3T6nvmNihK4M4oilz?usp=sharing)

## :book:Papers
###### 1. 폐질환 의심 응급환자의 진단 과정 단축을 위한 AI기반 X-ray진단 시스템
###### https://drive.google.com/file/d/1FnQGBRWvJ70iH2Rut0L7hjO-4Bt15vpc/view?usp=sharing
###### 2. PGGAN synthetic data를 활용한 Class간 데이터분포의 불균형 완화가 X-ray 질병 진단 정확도에 미치는 영향 연구
###### https://drive.google.com/file/d/1OPLWdxKm7L0jW0QhTIYEo4-7AER3-JNz/view?usp=sharing
