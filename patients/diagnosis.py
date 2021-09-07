from .classifier_model import *

# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# sys.path.append("C:\Users\Choi\Desktop\GP\GraduationProject01-main")
from yolov5.detect import *


def diagnosis(model, img, image_dir, df, labels):
    preprocessed_input = load_image(img, image_dir, df)
    predictions = model.predict(preprocessed_input)[0]
    predictions_sort = list(reversed(sorted(predictions)))
    disease = []
    percentage = []
    for i in range(0, 3):
        index = np.argwhere(predictions == predictions_sort[i])[0][0]
        disease.append(labels[index])
        percentage.append(predictions[index] * 100)
    return disease, percentage


def detection(img):
    main(img)
