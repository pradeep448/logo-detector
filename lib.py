# this is library file

from sklearn.ensemble import RandomForestClassifier
from skimage import exposure, feature
import numpy as np
import cv2 as cv
import glob
import os
import joblib
import sklearn
import sys
import platform as pf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import shutil

# Get Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
head, _ = os.path.split(ROOT_DIR)

######## functions #########

# path divider per OS type
def get_path_divider():
    if pf.system()=='Windows':
        path_div='\\'
    else:
        path_div='/'
    return path_div

# absolute path
def abspath(loc):
    return os.path.abspath(loc)

def predict(imagePath,model):
    image = cv.imread(imagePath)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    logo = cv.resize(gray, (200, 100))
    hist = feature.hog(
        logo,
        orientations=9,
        pixels_per_cell=(10, 10),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm="L1"
    )
    # Predict in model
    return model.predict(hist.reshape(1, -1))[0]

# class name mapper
def mapper(class_):
    if class_ in [1,'1']:
        return 'Present'
    elif class_ in [0,'0']:
        return 'Absent'
############################

path_div=get_path_divider()
model_path=f'model{path_div}model.pkl'