from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from skimage import exposure, feature
import numpy as np
import cv2 as cv
import glob
import os
import joblib
import imgaug.augmenters as iaa
import sklearn
import sys


# Get Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
head, _ = os.path.split(ROOT_DIR)

# print(head,'\n',_)

def image_augment(img_path):
    # 1. Load Dataset
    images = []
    # images_path = glob.glob(f"{image_path}")
    # for img_path in images_path:
    img = cv.imread(img_path)
    print(np.shape(img))
    images.append(img)
        
    # 2. Image Augmentation
    augmentation = iaa.Sequential([
        # 1. Flip
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),

        # 2. Affine
        iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-30, 30),
                scale=(0.5, 1.5)),

        # 3. Multiply
        iaa.Multiply((0.8, 1.2)),

        # 4. Linearcontrast
        iaa.LinearContrast((0.6, 1.4)),

        # Perform methods below only sometimes
        iaa.Sometimes(0.5,
            # 5. GaussianBlur
            iaa.GaussianBlur((0.0, 3.0))
            )
    ])
    return augmentation(images=images)