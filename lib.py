from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure, feature
import numpy as np
import cv2 as cv
import glob
import os


# Get Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
head, _ = os.path.split(ROOT_DIR)

# print(head,'\n',_)