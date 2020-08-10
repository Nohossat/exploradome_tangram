from .processing import preprocess_img, display_contour
from .moments import find_moments
import pandas as pd
import os
import re
import cv2
import numpy as np

DATA_PATH = 'data/'

def get_files(directory = DATA_PATH + '/tangrams'):
    """
    get a dict with the image_name and the path of all images in tangrams folder
    author : @Nohossat
    """
    images = []
    assert os.path.exists(directory), "the directory doesn't exist"

    for folder, sub_folders, files in os.walk(directory):
        for file in files:
            filename, file_extension = os.path.splitext(file) # we just want the filename to save the path
            file_path = os.path.join(folder, file)
            if file.endswith((".jpg", ".png")) and not file.startswith('frame'):
                pattern = re.compile(r"[a-zA-Z]+") # in case there is any number or underscore in the name
                label = pattern.match(filename).group()
                images.append((label, file_path))

    return images

def get_nb_corners(img):
    """
    Parameters :
    img : OpenCV

    Returns :
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,5,3,0.04)
    ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

    """
    This part is not for production
    It allow to visualize on an image the corner point in red
    """
    #for i in range(1, len(corners)):
        # print(corners[i])
        # print(len(corners))
        # img[dst>0.1*dst.max()]=[0,0,255]
        # cv2_imshow(img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows
        
    return len(corners)
"""
Results nb of corners
bol = 12
####
coeur = 13
marteau = 13
montagne = 13
####
bateau = 14
maison = 14
tortue = 14
####
chat = 15 
cygne = 15
pont = 15 
####
lapin = 19
####
renard = 20 
"""
