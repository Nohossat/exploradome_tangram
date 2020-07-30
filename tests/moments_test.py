
"""
Created on Tue Jul 28 15:11:06 2020

@author: Renata
"""

import pytest
import cv2
import numpy as np
import imutils
import pandas as pd
import os
 
from moments.py import *

def get_files_test():
    dicto = {'bateau.jpg', }
    assert get_files() == dicto
    #assert os.path.exists(dirname), "the directory doesn't exist"

def preprocess_img_test():
    from PIL import Image
try:
    im=Image.open(filename)
    # do stuff
except IOError:
    # filename not an image file
    # verification extension of file *.img
    
def find_humos_test(img,filename, sensitivity_to_light=50):
    os.path.exists(dirname), "the directory doesn't exist"

def resize_test():
    os.path.exists(dirname), "the directory doesn't exist"

def save_moments_test():
    os.path.exists(dirname), "the directory doesn't exist"
    
def read_video_test(video=False):
    #verification de path of vidéo
    if __name__ == '__main__':
    #test distance
    get_distance_img_test('cygne)
    