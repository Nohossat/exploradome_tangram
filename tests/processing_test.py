"""
@author: Renata
"""

import pytest
import cv2
import numpy as np
import imutils
import pandas as pd
import os
import re
 
from tangram_app.processing import *

def test_preprocess_img():
    img = 'data/tangrams/bateau_4_right.jpg'
    pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
    result = pattern.search(img)
    side = result.group(2)
    img_cv = cv2.imread(img)
    result = preprocess_img(img_cv, side=side)
    assert isinstance(result, list)

def test_preprocess_img_2():
    pass

def test_extract_triangles_squares():   
    pass

def test_blur()
    pass

def test_get_contours():
    pass

#test cv image and resizes it. 
def test_resize():
    img = 'data/tangrams/bateau_4_right.jpg'
    # get size to analyze from image path
    pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
    result = pattern.search(img)
    side = result.group(2)
    img_cv = cv2.imread(img)
    image_resize = resize(img_cv, side=side)
    assert image_resize.shape[0:2] == (494, 360), 'image is too big'
 
def test_display_contour():
    pass

def test_extract_triangles_squares_2():    
    pass

def test_crop():
    pass

def test_contour_intersect():
    pass

def test_detect_black_color():
    pass

def test_detect_white_color():
    pass
