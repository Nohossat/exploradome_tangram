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
    img = 'data/test_images/bateau_4_right.jpg'
    
    # get side
    pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
    result = pattern.search(img)
    side = result.group(2)

    img_cv = cv2.imread(img)
    cnts, img = preprocess_img(img_cv, side=side)
    assert isinstance(cnts, list)
    assert isinstance(img, np.ndarray)

<<<<<<< HEAD
'''def test_preprocess_img_2():
    pass
=======
def test_preprocess_img_2():
    img = 'data/test_images/bateau_4_right.jpg'
    
    # get side
    pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
    result = pattern.search(img)
    side = result.group(2)
>>>>>>> b325b4ae29a0707c6fed2b571b5b4f4788886fa8

    img_cv = cv2.imread(img)
    cnts, img = preprocess_img_2(img_cv, side=side)
    assert isinstance(cnts, list)
    assert isinstance(img, np.ndarray)
    
def test_extract_triangles_squares():   
    img = 'data/test_images/bateau_4_right.jpg'
    
    # get side
    pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
    result = pattern.search(img)
    side = result.group(2)

    img_cv = cv2.imread(img)
    img_cropped = crop(img_cv, side=side)
    image_blurred = blur(img_cropped,1)
    first_cnts = get_contours(image_blurred)
    img = extract_triangles_squares(first_cnts, image_blurred)
    assert isinstance(img, np.ndarray)

def test_blur():
    img = 'data/test_images/bateau_4_right.jpg'
    img_cv = cv2.imread(img)
    image_blurred = blur(img_cv, 3)
    assert isinstance(image_blurred, np.ndarray)

def test_get_contours():
<<<<<<< HEAD
    pass'''
=======
    img = 'data/test_images/bateau_4_right.jpg'
>>>>>>> b325b4ae29a0707c6fed2b571b5b4f4788886fa8

    # get side
    pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
    result = pattern.search(img)
    side = result.group(2)
    img_cv = cv2.imread(img)
<<<<<<< HEAD
    image_resize = resize(img_cv, side=side)
    assert image_resize.shape[0:2] == (494, 360), 'image is too big'
 
'''def test_display_contour():
    pass
=======

    img_cropped = crop(img_cv, side=side)
    image_blurred = blur(img_cropped,1)
    cnts = get_contours(image_blurred)
    assert isinstance(cnts, list)
>>>>>>> b325b4ae29a0707c6fed2b571b5b4f4788886fa8

def test_extract_triangles_squares_2():    
    img = 'data/test_images/bateau_4_right.jpg'
    
    # get side
    pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
    result = pattern.search(img)
    side = result.group(2)

    img_cv = cv2.imread(img)
    img_cropped = crop(img_cv, side=side)
    image_blurred = blur(img_cropped,1)
    first_cnts = get_contours(image_blurred)
    cnts, img = extract_triangles_squares_2(first_cnts, image_blurred)

    assert isinstance(cnts, list)
    assert isinstance(img, np.ndarray)

def test_crop():
    img = 'data/test_images/bateau_4_right.jpg'

    # get side
    pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
    result = pattern.search(img)
    side = result.group(2)

    img_cv = cv2.imread(img)
    previous_size = img_cv.shape
    print(previous_size)

    img = crop(img_cv, side=side)

    assert previous_size > img.shape, "The cropping didn't occur"

<<<<<<< HEAD
def test_detect_white_color():
    pass
'''
=======
>>>>>>> b325b4ae29a0707c6fed2b571b5b4f4788886fa8
