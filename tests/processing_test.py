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

def test_preprocess_img_2():
    img = 'data/test_images/bateau_4_right.jpg'
    
    # get side
    pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
    result = pattern.search(img)
    side = result.group(2)

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
    img = 'data/test_images/bateau_4_right.jpg'

    # get side
    pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
    result = pattern.search(img)
    side = result.group(2)
    img_cv = cv2.imread(img)

    img_cropped = crop(img_cv, side=side)
    image_blurred = blur(img_cropped,1)
    cnts = get_contours(image_blurred)
    assert isinstance(cnts, list)

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

