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
 
