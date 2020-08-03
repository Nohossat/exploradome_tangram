
"""


@author: Renata
"""

import pytest
import cv2
import numpy as np
import imutils
import pandas as pd
import os
 
from ..processing import *

def test_preprocess_img():
    assert preprocess_img(img, side=None, crop = True, sensitivity_to_light=50)== img[0:int(img.shape[0]),int(img.shape[1]/2):]

#test cv image and resizes it. 
def test_resize():
    img_test.shape = img.shape
    assert resize(img, percent=20) == img_test.shape/5, 'image is too big'

#test the contour of the image
def test_display_contour():
    assert test_display_contour(cnts, img) == os.path.exists(dirname), "the directory doesn't exist"
    
