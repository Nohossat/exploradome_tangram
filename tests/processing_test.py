
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
    img = '../data/tangrams/bateau.jpg'
    img_cv = cv2.imread(img)
    result = preprocess_img(img_cv, side=None, crop = False)
    assert isinstance(result, tuple)
    assert isinstance(result[0], list)
    assert isinstance(result[1], np.ndarray)

#test cv image and resizes it. 
def test_resize():
    img = '../data/tangrams/bateau.jpg'
    img_cv = cv2.imread(img)
    image_resize = resize(img_cv)
    assert image_resize.shape[0:2] == tuple(int(dim / 5) for dim in img_cv.shape[0:2]), 'image is too big'
 
