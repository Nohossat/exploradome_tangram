
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
import re
 
from tangram_app.moments import *
from tangram_app.processing import preprocess_img
from tangram_app.utils import get_files


def test_find_humoments():
     img = 'data/test_images/bateau_4_right.jpg'
     pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
     result = pattern.search(img)
     side = result.group(2)
     img_cv = cv2.imread(img)
     cnts, img = preprocess_img(img_cv, side=side)
     humoments = find_moments(cnts, filename = 'bateau', hu_moment = True)
     assert humoments.shape == (8,) , "Humoments aren't correct"


def test_find_moments():
     img = 'data/test_images/bateau_4_right.jpg'
     pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
     result = pattern.search(img)
     side = result.group(2)
     img_cv = cv2.imread(img)
     cnts, img = preprocess_img(img_cv, side=side)
     moments = find_moments(cnts, filename = 'bateau', hu_moment = False)
     assert len(moments.keys()) == 25 , "Moments aren't correct"


def test_save_moments():
    images = get_files(directory='data/tangrams')
    humoments, moments = save_moments(images, directory = 'tests/data')
    assert isinstance(humoments, pd.core.frame.DataFrame), 'Humoments are not correct'
    assert isinstance(moments, pd.core.frame.DataFrame), 'Moments are not correct'
    assert os.path.exists('tests/data/hu_moments.csv')
    assert os.path.exists('tests/data/moments.csv')
