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

from tangram_app.utils import *

def test_get_files():
    assert len(get_files(directory='data/tangrams')) == 12, 'Images must be 12'

#test visualize on an image the corner point in red
def test_get_nb_corners():
    img = 'data/tangrams/bateau_4_right.jpg'
    img_cv = cv2.imread(img)
    assert get_nb_corners(img_cv) == 74, 'Nombre of corners is incorrect'
