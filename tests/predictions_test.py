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
 
from tangram_app.processing import preprocess_img, preprocess_img_2
from tangram_app.predictions import get_predictions, get_predictions_with_distances

def test_get_predictions():
    img = 'data/test_images/bateau_4_right.jpg'

    # get size to analyze from image path
    pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
    result = pattern.search(img)
    side = result.group(2)

    img_cv = cv2.imread(img)

    # get predictions
    probas = get_predictions(img_cv, preprocess_img, side)
    assert isinstance(probas, pd.core.frame.DataFrame), 'Predictions should be dataframe'
    assert probas.loc[0, 'target'] == 'bateau', 'Predictions should be bateau'

def test_get_predictions_with_distances():
    # test the probabilities to belong to each class in descending order
    img = 'data/test_images/bateau_4_right.jpg'

    # get size to analyze from image path
    pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
    result = pattern.search(img)
    side = result.group(2)

    img_cv = cv2.imread(img)
    probas = get_predictions_with_distances(img_cv, side, preprocess_img_2)
    assert isinstance(probas, pd.core.frame.DataFrame), 'Predictions should be dataframe'
    assert probas.loc[0, 'target'] == 'bateau', 'Predictions should be bateau'
