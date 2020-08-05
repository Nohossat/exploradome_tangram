"""
@author: Renata
"""

import pytest
from tangram_app.prepare_tangrams_dataset import *

#test if get images is in jpg format

def test_get_files():
    assert len(get_files(directory='data/tangrams')) == 12, 'Images must be 12'

def test_save_moments():
# test moments and hu_moments into CSV files and return them as Pandas dataframe
    images = get_files(directory='data/tangrams') # we have to change the images in this one
    humoments, moments = save_moments(images, directory = 'tests/data/')
    assert isinstance(humoments, pd.core.frame.DataFrame), 'Humoments are not correct'
    assert isinstance(moments, pd.core.frame.DataFrame), 'Moments are not correct'
    assert os.path.exists('data/hu_moments.csv')
    assert os.path.exists('data/moments.csv')
