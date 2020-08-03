"""
@author: Renata
"""

import pytest
from ..prepare_tangrams_dataset import *

#test if get images is in jpg format

def test_get_files():
    dicto = {'bateau.jpg', 'bol.jpg', 'chat.jpg', 'coeur.jpg', 'cygne.jpg', 'lapin.jpg', 'maison.jpg', 'martaeu.jpg', 'montagne.jpg', 'pont.jpg', 'renard.jpg', 'tortue.jpg'
    }
    assert get_files() == dicto

def test_save_moments():
# test moments and hu_moments into CSV files and return them as Pandas dataframe
    assert save_moments(images) == hu_moments, 'Moments are not correct'
