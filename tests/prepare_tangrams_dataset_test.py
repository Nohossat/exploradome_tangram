"""
@author: Renata
"""

import pytest
from ..prepare_tangrams_dataset import *

#test if get images is in jpg format

def test_get_files():
    dicto = [('bateau', '../data/tangrams\\bateau.jpg'), ('bateau', '../data/tangrams\\bateau_1.jpg'), ('bol', '../data/tangrams\\bol.jpg'), ('chat', '../data/tangrams\\chat.jpg'), ('coeur', '../data/tangrams\\coeur.jpg'), ('cygne', '../data/tangrams\\cygne.jpg'), ('lapin', '../data/tangrams\\lapin.jpg'), ('marteau', '../data/tangrams\\marteau.jpg'), ('montagne', '../data/tangrams\\montagne.jpg'), ('pont', '../data/tangrams\\pont.jpg'), ('tortue', '../data/tangrams\\tortue.jpg')]
    assert get_files(directory='../data/tangrams') == dicto

def test_save_moments():
# test moments and hu_moments into CSV files and return them as Pandas dataframe
    images = get_files(directory='../data/tangrams') 
    humoments, moments = save_moments(images, directory = '../data')
    assert isinstance(humoments, pd.core.frame.DataFrame), 'Humoments are not correct'
    assert isinstance(moments, pd.core.frame.DataFrame), 'Moments are not correct'
    assert os.path.exists('../data/hu_moments.csv')
    assert os.path.exists('../data/moments.csv')
