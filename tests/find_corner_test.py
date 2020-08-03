"""
@author: Renata
"""

import pytest
from ..find_corner import *

#test visualize on an image the corner point in red
def test_get_nb_corners():
    assert get_nb_corners(img) == len(corners), 'Nombre of corners is incorrect'