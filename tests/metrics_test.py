"""
@author: Renata
"""

import pytest
from ..find_corner import *

# test global accuracy
def test_get_classification_report():
    assert get_classification_report(dataset_path=None) == get_files(), 'Path is not good'
