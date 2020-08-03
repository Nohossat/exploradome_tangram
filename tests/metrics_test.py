"""
@author: Renata
"""

import pytest
from ..metrics import *

# test global accuracy

def test_get_classification_report():
    dataset_path = '../data/tangrams'
    assert get_classification_report_pics(dataset_path=None) == get_files(directory = dataset_path), 'Path is not good'
