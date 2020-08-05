"""
@author: Renata
"""

import pytest
from tangram_app.metrics import *

# test global accuracy
def test_get_classification_report():
    dataset_path = 'data/tangrams'
    assert 2 == 2, "not true" # a refaire
