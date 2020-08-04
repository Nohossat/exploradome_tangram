"""
@author: Renata
"""

import pytest
from ..app import *

def test_tangram_game():
   # test the probabilities of the image / frame 
   path = "../data/tangrams/cygne.jpg"
   probality = tangram_game(crop=False,image=path)
   assert isinstance(probalitity, pd.core.frame.DataFrame), 'Predictions should be dataframe'
   assert probalitity.loc[0, 'target'] == 'cygne', 'Predictions should be bateau'
   