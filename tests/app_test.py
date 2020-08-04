"""
@author: Renata
"""

import pytest
from ..app import *

def test_tangram_game():
   # test the probabilities of the image / frame 
   path = "../data/tangrams/cygne.jpg"
   probality = tangram_game(hu_moments_dataset='../data/hu_moments.csv', crop=False,image=path)
   assert isinstance(probality, pd.core.frame.DataFrame), 'Predictions should be dataframe'
   assert probality.loc[0, 'target'] == 'cygne', 'Predictions should be bateau'
   