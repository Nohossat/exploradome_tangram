"""
@author: Renata
"""

import pytest
from tangram_app.tangram_game import *

def test_tangram_game():
   # test the probabilities of the image / frame 
   path = "data/tangrams/bateau_4_right.jpg"
   probability = tangram_game(image=path)
   assert isinstance(probability, pd.core.frame.DataFrame), 'Predictions should be dataframe'
   assert probability.loc[0, 'target'] == 'bateau', 'Predictions should be bateau'
   
def test_display_predictions():
   pass