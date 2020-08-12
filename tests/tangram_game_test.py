"""
@author: Renata
"""

import pytest
from tangram_app.tangram_game import *
from tangram_app.processing import *
from tangram_app.predictions import *

def test_tangram_game():
   # test the probabilities of the image / frame 
   path = "data/test_images/bateau_4_right.jpg"
   probability = tangram_game(image=path, prepro=preprocess_img_2, pred_func=get_predictions_with_distances)
   assert isinstance(probability, pd.core.frame.DataFrame), 'Predictions should be dataframe'
   assert probability.loc[0, 'target'] == 'bateau', 'Predictions should be bateau'
  