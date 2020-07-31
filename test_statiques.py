import pytest
from app import *
import re

def get_classification_report():
    """
    from a set of images, get global accuracy, precision, recall
    """
    total_predictions = 0
    good_predictions = 0

    image = "lapin_1.jpg"

    pattern = re.compile(r"[a-zA-Z]+")
    label = pattern.match(image).group()
    image_path = 'data/tangrams/cygne.jpg'
    predictions = tangram_game(image = image_path, crop=False)
    print(predictions.loc[0, 'target']) # get first prediction

    if predictions.loc[0, 'target'] == 'cygne':
        good_predictions += 1

    total_predictions += 1

    return f'{good_predictions / total_predictions * 100} %'




if __name__ == "__main__":
    print(get_classification_report())

