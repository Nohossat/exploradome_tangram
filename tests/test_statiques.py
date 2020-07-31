from exploradome_tangram.app import *
import re

def get_classification_report():
    """
    from a set of images, get global accuracy, precision, recall
    """

    image = "lapin_1.jpg"

    pattern = re.compile(r"[a-zA-Z]+")
    label = pattern.match(image).group()

    predictions = tangram_game(image = 'data/tangrams/cygne.jpg', crop=False)
    print(predictions.loc[0, 'target']) # get first prediction


if __name__ == "__main__":
    get_classification_report()

