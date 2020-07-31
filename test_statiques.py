from app import tangram_game
from prepare_tangrams_dataset import get_files
import re

def get_classification_report():
    """
    from a set of images, get global accuracy, precision, recall
    """
    total_predictions = 0
    good_predictions = 0
    #google_drive_pics = "https://drive.google.com/drive/folders/1SEroxKkziBIJ1HAYNptcdIoSH1iJCz7U?usp=sharing"

    images = get_files() # get dataset images

    for label, img_path in images:
        predictions = tangram_game(image = image_path, crop=False)
        print(predictions.loc[0, 'target']) # get first prediction

        if predictions.loc[0, 'target'] == 'cygne':
            good_predictions += 1

        total_predictions += 1

    return f'{good_predictions / total_predictions * 100} %'




if __name__ == "__main__":
    print(get_classification_report())

