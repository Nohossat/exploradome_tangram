from app import tangram_game
from prepare_tangrams_dataset import get_files
import re
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report

def get_classification_report():
    """
    from a set of images, get global accuracy, precision, recall
    """
    total_predictions = 0
    y_true = []
    y_pred = []
    classes = ["bateau", "bol", "chat", "coeur", "cygne", "lapin", "maison", "marteau", "montagne", "pont", "renard", "tortue"]

    #google_drive_pics = "https://drive.google.com/drive/folders/1SEroxKkziBIJ1HAYNptcdIoSH1iJCz7U?usp=sharing"

    images = get_files() # get dataset images

    for label, img_path in images:
        predictions = tangram_game(image = img_path, crop=False)

        y_true.append(label)
        y_pred.append(predictions.loc[0, 'target'])

    conf_matrix = confusion_matrix(y_true, y_pred, labels=classes)
    report = classification_report(y_true, y_pred, target_names=classes)

    return report


if __name__ == "__main__":
    print(get_classification_report())

