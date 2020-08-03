from app import tangram_game, tangram_game_live_test
from prepare_tangrams_dataset import get_files
import re
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# test statiques
def get_classification_report_pics(dataset_path=None):
    """
    from a set of images, get global accuracy, precision, recall
    """
    total_predictions = 0
    y_true = []
    y_pred = []
    classes = ["bateau", "bol", "chat", "coeur", "cygne", "lapin", "maison", "marteau", "montagne", "pont", "renard", "tortue"]

    if dataset_path is None:
        images = get_files() # get images in data/tangrams
    else :
        images = get_files(directory=dataset_path)

    for label, img_path in images: 
        predictions = tangram_game(image = img_path, crop=False)

        # if predictions is None:
            # continue

        y_true.append(label)
        y_pred.append(predictions.loc[0, 'target'])

    conf_matrix = confusion_matrix(y_true, y_pred, labels=classes)
    report = classification_report(y_true, y_pred, target_names=classes)

    sns.heatmap(conf_matrix, annot = True, xticklabels=classes, yticklabels=classes)
    plt.show()

    return report


# test dynamiques
def get_classification_report_videos(video_folder):
    videos = []
    correct_predictions = 0
    sides = ["right", "left", "right", "right", "left", "right", 
            "left", "right", "left", "right", "left", "right", "left", "right"]

    for folder, sub_folders, files in os.walk(video_folder):
        for file in files:
            filename, file_extension = os.path.splitext(file) # we just want the filename to save the path
            file_path = os.path.join(folder, file)

            if file.endswith(".mov"):
                pattern = re.compile(r"[a-zA-Z]+") # in case there is any number or underscore in the name
                label = pattern.match(filename).group()
                videos.append((label, file_path))
    
    i = 0
    for label, video_path in videos:
        prediction = tangram_game_live_test(crop=True, side=sides[i], video=video_path)
        i += 1

        if prediction == label:
            correct_predictions += 1

        print(f'- label : {label}\n- prediction: {prediction}\n- correct_predictions: {correct_predictions}\n========\n')


if __name__ == "__main__":

    # static testing
    path = "/Users/nohossat/Documents/exploradome_videos/TangrIAm dataset"
    path2 = "/Users/nohossat/Documents/exploradome_videos/photos"
    print(get_classification_report_pics(dataset_path=path2))

    # live testing
    # get_classification_report_videos(video_folder="/Users/nohossat/Documents/exploradome_videos/videos/")

    

