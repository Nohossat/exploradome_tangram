from processing import preprocess_img, display_contour
from moments import find_moments
import pandas as pd
import os
import cv2

DATA_PATH = 'data/'

def get_files():
    """
    get a dict with the image_name and the path of all images in tangrams folder
    author : @Nohossat
    """
    images = {}
    dirname = DATA_PATH + '/tangrams'
    assert os.path.exists(dirname), "the directory doesn't exist"

    for file in os.listdir(dirname):
        filename, file_extension = os.path.splitext(file) # we just want the filename to save the path
        if file.endswith(".jpg"):
            images[filename] = os.path.join(dirname, file)
    return images

def save_moments(images):
    """
    compute moments / hu moments for all images in our dataset

    =========

    Parameters : 

    images : dict with images names and paths

    ========

    Return : save moments and hu_moments into CSV files and return them as Pandas dataframe

    author : @Nohossat
    """

    hu_moments = []
    moments = []

    for image_name, image_path in images.items():
        img_cv = cv2.imread(image_path)

        cnts, img = preprocess_img(img_cv, split=False)
        # display_contour(cnts, img) - for testing purposes

        hu_moments.append(find_moments(cnts, image_name))
        moments.append(find_moments(cnts, image_name, hu_moment=False))

        hu_moments_df = pd.DataFrame(hu_moments)
        hu_moments_df.to_csv(DATA_PATH +'/hu_moments.csv', index=False)

        moments_df = pd.DataFrame(moments)
        moments_df.to_csv(DATA_PATH + '/moments.csv', index=False)

    return hu_moments, moments
    

if __name__ == "__main__":
    images = get_files()
    save_moments(images)