from .processing import preprocess_img, display_contour
from .moments import find_moments
import pandas as pd
import os
import re
import cv2

DATA_PATH = 'data/'

def get_files(directory = DATA_PATH + '/tangrams'):
    """
    get a dict with the image_name and the path of all images in tangrams folder
    author : @Nohossat
    """
    images = []
    assert os.path.exists(directory), "the directory doesn't exist"

    for folder, sub_folders, files in os.walk(directory):
        for file in files:
            filename, file_extension = os.path.splitext(file) # we just want the filename to save the path
            file_path = os.path.join(folder, file)
            if file.endswith((".jpg", ".png")) and not file.startswith('frame'):
                pattern = re.compile(r"[a-zA-Z]+") # in case there is any number or underscore in the name
                label = pattern.match(filename).group()
                images.append((label, file_path))

    return images

def save_moments(images, directory):
    """
    compute moments / hu moments for all images in our dataset

    Parameters : 

    images : dict with images names and paths

    ========
    Return : save moments and hu_moments into CSV files and return them as Pandas dataframe

    author : @Nohossat
    """

    hu_moments = []
    moments = []

    for image_name, image_path in images:
        img_cv = cv2.imread(image_path)

        cnts, img = preprocess_img(img_cv, crop=False)
        # display_contour(cnts, img) - for testing purposes

        hu_moments.append(find_moments(cnts, image_name))
        moments.append(find_moments(cnts, image_name, hu_moment=False))

        hu_moments_df = pd.DataFrame(hu_moments)
        hu_moments_df.to_csv(directory +'/hu_moments.csv', index=False)

        moments_df = pd.DataFrame(moments)
        moments_df.to_csv(directory + '/moments.csv', index=False)

    return hu_moments_df, moments_df
    

if __name__ == "__main__":
    path = "/Users/nohossat/Documents/exploradome_videos/TangrIAm dataset"
    images = get_files(directory=path)
    print(len(images))
    # print(save_moments(images, directory = DATA_PATH))