
"""
Created on Tue Jul 28 15:11:06 2020

@author: Renata
"""
import pytest

from tangram_app.distances import *
from tangram_app.processing import *
import os
import re

def test_dist_humoment():
   assert dist_humoment(0.026322267294741648, 0.0027161374010481088) == 0.02360612989369354, 'distance_humoment not working'


def test_detect_forme():
    img = 'data/test_images/bateau_4_right.jpg'
    
    # get side
    pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
    result = pattern.search(img)
    side = result.group(2)

    img_cv = cv2.imread(img)
    cnts, img = preprocess_img_2(img_cv, side=side)
    cnts_forms = detect_forme(cnts, img)
    assert isinstance(cnts_forms, list), "the cnts_forms format isn't correct"

def test_distance_formes():
    img = 'data/test_images/bateau_4_right.jpg'
    
    # get side
    pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
    result = pattern.search(img)
    side = result.group(2)

    img_cv = cv2.imread(img)
    cnts, img = preprocess_img_2(img_cv, side=side)
    cnts_forms = detect_forme(cnts, img)
    centers, perimeters = distance_formes(cnts_forms)
    print(centers)
    assert isinstance(centers, dict), "centers should be a dict"
    assert isinstance(perimeters, dict), "centers should be a dict"

def test_ratio_distance():
    img = 'data/test_images/bateau_4_right.jpg'
    
    # get side
    pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
    result = pattern.search(img)
    side = result.group(2)

    img_cv = cv2.imread(img)
    cnts, img = preprocess_img_2(img_cv, side=side)
    cnts_forms = detect_forme(cnts, img)
    centers, perimeters = distance_formes(cnts_forms)
    distances = ratio_distance(centers, perimeters)
    assert isinstance(distances, dict), "the distances should be stored inside a dict"
    

def test_sorted_distances():
    img = 'data/test_images/bateau_4_right.jpg'
    
    # get side
    pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
    result = pattern.search(img)
    side = result.group(2)

    img_cv = cv2.imread(img)
    cnts, img = preprocess_img_2(img_cv, side=side)
    cnts_forms = detect_forme(cnts, img)
    centers, perimeters = distance_formes(cnts_forms)
    distances = ratio_distance(centers, perimeters)
    sorted_dists = sorted_distances(distances)
    assert isinstance(sorted_dists, dict), "the sorted distances should be stored inside a dict"
    

def test_create_all_types_distances():
    create_all_types_distances("tests/data/data.csv")
    assert os.path.exists('tests/data/data.csv'), "the data csv with the distances should exist"

def test_mse_distances():
    data = pd.read_csv("data/tangram_properties/data.csv", sep=";")
    
    img = 'data/test_images/bateau_4_right.jpg'
    # get side
    pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
    result = pattern.search(img)
    side = result.group(2)

    # preprocessing img
    img_cv = cv2.imread(img)
    cnts, img = preprocess_img_2(img_cv, side=side)

    # compute distances between its shapes
    cnts_forms = detect_forme(cnts, img)
    centers, perimeters = distance_formes(cnts_forms)
    distances = ratio_distance(centers, perimeters)
    sorted_dists = sorted_distances(distances)
    
    # get mses
    mses = mse_distances(data, sorted_dists)
    assert isinstance(mses, list), "The mses should be stored in list"
    assert len(mses) == 12, "the MSES list should have a length of 12"
