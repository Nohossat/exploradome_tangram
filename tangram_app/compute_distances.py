import numpy as np
import os
import cv2
import imutils
import math
import pandas as pd
from .processing import preprocess_img


def detect_forme(cnts, image):
    '''
    This function detects all triangle squart and quadilater shape in the image
    author: @Gautier
    ================================
    Parameter:
     @cnts: contours that previous function returns
     @image: image that we have preprocessed
    '''
    cnts_output = []

    for cnt in cnts:
        perimetre = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimetre, True)

        area = cv2.contourArea(cnt)
        img_area = image.shape[0] * image.shape[1]
        # print("image area", img_area)
        if area / img_area > 0.0001:
            # for triangle, if the shape has 3 angles
            if len(approx) == 3:
                cnts_output.append(cnt)
            # for quadrilater, if the shape has 4 angles
            elif len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)

                ratio = w / float(h)
                # if the ratio is correct, we take this shape as a quadrilater
                if (ratio >= 0.33 and ratio <= 3):
                    cnts_output.append(cnt)

    return cnts_output


def merge_tangram(image, contours):
    '''
    This functions puts all shapes in a black background image, it deletes all others shapes
    author: @Gautier
    ================================
    Parameter:
     @image: the original image
     @contours: all contours that returns last function
    '''
    # Create a new black image
    out_image = np.zeros(image.shape, image.dtype)

    # Put all our contours formes with white color
    cv2.drawContours(out_image, contours, -1, (255, 255, 255), thickness=-1)
    return out_image, contours


def distance_formes(contours):
    '''
    In the first step this functions separates all shapes in 5 shapes differents: small triangle, midlle triangle, big triangle, square and 
    In the second step it calculates all distance between 2 shapes
    author: @Gautier
    ==============================
    Parameter:
     @contours: the cv2.contours that returns last function
    '''
    formes = {"triangle": [], "squart": [], "parallelo": []}

    for cnt in contours:

        perimetre = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimetre, True)
        # if the shape has 3 angles we consider this shape is a triangle
        if len(approx) == 3:
            formes["triangle"].append(cnt)
         # if the shape has 4 angles we consider this shape is a quadrilateral
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)

            ratio = w / float(h)
            # if the quadrilateral has this ratio between height and width, we consider this is a square
            if ratio >= 0.9 and ratio <= 1.1:
                formes["squart"].append(cnt)
            # if the quadrilateral has this ratio between height and width, we consider this is a parallelogram
            elif (ratio >= 0.3 and ratio <= 3.3):
                formes["parallelo"].append(cnt)

    # dictionnay to take barycenters of all shapes
    centers = {"smallTriangle": [], "middleTriangle": [],
               "bigTriangle": [], "squart": [], "parallelo": []}
    # dictionnay to take perimeters of all shapes
    perimeters = {"smallTriangle": [], "middleTriangle": [],
                  "bigTriangle": [], "squart": [], "parallelo": []}

    # we detecte the size of trianlge, we compare the area of triangle to the unique square's, if we detect we have just one parallelogram
    if len(formes['triangle']) > 0:
        if len(formes["squart"]) == 1:
            areaSquart = cv2.contourArea(formes["squart"][0])
            for triangle in formes['triangle']:
                triangle_perimeter = cv2.arcLength(triangle, True)
                M = cv2.moments(triangle)
                triangle_center = (
                    int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                areaTriangle = cv2.contourArea(triangle)
                rapport = areaTriangle / areaSquart
                if rapport < 0.5:
                    centers['smallTriangle'].append(triangle_center)
                    perimeters['smallTriangle'].append(triangle_perimeter)
                elif rapport < 1.15:
                    centers['middleTriangle'].append(triangle_center)
                    perimeters['smallTriangle'].append(triangle_perimeter)
                else:
                    centers['bigTriangle'].append(triangle_center)
                    perimeters['smallTriangle'].append(triangle_perimeter)

        # we detecte the size of trianlge, we compare the area of triangle to the unique square's, if we detect we have just one parallelogram
        elif len(formes["parallelo"]) == 1:
            areaSquart = cv2.contourArea(formes["parallelo"][0])

            for triangle in formes['triangle']:

                triangle_perimeter = cv2.arcLength(triangle, True)
                M = cv2.moments(triangle)
                triangle_center = (
                    int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                areaTriangle = cv2.contourArea(triangle)
                rapport = areaTriangle / areaSquart
                if rapport < 0.6:
                    centers['smallTriangle'].append(triangle_center)
                    perimeters['smallTriangle'].append(triangle_perimeter)
                elif rapport < 1.5:
                    centers['middleTriangle'].append(triangle_center)
                    perimeters['smallTriangle'].append(triangle_perimeter)
                else:
                    centers['bigTriangle'].append(triangle_center)
                    perimeters['smallTriangle'].append(triangle_perimeter)

        # else we compare between the bigger triangle and the smaller triangle
        else:

            triangleArea = [cv2.contourArea(triangle)
                            for triangle in formes['triangle']]
            min_triangle_area = min(triangleArea)

            max_triangle_area = max(triangleArea)

            # Si c'est plus grand triangle avec plus petit triangle
            if max_triangle_area / min_triangle_area > 5:

                for triangle in formes['triangle']:
                    triangle_perimeter = cv2.arcLength(triangle, True)
                    M = cv2.moments(triangle)
                    triangle_center = (
                        int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                    if max_triangle_area / cv2.contourArea(triangle) > 5:
                        centers['smallTriangle'].append(triangle_center)
                        perimeters['smallTriangle'].append(triangle_perimeter)
                    elif max_triangle_area / cv2.contourArea(triangle) > 2:
                        centers['middleTriangle'].append(triangle_center)
                        perimeters['smallTriangle'].append(triangle_perimeter)
                    else:
                        centers['bigTriangle'].append(triangle_center)
                        perimeters['smallTriangle'].append(triangle_perimeter)

        # for case of square
        for squart in formes['squart']:
            squart_perimeter = cv2.arcLength(squart, True)
            M = cv2.moments(squart)
            squart_center = (int(M["m10"] / M["m00"]),
                             int(M["m01"] / M["m00"]))
            centers['squart'].append(squart_center)
            perimeters['squart'].append(squart_perimeter)

        # for case of parallelogram
        for parallelo in formes['parallelo']:
            parallelo_perimeter = cv2.arcLength(parallelo, True)
            M = cv2.moments(parallelo)
            parallelo_center = (
                int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            centers['parallelo'].append(parallelo_center)
            perimeters['parallelo'].append(parallelo_perimeter)

    return centers, perimeters


def ratio_distance(centers, perimeters):
    '''
    This function calculate all ratios of distances between shapes with a shape's side  
    ==================================================
    Parameters
     @centers: an array of centers,  a center point is a tuple of 2 numbers: abscissa, ordinate, and a 
     @perimeters: a dictionnay of perimeters, it has keys of all shape's name
    '''

    distances = {}

    for forme1, centers1 in centers.items():
        for forme2, centers2 in centers.items():
            inx = []
            for i in range(len(centers1)):
                for j in range(len(centers2)):
                    if forme1 + str(i) != forme2 + str(j):
                        if (forme1 + "_" + str(i + 1) + "-" + forme2 + "_" + str(j + 1) not in list(distances)) and (
                                forme2 + "_" + str(j + 1) + "-" + forme1 + "_" + str(i + 1) not in list(distances)):
                            inx.append(str(i) + str(j))
                            x1, y1 = centers1[i]
                            x2, y2 = centers2[j]
                            absolute_distance = round(
                                math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2)), 2)

                            if len(perimeters['squart']) > 0:
                                squart_perimeter = perimeters['squart'][0]
                                relative_distance = round(
                                    absolute_distance / squart_perimeter * (4 * math.sqrt(1 / 8)), 2)
                                distances[forme1 + "-" + forme2 + "_" +
                                          str(i + 1) + str(j + 1)] = relative_distance

                            elif len(perimeters['parallelo']) > 0:
                                parallelo_perimeter = perimeters['parallelo'][0]
                                relative_distance = round(
                                    absolute_distance / parallelo_perimeter * (2 * math.sqrt(1 / 8) + 1), 2)
                                distances[forme1 + "-" + forme2 + "_" +
                                          str(i + 1) + str(j + 1)] = relative_distance

                            elif len(perimeters['smallTriangle']) > 0:
                                smallTriangle_perimeter = perimeters['smallTriangle'][0]
                                relative_distance = round(
                                    absolute_distance / smallTriangle_perimeter * (2 * math.sqrt(1 / 8) + 1 / 2), 2)
                                distances[forme1 + "-" + forme2 + "_" +
                                          str(i + 1) + str(j + 1)] = relative_distance

                            elif len(perimeters['middleTriangle']) > 0:
                                middleTriangle_perimeter = perimeters['middleTriangle'][0]
                                relative_distance = round(
                                    absolute_distance / middleTriangle_perimeter * (1 + math.sqrt(1 / 2)), 2)
                                distances[forme1 + "-" + forme2 + "_" +
                                          str(i + 1) + str(j + 1)] = relative_distance

                            elif len(perimeters['bigTriangle']) > 0:
                                bigTriangle_perimeter = perimeters['bigTriangle'][0]
                                relative_distance = round(
                                    absolute_distance / bigTriangle_perimeter * (1 + 2 * math.sqrt(1 / 2)), 2)
                                distances[forme1 + "-" + forme2 + "_" +
                                          str(i + 1) + str(j + 1)] = relative_distance

                            else:
                                distances[forme1 + "-" + forme2 + "_" + str(i + 1) + str(
                                    j + 1)] = 1
    return distances


def sorted_distances(distances):
    '''
    # This function sort all distances by form  in accending
    '''
    data_distances = {"smallTriangle-smallTriangle": [], "smallTriangle-middleTriangle": [],
                      "smallTriangle-bigTriangle": [], "smallTriangle-squart": [], "smallTriangle-parallelo": [],
                      "middleTriangle-bigTriangle": [], "middleTriangle-squart": [], "middleTriangle-parallelo": [],
                      "bigTriangle-bigTriangle": [], "bigTriangle-squart": [], "bigTriangle-parallelo": [],
                      "squart-parallelo": []
                      }

    # keys of all shapes
    keys = ["smallTriangle", "middleTriangle",
            "bigTriangle", "squart", "parallelo"]

    liste = []

    # sorted all distances in all shapes default array
    for i in range(len(keys)):
        for j in range(i):
            liste.append(keys[j] + "-" + keys[i])
    liste.append("smallTriangle-smallTriangle")
    liste.append("bigTriangle-bigTriangle")

    for key, value in distances.items():

        for title in liste:
            if title in key:
                data_distances[title].append(value)
        for title in liste:
            data_distances[title] = sorted(data_distances[title])

    if len(data_distances['smallTriangle-smallTriangle']) > 1:
        del data_distances['smallTriangle-smallTriangle'][1]

    if len(data_distances['bigTriangle-bigTriangle']) > 1:
        del data_distances['bigTriangle-bigTriangle'][1]

    data_sortered = {}

    # Ordered all shapes in a dictionnay with sigle index to a distace value, not a index for an array of distances
    for key, value in data_distances.items():
        if len(value) > 1:
            for i in range(len(value)):
                data_sortered[key + str(i + 1)] = value[i]
        elif len(value) == 1:
            data_sortered[key+str(1)] = value[0]
    return data_sortered


def img_to_sorted_dists(img_cv):
    '''
    It takes a img_cv in input, and returns a dictionnay of shape with distance ordered
    '''
    cnts, img = preprocess_img(img_cv)
    cnts_form = detect_forme(cnts, img)

    image, contours = merge_tangram(img, cnts_form)
    centers, perimeters = distance_formes(contours)
    distances = ratio_distance(centers, perimeters)
    sorted_dists = sorted_distances(distances)
    return sorted_dists


def create_all_types_distances(link):
    '''
    Create a dataframe by 12 images with our img_to_sorted_dists function and save this as a CSV file.
    '''

    images = ['bateau.jpg', 'bol.jpg', 'chat.jpg', 'coeur.jpg', 'cygne.jpg', 'lapin.jpg', 'maison.JPG', 'marteau.jpg',
              'montagne.jpg', 'pont.jpg', 'renard.JPG', 'tortue.jpg']
    data = pd.DataFrame(
        columns=['smallTriangle-smallTriangle1', 'smallTriangle-middleTriangle1', 'smallTriangle-middleTriangle2',
                 'smallTriangle-bigTriangle1', 'smallTriangle-bigTriangle2', 'smallTriangle-bigTriangle3',
                 'smallTriangle-bigTriangle4', 'smallTriangle-squart1', 'smallTriangle-squart2',
                 'smallTriangle-parallelo1', 'smallTriangle-parallelo2', 'middleTriangle-bigTriangle1',
                 'middleTriangle-bigTriangle2', 'middleTriangle-squart1', 'middleTriangle-parallelo1',
                 'bigTriangle-bigTriangle1', 'bigTriangle-squart1', 'bigTriangle-squart2', 'bigTriangle-parallelo1',
                 'bigTriangle-parallelo2', 'squart-parallelo1', 'classe'])

    for im in images:
        img_cv = cv2.imread('data/tangrams/' + im)
        sorted_dists = img_to_sorted_dists(img_cv)
        classe = im.split('.')[0]
        sorted_dists['classe'] = classe
        data = data.append(sorted_dists, ignore_index=True)

    data.to_csv(link, sep=";")


def mse_distances(data, sorted_dists):
    '''
    This function returns a list of rmse that we have between our tangram with all classes 
    author: @Gautier
    ============================================
    Parmeters:
     @data: the dataframe containing all distances of shapes of all classes
     @sorted_dists: the dictionnay containing our tangram's shape distances
    '''
    mses = []
    for i in range(data.shape[0]):
        ligne = data.iloc[i]
        mses.append(round(math.sqrt(sum(
            [pow(ligne[index] - sorted_dists[index], 2) for index, _ in sorted_dists.items() if
             index in ligne.keys()])), 3))
    return mses


def propablite_of_classes(rmse):
    pass


def img_to_sorted_dists(img_cv, side, prepro=False):
    '''
    This function take in input a image and return a dictionnay of shape distances
    author: @Gautier, @Nohossat
    ================================
    Parameters:
     @img_cv: input image
     @side: it take the position of the table, if side is left we take just the left side of table, right we take the right side
    '''

    if prepro:
        cnts, cropped_img = prepro(img_cv, side=side)
    else:
        cnts, cropped_img = preprocess_img(img_cv, side=side)

    cnts_form = detect_forme(cnts, cropped_img)
    image, contours = merge_tangram(cropped_img, cnts_form)

    centers, perimeters = distance_formes(contours)
    distances = ratio_distance(centers, perimeters)
    sorted_dists = sorted_distances(distances)
    return sorted_dists  # we get the proba
