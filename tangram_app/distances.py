import numpy as np
import numpy as np
import os
import cv2
import imutils
import math
import pandas as pd


# different ways to calculate distances between Hu Moments
def dist_humoment1(hu1,hu2):
    """
    return sum absolute difference of inverse of humoment value
    author: @Gautier
    """
    distance =  np.sum(abs(1/hu1-1/hu2))
    return distance

def dist_humoment2(hu1,hu2):
    """
    return the sum of manatan distance, sum of absolute difference
    author: @Gautier
    """
    distance =  np.sum(abs(hu1-hu2))
    return distance

def dist_humoment3(hu1,hu2):
    """
    return sum of absolute difference  divide absolute of second humoment
    author: @Gautier
    """
    distance =  np.sum(abs(hu1-hu2)/abs(hu2))
    return distance

def dist_humoment4(hu1,hu2):
    """
    return sum of euclidienne distance,sum of squart of pow difference
    author: @Gautier
    """
    distance = np.linalg.norm(hu1-hu2)
    return distance


# see what functions we really use in the following ones @Gautier
def detect_forme(cnts, image):
    cnts_output = []

    for cnt in cnts:
        perimetre = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimetre, True)

        area = cv2.contourArea(cnt)
        img_area = image.shape[0] * image.shape[1]
        # print("image area", img_area)
        if area / img_area > 0.0001:
            # for triangle
            if len(approx) == 3:
                cnts_output.append(cnt)
            # for quadrilater
            elif len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)

                ratio = w / float(h)
                # if ratio >= 0.95 and ratio <= 1.05:
                # cnts_output.append(cnt)

                # elif(ratio >= 0.3 and ratio < 0.95) or (ratio > 1.05 and ratio <= 3):
                if (ratio >= 0.33 and ratio <= 3):
                    cnts_output.append(cnt)

    return cnts_output

def merge_tangram(image, contours):
    # Create a new black image
    out_image = np.zeros(image.shape, image.dtype)

    # Put all our contours formes with white color
    cv2.drawContours(out_image, contours, -1, (255, 255, 255), thickness=-1)
    return out_image, contours

def distance_formes(contours):
    formes = {"triangle": [], "squart": [], "parallelo": []}

    for cnt in contours:

        perimetre = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimetre, True)
        if len(approx) == 3:
            formes["triangle"].append(cnt)

        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)

            ratio = w / float(h)

            if ratio >= 0.9 and ratio <= 1.1:
                formes["squart"].append(cnt)

            elif (ratio >= 0.3 and ratio <= 3.3):
                formes["parallelo"].append(cnt)

    centers = {"smallTriangle": [], "middleTriangle": [],
               "bigTriangle": [], "squart": [], "parallelo": []}
    perimeters = {"smallTriangle": [], "middleTriangle": [],
                  "bigTriangle": [], "squart": [], "parallelo": []}

    # Comparer la taille des triangle à parallélograme unique
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

        # Comparer la taille des triangle à parallélograme unique
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

        for squart in formes['squart']:
            squart_perimeter = cv2.arcLength(squart, True)
            M = cv2.moments(squart)
            squart_center = (int(M["m10"] / M["m00"]),
                             int(M["m01"] / M["m00"]))
            centers['squart'].append(squart_center)
            perimeters['squart'].append(squart_perimeter)

        for parallelo in formes['parallelo']:
            parallelo_perimeter = cv2.arcLength(parallelo, True)
            M = cv2.moments(parallelo)
            parallelo_center = (
                int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            centers['parallelo'].append(parallelo_center)
            perimeters['parallelo'].append(parallelo_perimeter)

    return centers, perimeters

def unique_centers(centers):
    centres_unique_form = {}

    for x, y in centers.items():
        center = [0, 0]
        for ele in y:
            i, j = ele
            center[0] += i
            center[1] += j
        centres_unique_form[x] = (
            int(center[0] / len(y)), int(center[1] / len(y)))
    return centres_unique_form

def distance_forme(centers):
    distances = {}
    for forme1, center1 in centers.items():
        for forme2, center2 in centers.items():
            if forme1 != forme2:
                x1, y1 = center1
                x2, y2 = center2
                distances[forme1 + "-" +
                          forme2] = round(math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2)), 2)
    return distances

def ratio_distance(centers, perimeters):
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
    data_distances = {"smallTriangle-smallTriangle": [], "smallTriangle-middleTriangle": [],
                      "smallTriangle-bigTriangle": [], "smallTriangle-squart": [], "smallTriangle-parallelo": [],
                      "middleTriangle-bigTriangle": [], "middleTriangle-squart": [], "middleTriangle-parallelo": [],
                      "bigTriangle-bigTriangle": [], "bigTriangle-squart": [], "bigTriangle-parallelo": [],
                      "squart-parallelo": []
                      }

    keys = ["smallTriangle", "middleTriangle",
            "bigTriangle", "squart", "parallelo"]

    liste = []
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

    for key, value in data_distances.items():
        if len(value) > 1:
            for i in range(len(value)):
                data_sortered[key + str(i + 1)] = value[i]
        elif len(value) == 1:
            data_sortered[key+str(1)] = value[0]
    return data_sortered

def create_all_types_distances(link):
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
    mses = []
    for i in range(data.shape[0]):
        ligne = data.iloc[i]
        mses.append(round(math.sqrt(sum(
            [pow(ligne[index] - sorted_dists[index], 2) for index, _ in sorted_dists.items() if
             index in ligne.keys()])), 3))
    return mses
