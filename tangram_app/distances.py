import numpy as np
import numpy as np
import os
import cv2
import imutils
import math
import pandas as pd

from tangram_app.processing import preprocess_img_2

# caclate distances for Hu Moments
def dist_humoment(hu1,hu2):
    """
    return sum of euclidienne distance,sum of squart of pow difference
    author: @Gautier
    """
    distance = np.linalg.norm(hu1-hu2)
    return distance

def detect_forme(cnts, image):
    '''
    This function detects all triangle and rectangle shapes in the image
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
        if area / img_area > 0.005:
             # for triangle, if the shape has 3 angles
            if len(approx) == 3:
                cnts_output.append(cnt)
                
            # for quadrilater, if the shape has 4 angles
            elif len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)

                ratio = w / float(h)
                # if the ratio is correct, we take this shape as a quadrilateral
                if (ratio >= 0.33 and ratio <= 3):
                    cnts_output.append(cnt)

    return cnts_output

def distance_formes(contours):
    '''
    In the first step this function separates all shapes in 5 different shapes: small triangle, midlle triangle, big triangle, square and parallelogram
    In the second step it calculates the perimeters and centers of all shapes, and remove duplicate ones
    author: @Gautier
    ==============================
    Parameter:
     @contours: the cv2.contours that returns last function

    Returns : centers and perimeters of all shapes
    '''
    
    formes = {"triangle": [], "squart": [], "parallelo": []}
    
    for cnt in contours:
        perimetre = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.05 * perimetre, True)
        # if the shape has 3 angles we consider this shape is a triangle
        if len(approx) == 3:
            formes["triangle"].append(cnt)
        # if the shape has 4 angles we consider this shape is a quadrilateral
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ratio = w / float(h)
            
             # if the quadrilateral has this ratio between height and width, we consider this is a square
            # if ratio >= 0.9 and ratio <= 1.1:
            if ratio >= 0.7 and ratio <= 1.3:
                formes["squart"].append(cnt)

            # if the quadrilateral has this ratio between height and width, we consider this is a parallelogram
            # elif (ratio >= 0.3 and ratio <= 3.3):
            elif (ratio >= 0.2 and ratio <= 5):
                formes["parallelo"].append(cnt)

    # Remove isolated shapes
    formes = delete_isolate_formes3(formes,200)

    # dictionnay to take barycenters of all shapes
    centers = {"smallTriangle": [], "middleTriangle": [],
               "bigTriangle": [], "squart": [], "parallelo": []}
    
    # dictionnay to take perimeters of all shapes
    perimeters = {"smallTriangle": [], "middleTriangle": [],
                  "bigTriangle": [], "squart": [], "parallelo": []}


    if len(formes['triangle']) > 0:
         # we detecte the size of trianlge, we compare the area of triangle to the unique square's, if we detect we have just one parallelogram
        
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
                    perimeters['middleTriangle'].append(triangle_perimeter)
                else:
                    centers['bigTriangle'].append(triangle_center)
                    perimeters['bigTriangle'].append(triangle_perimeter)

            
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
                    perimeters['middleTriangle'].append(triangle_perimeter)
                else:
                    centers['bigTriangle'].append(triangle_center)
                    perimeters['bigTriangle'].append(triangle_perimeter)

        else:
            # else we compare between the bigger triangle and the smaller triangle
            triangleArea = [cv2.contourArea(triangle)  for triangle in formes['triangle']]
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
                        perimeters['middleTriangle'].append(triangle_perimeter)
                        
                    else:
                        centers['bigTriangle'].append(triangle_center)
                        perimeters['bigTriangle'].append(triangle_perimeter)
                        
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

    # Remove duplicate shapes
    centers2 = {}
    perimeters2 = {}

    for key,values in centers.items():
        centers2[key] = []
        perimeters2[key] = []
        if(len(values)) > 0:
            centers2[key].append(centers[key][0])
            perimeters2[key].append(perimeters[key][0])
            for i in range(1,len(values)):
                x1, y1 = values[i]
                isDistanceBigger = True
                for j in range(i):
                    x2, y2 = values[j]
                    distance = round(math.sqrt(pow(x1 - x2, 2)+pow(y1 - y2, 2)),2)
                    if distance < 20:
                        isDistanceBigger = False
                if isDistanceBigger:
                    centers2[key].append(centers[key][i])
                    perimeters2[key].append(perimeters[key][i])

    return centers2, perimeters2

def delete_isolate_formes3(formes, threshold=10):
    '''
    Delete all shapes if the most distance between the shape to all shapes is bigger than the threshold
    Author: @Gautier
    =================================================
    Parameter:
     @formes: the dictionnay containing keys of shapes and the array containing the contour
     @threshold: the threshold of distance between shapes
    '''
    mindistances = {}
    # Save all min distances by shape to a dictionary
    for keys1, values1 in formes.items():
        mindistances[keys1] = []
        for i in range(len(values1)):
            M1 = cv2.moments(values1[i])
            center_i_x, center_i_y = int(M1["m10"] / M1["m00"]),int(M1["m01"] / M1["m00"])
            min_distance = 99999999
            for keys2, values2 in formes.items():
                for j in range(len(values2)):
                    # if keys1 != keys2 and i != j:
                    M2 = cv2.moments(values2[j])
                    center_j_x, center_j_y = int(M2["m10"] / M2["m00"]),int(M2["m01"] / M2["m00"])
                    distance = math.sqrt(pow(center_i_x-center_j_x,2)+pow(center_i_y-center_j_y,2))
                    if distance < min_distance and distance > 0:
                        min_distance = distance
            mindistances[keys1].append(min_distance)
            
    # we take all shapes if this shape is smaller than threshold
    forme_output = {}
    for keys, values in mindistances.items():
        forme_output[keys] = []
        for i in range(len(values)):
            if mindistances[keys][i] < threshold:
                forme_output[keys].append(formes[keys][i])

    return forme_output        

def ratio_distance(centers, perimeters):
    '''
    This function calculate all ratios of distances between shapes with a shape's side  
    Author: @Gautier
    ==================================================
    Parameters
     @centers: an array of centers,  a center point is a tuple of 2 numbers: absciss, ordinate, and a 
     @perimeters: a dictionnay of perimeters, it has keys of all shape's name
    
    Returns : distances of all shapes between each other
    '''
    distances = {}

    for forme1, centers1 in centers.items():
        inx = []
        for i in range(len(centers1)):
            for forme2, centers2 in centers.items():
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
                                    j + 1)] = 0
    return distances

def sorted_distances(distances):
    '''
    This function sorted the distances between same relationship of two shapes
    Author: @Gautier
    =======================================
    Parameter: 
     @distances: dictionnay of distances between all two shapes, like this {"smallTriangle-middleTriangle_11":[0.5], "smallTriangle-middleTriangle_21":[0.4]}
     
    Output: a dictionary of distances between all two shapes but ordered by distance and rename keys by the nomber of the same relation, example: {"smallTriangle-middleTriangle1":[0.4], "smallTriangle-middleTriangle2":[0.5]}
    '''
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
    '''
    Create a csv file of all distances of shapes of all ours classes 
    author: @Gautier
    ==========================
    parameter:
        link: directory to save the csv file
    '''
    images = ['bateau.jpg', 'bol.jpg', 'chat.jpg', 'coeur.jpg', 'cygne.jpg', 'lapin.jpg', 'maison.jpg', 'marteau.jpg',
              'montagne.jpg', 'pont.jpg', 'renard.jpg', 'tortue.jpg']

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
        cnts, img = preprocess_img_2(img_cv, side=None)
        cnts_forms = detect_forme(cnts, img)
        centers, perimeters = distance_formes(cnts_forms)
        distances = ratio_distance(centers, perimeters)
        sorted_dists = sorted_distances(distances)
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
