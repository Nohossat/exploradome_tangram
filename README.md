<p align="center"><img width=100% src="https://github.com/Nohossat/exploradome_tangram/blob/numpy---team-4/data/logo.jpg"></p>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Python](https://img.shields.io/badge/python-v3.8-blue.svg)
[![Build Status](https://travis-ci.org/anfederico/Clairvoyant.svg?branch=master)](https://travis-ci.org/anfederico/Clairvoyant)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)


# exploradome_tangram
#### Real-time tangram form detection from a live video stream.

The goal of the game of tangram is to replicate the outline shown on the target card using a set of seven individual polygons (triangles, square or parallelogram). Tangram can be played either by multiple teams at the same time, racing to be the first to finish the outline. 

This project's objective is to detect the similarity between the shape drawn by each player at any point in time and the twelve possible target classes. The predictions are to be in real-time from a live video feed of the game board.

The output is 12 class probabilities, served at a minimum frequency of 1 output per second.

## Installation and execution

```shell
git clone https://github.com/Nohossat/exploradome_tangram.git
cd exploradome_tangram
python -m venv venv/
source venv/Scripts/activate # Windows
source venv/bin/activate # Mac
pip install -r requirements.txt

# launch the app with an external webcam, focusing on the right side of the board 
python3 main.py --mode 1 --side right

# launch the app using a video saved locally, focusing on the left side of the board
python3 main.py --mode /videos/coeur.pm4 --side left

# launch the app with an image on the right side, focusing on the right side of the board
python3 main.py --mode /videos/coeur.jpg --side right
```

## Approach

Our approach has been to avoid the use of ML and DL techniques and leverage OpenCV's native capabilities to detect the individual shapes on the board and estimate their likeness to each of the twelve classes, using Hu Moments.

#### Image processing steps

- Splitting the video feed into two halves (left/right)
- Identifying all the edges on the board
- Keeping only the contours corresponding to regular geometric shapes (triangle, square, parallelogram).
- Computing the centroids of each geometric piece.
- Calculating the distance between each pair of pieces (resulting in 21 data points if all pieces are on the board).
- Comparing this distance scorecard to the distance scorecard of each of the 12 target outlines (using an RMSE distance metric over the 21 distance readings)
- Transforming the RMSE into a probability distribution.

## Classification results

Frame per seconds : 333 fps

### Prediction results in an ideal testing environment
The below results are valid for "final" predictions, ie the class predicted by the program once the correct form was finalized. 

<p align="center"><img width=60% src="https://github.com/Nohossat/exploradome_tangram/blob/numpy---team-4/data/metric.jpg"></p>
<p align="center"><img width=60% src="https://github.com/Nohossat/exploradome_tangram/blob/numpy---team-4/data/matrice.png"></p>

### Prediction results in a more challenging testing environment
The below results were obtained using a more challenging testing dataset (incomplete form, slight mistake in the final form, objects between the camera and the board, challenging light conditions...) and are probably closer to the expected production environment.






## Project metainformation

#### Dataset

Our approach, since it does not rely on machine/deep learning, does not require extensive training. A single example for each of the twelve target classes was sufficient to create the twelve benchmarks against which each form is compared.

Our data collection was hence designed to obtain a test set of images that would be as diverse as possible (in terms of orientation, image obstruction, light conditions...) so as to ensure maximize robustness in our software. For testing purposes, the dataset was enriched with changed lighting conditions (both brighter and darker).

#### Attempts and challenges

- A key issue has been the robustness of the preprocessing pipeline. Originally, our test dataset was too narrow to properly reflect possible light conditions. In early attempts, our preprocessing (clear isolation of the tangrams on the board), while working well in conditions similar to a subset of images similar to those used for calibration, did not generalize well other environments (different table orientation, light, shadows...). Changing the preprocessing pipeline (from a binary threshold to canny edge detection) yielded much more robust results.

- Our first approach was to use Hu Moments to compare the player's shape to each of the 12 classes, so as to leverage the Hu Moments' invariance to rotation, scale and position. However, this approach turned out to be a very poor predictor for shapes in the process of construction and overly sensitive to camera obstruction, motivating our switch to more robust data points taken from piece centroids' relative distance to each other.

