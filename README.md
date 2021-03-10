# EnemyTrackerRevisited
Automatic object tracking in a video game using feature detection in python (revisited)

For a demo, see https://youtu.be/ah1hZgFgeyY

### Motivation
This is work in progress showing feature detection (by the ORB algorithm), point clustering, and object tracking (by the Lucas-Kanade algorithm).  It is a revised and improved version of a previous project (https://github.com/susurrus1/enemyTracker).  In the demo video (see link above) the following elements are being displayed on top of the video clip of Dark Souls 3 gameplay:

    * empty circles: detected features on each frame (using ORB algorithm)
    * 
    * blue rectangles: regions of interest around clusters of detected features
    * 
    * green dots: trackers following an object (using Lucas-Kanade algorithm)
    * 
    * green text: counter associated with each tracker

### Installation

The code was written in Python 3.7 on Windows 10. You will need to first install the necessary modules (or you can download them from pypi.org):

pip install opencv-python

pip install numpy

pip install scipy

### Usage

Once the libraries are installed, just run the main code, e.g. by executing

python tracker.v3.py

from a command line, or start it from an IDE if you use one. This will read in a raw gameplay video file. The input file name is currently set to "enemy-approaches-ext.mp4" but you will need to change it to the name of your file. The program will then run and process and display each frame. The processed frames will also be saved to the video file "outvid.avi".

There are several parameters near the top of the python code that you can modify/optimize (and you may need to depending on the game):

inVideo = name of input video file to be processed

outVideo = name of output (processed) video (must be a .avi file)

subHistory = number of frames used in background subtraction

subThreshold = threshold level for background subtraction

orbFeatures = number of features to detect

maxClusterDistance = features within this distance are clustered together

minClusterPoints = minimum number of neighboring features that can be clustered together

minROIArea = minimum area of blue rectangle to be considered a region of interest

maxTrackers = maximum allowed number of trackers at any one time

trackerWinSize = (width,height) of window searched by tracker

trackerMaxLevel = number of levels for the Lucas-Kanade algorithm

trackerCriteria = parameter used by Lucas-Kanade algorithm

The program currently gets its input from a video file, but could be modified relatively easily to read from a live game window (see, e.g., https://github.com/susurrus1/DesktopObjectDetection), although the game would have to be displayed in windowed mode in order for the python script to be able to draw on the game screen.

Credits
I found the following resources extremely useful in writing this code:

 https://pysource.com/2018/05/17/background-subtraction-opencv-3-4-with-python-3-tutorial-32/
 https://pysource.com/2018/03/21/feature-detection-sift-surf-obr-opencv-3-4-with-python-3-tutorial-25/
 https://pysource.com/2018/05/14/optical-flow-with-lucas-kanade-method-opencv-3-4-with-python-3-tutorial-31/
