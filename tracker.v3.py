import cv2
import numpy as np
import scipy.cluster.hierarchy as hcluster

### Parameters that the user can modify:

# Video filenames:

inVideo = "enemy-approaches-ext.mp4"
outVideo = "outvid.avi"  # must be in avi format

# Background-subtraction parameters

subHistory = 32
subThreshold = 500

# Feature detection and clustering parameters

orbFeatures = 1000

maxClusterDistance = 30
minClusterPoints = 16
minROIArea = 200

# Tracker parameters

maxTrackers       = 10
trackerWinSize    = (15,15)
trackerMaxLevel   = 4
trackerCriteria   = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)

### End user section

########################################################################################################################

class LKTracker:
    def __init__(self,oldPoints,winSize,maxLevel,criteria):
        self.oldPoints = oldPoints
        self.lkParams  = dict(winSize = winSize,maxLevel = maxLevel,criteria = criteria)
        self.killCount = 0

    def update(self,oldGrayFrame,newGrayFrame):
        newPoints, status, error = cv2.calcOpticalFlowPyrLK(oldGrayFrame, newGrayFrame,self.oldPoints,None,**self.lkParams)
        self.oldPoints = newPoints

        return newPoints, status, error

    def challenge(self, roiList):
        matched = False
        x, y = self.oldPoints.ravel()
        for (x1, y1, x2, y2) in roiList:
            if x1 <= x <= x2 and y1 <= y <= y2:
                matched = True

        if matched:
            self.killCount += 1
        else:
            self.killCount -= 1

    def isInROI(self,x1,y1,x2,y2):
        x, y = self.oldPoints.ravel()
        if x1 <= x <= x2 and y1 <= y <= y2:
            inside = True
        else:
            inside = False

        return inside

def sharpenImage(image):
    tempImage = cv2.GaussianBlur(image, (0, 0), 3)
    tempImage = cv2.addWeighted(image, 1.5, tempImage, -0.5, 0)

    return tempImage

def findFeatures(grayImage):
    imageMask = subtractor.apply(grayImage)
    imageMask = cv2.GaussianBlur(imageMask, (5, 5), 0)

    keyPoints, descriptors = orb.detectAndCompute(imageMask, None)

    return keyPoints, descriptors

def findROI(keyPoints):
    roiList = []
    pointList = [k.pt for k in keyPoints]

    if len(pointList) >= 2:
        clusters = hcluster.fclusterdata(pointList, maxClusterDistance, criterion="distance")
        groups = [np.where(clusters == c_id)[0] for c_id in np.unique(clusters)]

        for group in groups:
            if len(group) >= minClusterPoints:
                groupPoints = [pointList[c] for c in group]
                groupPoints = np.array(groupPoints, dtype=np.float32)
                x, y, w, h = cv2.boundingRect(groupPoints)
                area = w * h
                if area >= minROIArea:
                    roiList.append((x, y, x + w, y + h))

    return roiList

########################################################################################################################

cap = cv2.VideoCapture(inVideo)
_,frame = cap.read()
height, width, c = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'XVID')
outvid = cv2.VideoWriter(outVideo,fourcc,30,(width,height))

orb = cv2.ORB_create(nfeatures=orbFeatures)

subtractor = cv2.createBackgroundSubtractorKNN(history=subHistory,dist2Threshold=subThreshold)

font = cv2.FONT_HERSHEY_SIMPLEX

trackerList = []

oldGray = np.array([[]])

while True:
    # Process frame
    _, frame = cap.read()
    frame = sharpenImage(frame)
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # identify features
    keyPoints, descriptors = findFeatures(grayFrame)
    frame = cv2.drawKeypoints(frame, keyPoints, None)

    # create regions of interest
    roiList = findROI(keyPoints)
    for (x1,y1,x2,y2) in roiList:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # cull useless trackers
    newTrackerList = []
    for tracker in trackerList:
        tracker.challenge(roiList)
        if tracker.killCount > 0:
            newTrackerList.append(tracker)
    trackerList = newTrackerList[:]

    # create new trackers
    for (x1, y1, x2, y2) in roiList:
        if len(trackerList) < maxTrackers:
            tracked = False
            for tracker in trackerList:
                if tracker.isInROI(x1,y1,x2,y2):
                    tracked = True
            if not tracked:
                point = np.array([[(x1+x2)// 2, (y1+y2)// 2]], dtype=np.float32)
                trackerList.append(LKTracker(point, trackerWinSize, trackerMaxLevel, trackerCriteria))

    #print("now tracking %d points" % (len(trackerList)))

    # update tracker positions
    if len(oldGray) > 0:
        for tracker in trackerList:
            newPoints, status, error = tracker.update(oldGray, grayFrame)

            # draw tracker info
            x, y = newPoints.ravel()
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            msg = str(tracker.killCount)
            frame = cv2.putText(frame, msg, (int(x+2), y), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Frame", frame)
    oldGray = grayFrame.copy()
    outvid.write(frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
