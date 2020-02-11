import cv2
import numpy as np
import argparse
import pdb

def callback(x):
    pass



ilowB = 0
ihighB = 255
ilowG = 0
ihighG = 255
ilowR = 0
ihighR = 255

# create trackbars for color change
cv2.createTrackbar('lowH','image',ilowB,255,callback)
cv2.createTrackbar('highH','image',ihighB,255,callback)

cv2.createTrackbar('lowS','image',ilowG,255,callback)
cv2.createTrackbar('highS','image',ihighG,255,callback)

cv2.createTrackbar('lowV','image',ilowR,255,callback)
cv2.createTrackbar('highV','image',ihighR,255,callback)


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help = 'path to the image')
args = vars(ap.parse_args())
#llamamos asi #python Color_Detector_1.py --image 1.jpg
#img = cv2.imread(args['image'])
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(args['image'])
cv2.namedWindow('image')

while(True):
    # grab the frame
    ret, frame = cap.read()
    ret2, frame2 = cap.read()
    #img = cv2.imread(args['image'])
     

    # get trackbar positions
    ilowB = cv2.getTrackbarPos('lowB', 'image')
    ihighB = cv2.getTrackbarPos('highB', 'image')
    ilowG = cv2.getTrackbarPos('lowG', 'image')
    ihighG = cv2.getTrackbarPos('highG', 'image')
    ilowR = cv2.getTrackbarPos('lowR', 'image')
    ihighR = cv2.getTrackbarPos('highR', 'image')

  
    lower_bgr = np.array([ilowB, ilowG, ilowR])
    higher_bgr = np.array([ihighB, ihighG, ihighR])
    mask = cv2.inRange(frame2, lower_bgr, higher_bgr)
    #pdb.set_trace()
    frame = cv2.bitwise_and(frame, frame, mask=mask)

    # show thresholded image
    cv2.imshow('image',frame)
    k = cv2.waitKey(1000) & 0xFF # large wait time to remove freezing
    if k == 113 or k == 27:
        break