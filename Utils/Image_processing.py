import cv2
import numpy as np


def process_background(background):
    cv2.startWindowThread()
    # cv2.namedWindow('Raw background')
    # cv2.namedWindow('edges')

    if len(background.shape) == 3:
        gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    else:
        gray = background

    # Blur
    blur = cv2.blur(gray, (15, 15))

    # Get maze edges
    edges = cv2.Canny(blur, 25, 30)

    # Get user to define Shelter ROI, Threat area ROI and the line between them or arena ROI
    print('\n\nPlease mark NEST ROI')
    shelter_roi = cv2.selectROI(gray, fromCenter=False)
    print('\n\nPlease mark THREAT ROI')
    threat_roi = cv2.selectROI(gray, fromCenter=False)

    print('\n\nPlease mark ARENA FLOOR ROI or SMALL BRIDGE ROI')
    var_roi = cv2.selectROI(gray, fromCenter=False)

    rois = {'Shelter':shelter_roi, 'Threat':threat_roi, 'Task':var_roi}
    return edges, rois




    """ Work in progress: extract maze structure from video """
    # lower = np.array(100)
    # upper = np.array(255)
    # thresh = cv2.inRange(blur, lower, upper)
    #
    # kernel = np.ones((3,3),np.uint8)
    # opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    #
    #
    # sure_bg = cv2.dilate(opening,kernel,iterations=3)
    #
    # dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    # ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
    #
    # _, cnts, _ = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL,
    #                               cv2.CHAIN_APPROX_NONE)
    #
    #
    #
    #
    # cv2.imshow('opening', opening)
    #
    # cv2.imshow('bg', sure_bg)
    #
    # cv2.imshow('fg', sure_fg)
    #
    # cv2.waitKey(0)
    #
    # cv2.moveWindow('opening', 0, 0)
    # cv2.moveWindow('bg', 1000, 0)
    # cv2.moveWindow('fg', 1000, 1000)


