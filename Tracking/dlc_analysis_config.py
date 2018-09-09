# coding: utf-8

"""
Video analysis using a trained network, based on code by
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu
"""

# The normal DLC setting file has much more options, we dont need them as we specify the file to be loaded in the
# DLC config .yml file (./Tracking/Configs)

# Filename and path to videos
videofolder = '../videos/'
videotype='.avi'  # type of videos to analyze

# Cropping
# ROI dimensions / bounding box (only used if cropping == True)
# x1,y1 indicates the top left corner and
# x2,y2 is the lower right corner of the croped region.
cropping = False
x1 = 0
x2 = 640
y1 = 277
y2 = 624
