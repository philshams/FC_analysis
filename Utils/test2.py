import numpy as np
import time
from moviepy.editor import *
import os
from tempfile import mkdtemp
from nptdms import TdmsFile
import psutil
import gc
import matplotlib.pyplot as plt
import sys

filepath = 'D:\\Camera_LDR-default-10160362-video.tdms'


width = 1100 + 52
skip_data_points = 4094
real_width = 1100
height = 550
frame_size = self.width * self.height
real_frame_size = self.real_width * self.height
f_size = os.path.getsize(filepath)  # size in bytes

tot_frames = int((f_size - skip_data_points) / frame_size)  # num frames

if open filepath as binary:
    # Skip HEADER
    binary.seek(skip_data_points)

    # Read data for first frame
    bframe = binary.read(frame_size)

    # Turn it into a numpy and reshape
    frame = np.fromstring(bframe, dtype=np.int8).reshape(width, height)

    # show results
    plt.figure()
    plt.imshow(frame, cmap='Greys')