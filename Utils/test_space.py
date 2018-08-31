from nptdms import TdmsFile
import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from  moviepy.editor import *
import imageio
import scipy.misc

# Variables
# filepath = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\data\\Camera_LDR-default-35046654-video.tdms'
filepath = 'Z:\\rig_behaviour2\\Camera_LDR-default-10160362-video.tdms'
memmap_path = 'C:\\Users\\Federico\\Documents\\GitHub\\FC_analysis\\Utils'

skip_data_points = 4094
width = 1100+52
real_width = 1100
height = 550
frame_size = width*height
real_frame_size = real_width*height

start = time.clock()
# Open file
f = open(filepath, 'rb')
f_size = os.path.getsize(filepath)

tot_frames = int((f_size-skip_data_points)/frame_size)

# !!!!!!!
t = TdmsFile(f, memmap_dir = memmap_path)
vid_data = t.__dict__['objects']["/'cam0'/'data'"].data

frames_list = []
all_frames = np.zeros((height, real_width, tot_frames))
for framen in range(tot_frames):
    print('Processing frame {}'.format(framen))
    linear_frame = vid_data[frame_size*framen:frame_size*(framen+1)]

    square_frame = linear_frame.reshape(height, width)
    square_frame = square_frame[:height, :real_width]

    frame = np.zeros((height, real_width, 3))
    frame[:] = np.nan
    frame[:, :, 0] = square_frame
    frame[:, :, 1] = square_frame
    frame[:, :, 2] = square_frame

    frames_list.append(frame)


print(time.clock()-start)



clip = ImageSequenceClip(frames_list, fps=30, with_mask=False)
clip.write_videofile("movie.mp4")




# Show what raw data look like
# data = []
# for i in range(frame_size*2):
#     f.seek(i)
#     px = ord(f.read(1))
#     data.append(px)
#
# plt.figure()
# plt.plot(data)
