from Utils.imports import *
from Utils.maths import line_smoother
import math
import numpy as np

audio, video = [], []

with open('D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis\\det_times.txt', 'r') as f:
    for line in f:
        # print(line)
        n = int(line.split(' ')[1].split('_')[0])
        r = int(line.split(' ')[-1])
        if n < 100 : video.append(r)
        else: audio.append(r)


print("""
Audio
    median - {}
    mean - {}
    std  - {}
    
Video
    median - {}
    mean - {}
    std  - {}
""".format(np.median(audio), np.mean(audio), np.std(audio),
           np.median(video), np.median(video), np.std(video)))

