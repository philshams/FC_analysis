import numpy as np
import os
from tempfile import mkdtemp
from nptdms import TdmsFile
import psutil
import gc
import time
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

import cv2


def show_mem_stats():
        giga = 1073741824
        stats = psutil.virtual_memory()
        print("""Total memory:           {} GB
          available:    {} GB
               free:    {} GB
               used:    {} GB
            percent:    {}%
        """.format(round(stats.total/giga, 2), round(stats.available/giga, 2),
                   round(stats.free/giga, 2), round(stats.used/giga, 2), stats.percent))
        return stats.available


def write_clip(limits):
    vidname = 'output_{}.mp4'.format(limits[0])
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    videowriter = cv2.VideoWriter(os.path.join(tempdir, vidname), fourcc,
                                  120, (1100, 550), False)

    for framen in range(limits[0], limits[1]):
        videowriter.write(tdms[framen])
    videowriter.release()


start_time = time.clock()

filepath = 'D:\\Camera_LDR-default-10160362-video.tdms'
filepath2 = 'D:\\test_mantis_v6-default-102077900-cam2.tdms'
#tempdir = mkdtemp(dir='D:\\')
tempdir = 'D:\\tmpmyzce9ia'

width = 1100 + 52
skip_data_points = 4094
real_width = 1100
height = 550
frame_size =width * height
real_frame_size = real_width * height
f_size = os.path.getsize(filepath)  # size in bytes
tot_frames = int((f_size - skip_data_points) / frame_size)  # num frames

print('Total number of frames {}'.format(tot_frames))

print('Opening binary')
bfile = open(filepath, 'rb')
show_mem_stats()

print('Opening mmap tdms')
tdms = TdmsFile(bfile, memmap_dir=tempdir)  # open tdms binary file as a memmapped object
show_mem_stats()

print('Extracting data')
tdms = tdms.__dict__['objects']["/'cam0'/'data'"].data.reshape((tot_frames, height, width), order='C')
show_mem_stats()

print('Got data, cleaning up cached memory')
gc.collect()

print('Cleaning up data')
tdms = tdms[:, :, :real_width]

print('Ready to plot')
free_mem = show_mem_stats()


print('Make video')
n_clips = 4
steps = np.linspace(0, tot_frames, n_clips+1).astype(int)
step = steps[1]
steps2 = np.asarray([x+step for x in steps])
limits = [s for s in zip(steps, steps2)][:-1]


pool = ThreadPool(n_clips)
results = pool.map(write_clip, limits)

print('It took {}s to process a file of {} bytes'.format(time.clock()-start_time, f_size))


a = 1
