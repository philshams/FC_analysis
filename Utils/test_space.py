from nptdms import TdmsFile
import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np

from nptdms import tdmsinfo

# Variables
filepath = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\data\\Camera_LDR-default-35046654-video.tdms'


skip_data_points = 4094
width = 1100
height = 550
frame_size = width*height

# Open file
f = open(filepath, 'rb')
# !!!!!!!
t = TdmsFile(f)




vid_data = t.__dict__['objects']["/'cam0'/'data'"].data

frame = vid_data[0:frame_size]

pdf = t.as_dataframe()
pdf = pdf.values
one_frame = pdf[0:frame_size]

# This actually has lots of 0s in it which we need to get rid of
zeros = np.where(one_frame == 0)[0]
one_real_frame = pdf[0:frame_size+len(zeros)]
one_real_frame = np.delete(one_real_frame, zeros)
plt.figure()
plt.imshow(one_real_frame.reshape(550, 1100))


real_frame = np.reshape(frame, (550,  1100))

plt.figure()
plt.imshow(real_frame)
# !!!!!


f_size = os.path.getsize(filepath)


# Show data structure
data = []
for i in range(frame_size*2):
    f.seek(i)
    px = ord(f.read(1))
    data.append(px)

plt.figure()
plt.plot(data)
# plt.show()

zeros = np.where(np.array(data) == 0)[0]

x = np.linspace(0, 2*frame_size, 2*frame_size)
ones = np.ones((len(zeros)))*100

plt.figure()
plt.plot(np.array(data))
# plt.plot(zeros, ones)
plt.plot(np.diff(np.array(data)))
plt.plot(np.diff(np.array(data),2))

diff = np.diff(np.array(data))
th = 70
frame_starts = np.where(diff > th)[0]
frame_ends = np.where(diff < -th)[0]

skip_starts = np.where(frame_starts<skip_data_points)[0][-1]
skip_ends = np.where(frame_ends<skip_data_points)[0][-1]

frame_starts = frame_starts[skip_starts+1:]
frame_ends = frame_ends[skip_ends+1:]

for idx, start in enumerate(np.nditer(frame_starts[0:550])):
    print(idx, start)

    end = frame_ends[idx]
    length = end-start
    print(length)


f.seek(skip_data_points+1)
for h in range(height):
    # Load the num of pixels that corresponds to a row
    row = f.read(height-1)






# Loop over frames
num_frames = 2
skip_frames = 1
try:
    for frame_n in range(num_frames):
        # if not frame_n % skip_frames == 0:
        #     continue
        print(frame_n)

        px_count = 0
        frame = np.zeros(frame_size)
        for d in range(4095, frame_size*2):
            if px_count == frame_size:
                break
            print(f.read(1))
            f.seek(d*frame_n)
            px = ord(f.read(1))
            if px != 0:
                frame[px_count] = px
                px_count += 1

        real_frame = np.reshape(frame, (1100, 550))

        # if frame_n % skip_frames == 0:
        plt.figure()
        plt.imshow(real_frame, cmap='Greys')
    plt.show()
except:

    plt.show()
