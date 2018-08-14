import pandas as pd
import os
from moviepy.editor import VideoFileClip
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

def vid_deeplabcut_probability_map():
    # Where to find the data
    datapath = 'C:\\Users\\Federico\\Documents\\GitHub\\DeepLabCut\\videos'

    dataname = '180528_CA2763_0DeepCut_resnet50_mazeJuly14shuffle1_50000.h5'
    scampname = '180528_CA2763_0DeepCut_resnet50_mazeJuly14shuffle1_50000scmap.npy'
    videoname = '180528_CA2763_0.avi'

    # Load the data
    Dataframe = pd.read_hdf(os.path.join(datapath, dataname))
    Scmap = np.load(os.path.join(datapath, scampname))

    clip = VideoFileClip(os.path.join(datapath, videoname))
    fps = clip.fps

    # Extract useful info
    bodyparts2plot = list(np.unique(Dataframe.columns.get_level_values(1)))
    scorer=np.unique(Dataframe.columns.get_level_values(0))[0]

    # Other set up stuff
    n_images = 10
    random_frames = np.random.randint(0, clip.duration*fps, n_images)
    img_counter = 0

    probmaps_order = [0, 2, 3, 1, 4, 5]

    # loop over frames
    for idx,row in Dataframe.iterrows():
        if not img_counter >= n_images:
            if idx in random_frames:
                f, axarr = plt.subplots(2, 3)
                axarr = axarr.flatten()
                for bpindex, bp in enumerate(bodyparts2plot):
                    probmap = Scmap[idx, :, :, probmaps_order[bpindex]]

                    frame_s = idx * (1/fps)
                    frame = clip.get_frame(frame_s)

                    axarr[bpindex].set(title=bp)
                    axarr[bpindex].axis('off')
                    axarr[bpindex].imshow(frame)
                    axarr[bpindex].imshow(probmap, cmap='inferno', alpha=0.45, extent=[0, frame.shape[1], frame.shape[0], 0])
        else:
            break

    plt.show()
    a = 1

# Start
vid_deeplabcut_probability_map()

