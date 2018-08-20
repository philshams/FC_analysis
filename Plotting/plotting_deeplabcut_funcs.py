import pandas as pd
import os
from moviepy.editor import VideoFileClip
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import copy

# Where to find the data
datapath = 'C:\\Users\\Federico\\Documents\\GitHub\\DeepLabCut\\videos'
session = 'VTE'
model = 'DeepCut_resnet50_fearcondJuly14shuffle1_50000'

dataname = session+model+'.h5'
scampname = session+model+'scmap.npy'
videoname = session+'.avi'


def general_setup():
    # Load the data
    Dataframe = pd.read_hdf(os.path.join(datapath, dataname))

    clip = VideoFileClip(os.path.join(datapath, videoname))
    fps = clip.fps

    # Extract useful info
    bodyparts2plot = list(np.unique(Dataframe.columns.get_level_values(1)))
    scorer=np.unique(Dataframe.columns.get_level_values(0))[0]

    return Dataframe, clip, fps, bodyparts2plot, scorer


class LineSet():
    def __init__(self, ax):
        self.head_1 = ax.plot([0, 1], [0, 1], color='green')
        self.head_2 = ax.plot([0, 1], [0, 1], color='green')
        self.head_3 = ax.plot([0, 1], [0, 1], color='green')
        self.body_1 = ax.plot([0, 1], [0, 1], color='lightcoral')
        self.body_2 = ax.plot([0, 1], [0, 1], color='lightcoral')
        self.body_3 = ax.plot([0, 1], [0, 1], color='lightcoral')
        self.tail = ax.plot([0, 1], [0, 1], color='orange')


def viz_deeplabcut_probability_map():
    Dataframe, clip, fps, bodyparts2plot, scorer = general_setup()
    Scmap = np.load(os.path.join(datapath, scampname))

    # Set up stuff
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


def viz_rotated_mouse_skeleton():
    def update_lines(line_set, data_set, alpha=1):
        for idx,(line_name, line) in enumerate(sorted(line_set.__dict__.items())):
            line[0].set_xdata(data_set[0][idx])
            line[0].set_ydata(data_set[1][idx])
            line[0].set_alpha(alpha)

    def get_bp_coords(dataframe, scorer, bp, center=True):
        if center:
            X = dataframe[scorer][bp]['x'].values[index]-dataframe[scorer]['body']['x'].values[index]
            Y = dataframe[scorer][bp]['y'].values[index]-dataframe[scorer]['body']['y'].values[index]
        else:
            X = dataframe[scorer][bp]['x'].values[index]
            Y = dataframe[scorer][bp]['y'].values[index]
        return [X, Y]

    #################################################################
    pcutoff = 0.001
    dotsize = 7

    Dataframe, clip, fps, bodyparts2plot, scorer = general_setup()

    # set up plotting
    plt.ion()
    f, ax = plt.subplots()
    ax.set(xlim=[-100, 100], ylim=[-100, 100])



    LineSet(ax)
    past_lineset = [LineSet(ax), LineSet(ax), LineSet(ax),
                    LineSet(ax), LineSet(ax), LineSet(ax), LineSet(ax), LineSet(ax),
                    LineSet(ax), LineSet(ax)]
    trace_length = len(past_lineset)
    past_dataset = []
    alphavalues = np.linspace(0.1, 1, trace_length)

    # plot
    for index, row in Dataframe.iterrows():
        frame_s = index * (1 / fps)
        frame = clip.get_frame(frame_s)\
        # ax.imshow(frame)

        l_ear, r_ear, snout, body, tail1, tail2 = [], [], [], [], [], []
        for bpindex, bp in enumerate(bodyparts2plot):
            if Dataframe[scorer][bp]['likelihood'].values[index] > pcutoff:

                if bp == 'L_ear' or bp == 'R_ear' or bp == 'snout':
                    dot_color = 'green'
                    if bp == 'L_ear':
                        l_ear = get_bp_coords(Dataframe, scorer, bp)
                    elif bp == 'R_ear':
                        r_ear = get_bp_coords(Dataframe, scorer, bp)
                    elif bp == 'snout':
                        snout = get_bp_coords(Dataframe, scorer, bp)

                elif bp == 'body':
                    dot_color = 'lightcoral'
                    body = get_bp_coords(Dataframe, scorer, bp)

                else:
                    dot_color = 'orange'
                    if bp == 'tail1':
                        tail1 = get_bp_coords(Dataframe, scorer, bp)
                    else:
                        tail2 = get_bp_coords(Dataframe, scorer, bp)


            # Plot lines between points:
        try:
            curr_dataset_x = [[l_ear[0], r_ear[0]], [l_ear[0], snout[0]], [r_ear[0], snout[0]],
                              [r_ear[0], body[0]], [l_ear[0], body[0]], [tail1[0], body[0]], [tail1[0], tail2[0]]]
            curr_dataset_y = [[l_ear[1], r_ear[1]], [l_ear[1], snout[1]], [r_ear[1], snout[1]],
                              [r_ear[1], body[1]], [l_ear[1], body[1]], [tail1[1], body[1]], [tail1[1], tail2[1]]]
            curr_dataset = [curr_dataset_x, curr_dataset_y]

            if len(past_dataset)<trace_length:
                past_dataset.append(curr_dataset)
            else:
                past_dataset.remove(past_dataset[0])
                past_dataset.append(curr_dataset)

            for data_index, dataset in enumerate(past_dataset):
                update_lines(past_lineset[data_index], dataset, alpha=alphavalues[data_index])

            plt.pause(0.01)

        except:
             pass



# Start
viz_deeplabcut_probability_map()
# viz_rotated_mouse_skeleton()
