import cv2
import numpy as np
import math
from moviepy.editor import VideoFileClip
import os
import shutil
import pandas as pd
import datetime

from Tracking.dlc_loadconfig import load_config
from nnet import predict
from Utils.utils_classes import Trial
from Utils.loadsave_funcs import load_yaml
from Utils.Data_rearrange_funcs import arrange_dlc_data

from Config import track_options

########################################################################################################################
# Trial data handling


def create_trial_metadata(trial_name, stim_type, start_frame, stop_frame, video_path):
    """ organise trial metadata """
    tr_metadata = {
        'Name': trial_name,
        'Stim type': stim_type,
        'Start frame': start_frame,
        'Stop frame': stop_frame,
        'Video path': video_path,
        'Created': datetime.datetime.now().strftime("%y-%m-%d-%H-%M"),
        'Last modified': datetime.datetime.now().strftime("%y-%m-%d-%H-%M"),
        'std tracking settings': load_yaml(track_options['cfg_std']),
        'dlc tracking settings': load_yaml(track_options['cfg_dlc'])
    }
    return tr_metadata


def merge_std_dlc_trials(old_trial, new_trial):
    """ merge data from the two ways of tracking """
    try:
        if new_trial.std_tracking['x'].any():
            old_trial.std_tracking = new_trial.std_tracking
    except:
        pass
    if new_trial.dlc_tracking:
        old_trial.dlc_tracking = new_trial.dlc_tracking
    return old_trial


########################################################################################################################
# Deep Lab Cut utils functions
def save_trial_clips(clips, dlc_videos_folder):
    """ save trials clips to .avi during std tracking"""
    clips_names = []
    # save clips
    for stim_type in clips.keys():
        for clip_name, clip in clips[stim_type].items():
            clips_names.append(clip_name)
            if not clip_name + '.avi' in os.listdir(dlc_videos_folder):
                clip.write_videofile(os.path.join(dlc_videos_folder, clip_name + '.avi'), codec='png')
    return clips_names


def dlc_setupTF(options):
    dlc_config_settings = load_yaml(options['cfg_dlc'])
    cfg = load_config(dlc_config_settings['dlc_network_posecfg'])
    cfg['init_weights'] = dlc_config_settings['dlc_network_snapshot']
    scorer = dlc_config_settings['scorer']
    sess, inputs, outputs = predict.setup_pose_prediction(cfg)

    return {'scorer': scorer, 'sess': sess, 'inputs': inputs, 'outputs': outputs, 'cfg': cfg}


def dlc_retreive_data(datafolder, database, clips_l):
    """
    Get the data saved as a result of dlc_analyseVideos.py and save them as trial data into the database

    From the .h5 file get to which session the belong to and load the DLC data into the database
    Loop over all .h5 files, create a container for each session, load and rearrange .h5 into a pandas DF and then
    put that into the database
    """

    # Organise the data generated from DLC
    sessions_data = {}  # Dict holding the pandas DF for each trial in each processed session
    clips_l = [item for sublist in clips_l for item in sublist]
    sessions_to_analyse = set([int(x.split('-')[0]) for x in clips_l])
    for fname in os.listdir(datafolder):
        if '.' in fname:
            if fname.split('.')[1] == 'h5':
                # Check if the .h5 file belongs to one of the sessions we are analysing
                sessid = fname.split('-')[0]
                if not int(sessid) in sessions_to_analyse:
                    continue

                print('            ... found Pandas dataframe: {}'.format(fname))
                trial_name = fname.split('_')[0] + "_" + fname.split('_')[1]
                trial_name = trial_name.split('D')[0]
                stim_type = fname.split('-')[:-2]

                if not sessid in sessions_data.keys():
                    sessions_data[sessid] = {}  # For each session create a dict to hold the DF for each trial

                # read pandas dataframe (DLC output) and rearrange them for easier access
                Dataframe = pd.read_hdf(os.path.join(datafolder, fname))
                dlc_data = arrange_dlc_data(Dataframe)

                # create trial object
                trial = Trial()
                trial_metadata = create_trial_metadata(trial_name, None, None, None, None)
                trial.metadata = trial_metadata
                trial.dlc_tracking['Posture'] = dlc_data
                sessions_data[sessid][trial_name] = trial

    # Return the list of trials for each session
    for session_name in database.index:
        sess_num = session_name.split('_')[0]
        if sess_num in sessions_data.keys():
            session = database.loc[session_name]
            dlc_trials = sessions_data[sess_num]

            for trial_name, trial in dlc_trials.items():
                old_trial = session['Tracking'][trial_name]
                session['Tracking'][trial_name] = merge_std_dlc_trials(old_trial, trial)

    return database


def dlc_clear_folder(datafolder, keepvids=True):
    """
    After running the dlc_analyseVideos.py function and having extracted the data, clean up the folder.
    If selected can store the trial vids.
    """
    storefld = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\data\\Trial videos'
    for fname in os.listdir(datafolder):
        if '.' in fname:
            if fname.split('.')[1] == 'avi':
                if keepvids:
                    vidname = os.path.join(datafolder, fname)
                    shutil.move(vidname, os.path.join(storefld, fname))
                else:
                    os.remove(os.path.join(datafolder, fname))
            else:
                os.remove(os.path.join(datafolder, fname))


########################################################################################################################
    # STD tracking utils functions
def get_contours(bg, frame, th_scaling, num_exp_cnts=1, max_cnt_area=5000, min_cnt_area=200):
    """
    Performas a number of image processing steps to extract contours from bg subtracted frame

    :param bg: background
    :param frame: frame
    :return: list of contours
    """
    # Binary
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Bg subtraction
    delta = cv2.absdiff(bg, gray)

    # Blur
    blur = cv2.GaussianBlur(delta, (11, 11), -1)

    # Threshold
    p = np.percentile(blur[:], 99.99) * th_scaling
    thresh = cv2.threshold(blur, p, 255, cv2.THRESH_BINARY)[1]

    # Opening
    kval = 5
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kval, kval))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k)

    # Get and select contours [by size]
    _, cnts, _ = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_NONE)

    # Select only those cnts that are within the size range
    if len(cnts) > num_exp_cnts:
        pre_selected_cnts = [c for c in cnts if cv2.contourArea(c) <= max_cnt_area]
        selected_cnts = [c for c in pre_selected_cnts if cv2.contourArea(c) >= min_cnt_area]
    else:
        selected_cnts = cnts

    if len(selected_cnts) != num_exp_cnts:
        if len(cnts):
            # as a desperate measure, let's use the cnt with the biggest area
            areas = [cv2.contourArea(c) for c in cnts]
            selected_cnts = [c for c in cnts if cv2.contourArea(c) == max(areas)]
        else:
            print('Problem with contour extraction: wrong number of contours')

    return opening, selected_cnts


def drawtrace(frame, coords, col, trace_length):
    """
    Draws a colored trace over the last few coordinates of the contour
    :param frame:
    :param coords:
    :param col:
    :return:
    """
    for p in range(trace_length):
        if (p) < len(coords) - 5:
            thickness = int(np.sqrt(20 / float(p / 8 + 1)) * 1)
            cv2.line(frame, (int(coords[-p - 1][0]), int(coords[- p - 1][1])),
                     (int(coords[- p - 2][0]), int(coords[- p - 2][1])),
                     col, thickness)


def calc_distance(p1, p2):
    """ calculates a distance on a 2d euclidean space, between two points"""
    dist = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    return dist


def check_orientation_errors(orientation, angle, f, start_frame):
    if f>start_frame+2:
        prevang = orientation[0, f - 2]
        diff = angle - prevang
        if 50<abs(diff) < 250:
            if diff<0:
                angle += 180
            else:
                angle -= 180

            if angle < 0:
                angle = 360+angle
            elif angle > 360:
                angle -= 360
    return angle


def get_tail_position(arena_floor, bg, frame, mouse_f, tail_threshold_scaling):
    if arena_floor:
        cropf = frame[arena_floor[1]:arena_floor[1]+arena_floor[3],
                arena_floor[0]:arena_floor[0]+arena_floor[2], :]
        bg = bg[arena_floor[1]:arena_floor[1]+arena_floor[3],
                arena_floor[0]:arena_floor[0]+arena_floor[2]]
        mouse_f = mouse_f[arena_floor[1]:arena_floor[1] + arena_floor[3],
             arena_floor[0]:arena_floor[0] + arena_floor[2]]
    else:
        cropf = frame.copy()

    # Get thresholded image including tail
    tail_f, tail_cnt = get_contours(bg, cropf, tail_threshold_scaling)

    # Dilate the thresholded image without the tail
    cnts, kval = [], 7
    while not cnts:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kval, kval))
        dilation = cv2.dilate(mouse_f, k, iterations=6)

        # Subtract the 'mouse' image from the 'tail' image
        diff = cv2.subtract(tail_f, dilation)

        # Get centroid of tail contour
        _, cnts, _ = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_NONE)
        if cnts:
            areas = [cv2.contourArea(c) for c in cnts]
            if max(areas) < 10:
                cnts = []

        kval -= 1
        if kval < 1:
            print('something went wrong with finding tale location, self_get_tail_position')
            break

    biggest = cnts[areas.index(max(areas))]
    M = cv2.moments(biggest)
    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])
    if arena_floor:
        centeroid = (int(x+arena_floor[0]), int(y+arena_floor[1]))
    else:
        centeroid = (int(x), int(y))

    cv2.circle(frame, centeroid, 5, (0, 0, 255), -1 )

    return centeroid


def display_results(f, frame, threshold, magnif_factor, tracking):
    def enlarge_frame(frame, mag_factor):
        resized_image = cv2.resize(frame, (frame.shape[1] * mag_factor, frame.shape[0] * mag_factor),
                                   interpolation=cv2.INTER_CUBIC)
        return resized_image

    # get larger version of frame
    frame = enlarge_frame(frame, magnif_factor)
    largth = enlarge_frame(threshold, magnif_factor)

    # display frame number
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, str(f), (100, 50), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, 'vel:{}'.format(str(int(tracking.velocity[0][f - 1]))), (100, 100), font, 2, (0, 255, 0)
                , 2, cv2.LINE_AA)
    cv2.putText(frame, 'dir:{}'.format(str(int(tracking.direction[0][f - 1]))), (350, 100), font, 2,
                (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, 'orient.:{}'.format(str(int(tracking.orientation[0][f - 1]))), (600, 100), font,
                2, (0, 255, 0), 2, cv2.LINE_AA)

    # display results
    cv2.namedWindow('frame')
    cv2.namedWindow('th')
    cv2.imshow('frame', frame)
    cv2.imshow('th', largth)
    cv2.moveWindow('frame', 0, 0)
    cv2.moveWindow('th', 0, 900)


