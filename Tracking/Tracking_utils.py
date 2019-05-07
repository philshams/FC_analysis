import cv2
import numpy as np
import math
import os
import shutil
import pandas as pd
import datetime
from Utils.loadsave_funcs import load_yaml
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

