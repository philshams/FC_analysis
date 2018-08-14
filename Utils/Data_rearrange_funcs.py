import pandas as pd
import numpy as np

from Utils.utils_classes import Trial, All_trials


def restructure_trial_data(trial, start_frame, stop_frame, stim_type, idx, vid_num):
    new_trial = Trial()

    new_trial.std_tracking['x'] = trial.x[0][start_frame:stop_frame-1]
    new_trial.std_tracking['y'] = trial.y[0][start_frame:stop_frame-1]
    new_trial.std_tracking['orientation'] = trial.orientation[0][start_frame:stop_frame-1]
    new_trial.std_tracking['direction'] = trial.direction[0][start_frame:stop_frame-1]
    new_trial.std_tracking['velocity'] = trial.velocity[0][start_frame:stop_frame-1]

    new_trial.name = '{}_{}_{}'.format(stim_type, vid_num, idx)

    return new_trial


def collate_cohort_trials(db, processed_session):
    # If we don't have an All_trials column in the database, create it
    if not 'All_trials' in db.keys():
        sLen = len(db[db.keys()[0]])
        db = db.assign(All_trials=pd.Series(np.random.randn(sLen)).values)

    # Add all trials to
    trials_d = db['All_trials']['processed']

    if isinstance(trials_d, float):
        # Initialise dictionary
        trials_d = All_trials()

    for stim_type, trials_l in trials_d.__dict__.items():
        augmented_trials = []

        if isinstance(processed_session.tracking_data[stim_type], list):  # Trials tracked by traditional tracking
            for trial in processed_session.tracking_data[stim_type]:
                trial.rois = processed_session.video_data['User ROIs']
                augmented_trials.append(trial)
        else:
            for trial_name, trial in processed_session.tracking_data[stim_type].items():
                tr = {
                    'dlc data':trial,
                    'rois':processed_session.video_data['User ROIs']
                }
                augmented_trials.append(tr)

        [trials_l.append(x) for x in augmented_trials]  # store all the trials

    db['All_trials'] = db['All_trials'].replace(['processed'], trials_d, inplace=True)  # Not sure why but this line is necessary
    db['All_trials']['processed'] = trials_d

    return