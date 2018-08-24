import os
import numpy as np
from Utils.maths import line_smoother

def get_session_videofiles(session):
    trials_fld = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\data\\dlc_trialVids'

    mouse_id = session['Metadata'].mouse_id

    # Get experiment folder
    temp_fld = session['Metadata'].video_file_paths[0][0].split('\\')
    fld = ''
    for string in temp_fld[0:-2]:
        fld = os.path.join(fld, string)

    # get all subfolders that belong to this session
    subfolds, recordings = [], []
    for f in os.listdir(fld):
        if mouse_id in f:
            subfolds.append(os.path.join(fld, f))
            recordings.append(f)

    # Get videos
    vids, short_vids = [], []
    for subfold in subfolds:
        for f in os.listdir(subfold):
            if 'avi' in f:
                vids.append(os.path.join(subfold, f))
                short_vids.append(os.path.join(subfold.split('\\')[-1], f))

    # Get trial videos
    for f in os.listdir(trials_fld):
        if 'avi' in f:
            # if str(session['Metadata'].session_id) in f:
            short_vids.append(f)
            vids.append(os.path.join(trials_fld, f))

    # Prepare output dict
    output = {}
    for short, long in zip(short_vids, vids):
        output[short] = long

    return output


def get_window_forpot(elapsed, fps, window):
    frame = int(np.round(elapsed * fps))
    windowfps = window * fps

    return frame, windowfps


def get_std_to_plot(data, elapsed, fps, window =2):
    try:
        totframes = len(data)
        frame, windowfps = get_window_forpot(elapsed, fps, window)

        smoothed_velocity = line_smoother(data['Velocity'].values, 51, 3)
        if frame>windowfps and (windowfps+frame)<totframes:
            vel = data['Velocity'][frame-windowfps:frame+windowfps].values
            smoothed_velocity = smoothed_velocity[frame-windowfps:frame+windowfps]
        else:
            vel = [data['Velocity'][frame]]
            smoothed_velocity = vel

        body_centre = (int(data['x'][frame]),  int(data['y'][frame]))

        ori = data['orientation'][np.ceil(elapsed * fps)]

        return body_centre, vel, ori, smoothed_velocity
    except:
        return False, False


def get_dlc_to_plot(data, elapsed, fps, window=2):
    frame, windowfps = get_window_forpot(elapsed, fps, window)
    totframes = len(data['Posture']['body'])

    velocity = data['Posture']['body']['Velocity'].values
    smoothed_velocity = line_smoother(velocity, 51, 3)
    if frame > windowfps and (windowfps + frame) < totframes:
        vel = velocity[frame - windowfps:frame + windowfps]
        smoothed_velocity = smoothed_velocity[frame - windowfps:frame + windowfps]
    else:
        vel = [velocity[frame]]
        smoothed_velocity = vel


    body = (int(data['Posture']['body']['x'][frame])
                   , int(data['Posture']['body']['y'][frame]))

    tail = (int(data['Posture']['snout']['x'][frame])
                   , int(data['Posture']['snout']['y'][frame]))

    head = (int(data['Posture']['tail']['x'][frame])
                   , int(data['Posture']['tail']['y'][frame]))

    return (body, tail, head), vel, smoothed_velocity
