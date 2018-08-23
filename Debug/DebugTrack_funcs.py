import os

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
            if str(session['Metadata'].session_id) in f:
                short_vids.append(f)
                vids.append(os.path.join(trials_fld, f))

    # Prepare output dict
    output = {}
    for short, long in zip(short_vids, vids):
        output[short] = long

    return output




