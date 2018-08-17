import pyexcel
import cv2
import numpy as np
import pandas as pd
import datetime

from Utils.utils_classes import Session_metadata, DataBase


def update_sessions_l(datalogpath):
    def return_ordered_stims(stims_string):
        stimuli = []
        if stims_string and not isinstance(stims_string, int):
            stimuli = [int(s) for s in stims_string.split(', ')]
        elif stims_string:
            stimuli = [int(stims_string)]

        return stimuli

    # Load excel spreadsheet
    loaded_excel = pyexcel.get_records(file_name=datalogpath)

    sessions_dict = {}
    # Read each line in the excel spreadsheet and load data accordingly
    for line in loaded_excel:
        session_id = line['Sess.ID']
        session_name = '{}_{}_{}'.format(line['Sess.ID'], line['Date'], line['MouseID'])

        if not session_name in sessions_dict.keys():
            # Create a new entry in the sessions dictionary
            session_metadata = Session_metadata()
            session_metadata.session_id = session_id
            session_metadata.experiment = line['Experiment']
            session_metadata.date = line['Date']
            session_metadata.mouse_id = line['MouseID']
            session_metadata.created = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
        else:
            # Update pre-existing entry in the sessions dictionary
            session_metadata = sessions_dict[session_name]

        # Update the stimuli lists
        session_metadata.stimuli['visual'].append(return_ordered_stims(line['VS']))
        session_metadata.stimuli['audio'].append(return_ordered_stims(line['US']))
        session_metadata.stimuli['digital'].append(return_ordered_stims(line['DS']))

        # Add video file path
        session_metadata.video_file_path.append(line['VideoPath'])

        # Add to dictionary (or update entry)
        sessions_dict[session_name] = session_metadata
    return sessions_dict


def get_session_videodata(session):
    """
    Get relevant variables for video files and crate empty ones to be filled in later on
    """
    # Get first frame of first video for future processing and number of frames in each video
    videos_data = {'Background': None, 'Frame rate': [], 'Number frames': []}

    for idx, videofile in enumerate(session['Metadata'].video_file_path):
        cap = cv2.VideoCapture(videofile)
        videos_data['Frame rate'].append(cap.get(cv2.CAP_PROP_FPS))
        videos_data['Number frames'].append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        if idx == 0:
            _, bg = cap.read()
            videos_data['Background'] = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)

    videos_data['Cumu. Num Frames'] = np.cumsum(videos_data['Number frames'])
    session['Video'] = videos_data
    return session


def generate_database(session_dict):
    indexes = sorted(session_dict.keys())
    database_template = DataBase()

    # Create empty database
    database = pd.DataFrame(index=indexes, columns=database_template.sessions.keys())

    # Fill in metadata
    for sessname, metadata in sorted(session_dict.items()):
        database['Metadata'][sessname] = metadata

    return database


