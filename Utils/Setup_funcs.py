import pyexcel
import cv2
import numpy as np
import pandas as pd
import datetime
import os
from nptdms import TdmsFile


from Utils.utils_classes import Session_metadata, DataBase


def get_sessions_metadata_from_yaml(datalogpath, database=None):
    # Load excel spreadsheet
    if database is None:
        print('========================\nCreating database from datalog.csv')
    else:
        print('========================\nUpdating database from datalog.csv')

    loaded_excel = pyexcel.get_records(file_name=datalogpath)

    sessions_dict = {}
    # Read each line in the excel spreadsheet and load data accordingly
    for line in loaded_excel:
        session_id = line['Sess.ID']
        session_name = '{}_{}_{}'.format(line['Sess.ID'], line['Date'], line['MouseID'])

        # if we are updating a pre-existing database, check if the session corrisponding to this line
        # already exists in the database. If so, skip the line
        print('------------------------\n     ... Session {}'.format(session_name))
        if database is not None:
            if session_name in database.index:
                print('           ... session already in database')
                continue

        # Create a new entry in the sessions dictionary
        session_metadata = Session_metadata()
        session_metadata.session_id = session_id
        session_metadata.experiment = line['Experiment']
        session_metadata.date = line['Date']
        session_metadata.mouse_id = line['MouseID']
        session_metadata.created = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

        # Get stims from .tdms file
        recordings = line['Sub Folders'].split('; ')
        for recording in recordings:
            path = os.path.join(line['Base fld'], line['Exp fld'], recording)
            for f in os.listdir(path):
                if '.avi' in f:
                    videopath = os.path.join(path, f)
                elif '.tdms' == f[-5:]:
                    tdmspath = os.path.join(path, f)
            # add file paths to metadata
            session_metadata.video_file_paths.append(videopath)
            session_metadata.tdms_file_paths.append(tdmspath)

            # load tdms and get stimuli
            try:
                print('           ... loading metadata from .tdms')
                tdms = TdmsFile(tdmspath)
                df_tdms = tdms.as_dataframe()

                for idx in df_tdms.loc[0].index:
                    if 'Stimulis' in idx:
                        framn = int(idx.split('  ')[1].split('-')[0])
                        if 'Visual' in idx:
                            session_metadata.stimuli['visual'].append(framn)
                        elif 'Audio' in idx:
                            session_metadata.stimuli['audio'].append(framn)
                        elif 'Digital' in idx:
                            session_metadata.stimuli['digital'].append(framn)
                        else:
                            print('couldnt load stim correctly')

            except:
                print('                  ... could not load .tdms  [buggggg need to check why sometimes it breaks')

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


def generate_database(session_dict):  # may be obsolete?
    indexes = sorted(session_dict.keys())
    database_template = DataBase()

    # Create empty database
    database = pd.DataFrame(index=indexes, columns=database_template.sessions.keys())

    # Fill in metadata
    for sessname, metadata in sorted(session_dict.items()):
        database['Metadata'][sessname] = metadata

    return database


