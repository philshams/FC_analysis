import pyexcel
import cv2
import numpy as np
import pandas as pd
import datetime
import os
from nptdms import TdmsFile
import sys

from Utils.utils_classes import Session_metadata, DataBase
from Utils.loadsave_funcs import save_data

from Config import save_name, savelogpath


def generate_database_from_metadatas(session_dict):  # may be obsolete?
    indexes = sorted(session_dict.keys())
    database_template = DataBase()

    # Create empty database
    database = pd.DataFrame(index=indexes, columns=database_template.sessions.keys())

    # Fill in metadata
    for sessname, metadata in sorted(session_dict.items()):
        database['Metadata'][sessname] = metadata

    print('==========================\n==========================\n\nDatabase initialised succesfully')

    return database


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

            # if the path we got doesn't exist, save the results as they are now so that we don't loose everything
            if not os.path.exists(path):
                print('-- !! Something went wrong!\nThis path doesnt exist: {}'.format(path))
                print('Saving the database created so far as {}'.format(os.path.join(savelogpath,
                                                                                     save_name+'_emergency_save')))
                if database is None:
                    db = generate_database_from_metadatas(sessions_dict)
                else:
                    justaddedd = generate_database_from_metadatas(sessions_dict)
                    db = pd.concat(database, justaddedd)

                save_data(savelogpath, save_name, name_modifier='_emergency_save', object=db)
                sys.exit('Closing application....')

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
                        # get frame number
                        try:
                            if '  ' in idx:
                                framen = int(idx.split('  ')[1].split('-')[0])
                            else:
                                framen = int(idx.split(' ')[2].split('-')[0])
                        except:
                            print('                  ... Something went wrong while trying'
                                  ' to extract stimulus frame from tdms')
                            a = 1

                        # store frame number in metadata
                        if 'Visual' in idx:
                            session_metadata.stimuli['visual'].append(framen)
                        elif 'Audio' in idx:
                            session_metadata.stimuli['audio'].append(framen)
                        elif 'Digital' in idx:
                            session_metadata.stimuli['digital'].append(framen)
                        else:
                            print('                  ... couldnt load stim correctly')

            except:
                print('                  ... could not load .tdms ')

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




