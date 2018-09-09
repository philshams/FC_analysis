import pyexcel
import cv2
import numpy as np
import pandas as pd
import datetime
import os
from nptdms import TdmsFile
import sys
from multiprocessing.dummy import Pool as ThreadPool

from Utils.utils_classes import Session_metadata, DataBase
from Utils.loadsave_funcs import save_data

from Config import save_name, savelogpath


def create_database(datalogpath, database=None):
    """
    Creates a new database from scratch loading the info from datalog.csv
    If a pre-existing database is passed to it it will simply updated it adding the sessions that are in
    datalog.csv but are not already present in the database

    It loads the info from datalog.csv, crates the database [calling generate_database_from_metadatas()],
    and extracts info from the each session's .tdms file.

    :param datalogpath: path where the datalog.csv file is saved
    :param database: Optional. Database object to be updated. If missing a new one is created
    :return:
    """

    def generate_database_from_metadatas(session_dict):
        """
        Given a dictionary of session objects with their metadata creates a new database
        with all the sessions in the database and the associated metadata

        :param session_dict: dictionary. Keys are sessions name and values are sessions metadata
        :return: the newly created database
        """

        # Create empty database from template class
        database_template = DataBase()
        indexes = sorted(session_dict.keys())
        database = pd.DataFrame(index=indexes, columns=database_template.sessions._fields)

        # Fill in metadata from the dictionary
        for sessname, metadata in sorted(session_dict.items()):
            database['Metadata'][sessname] = metadata

        print('==========================\n==========================\n\nDatabase initialised succesfully')
        return database

    def get_session_videodata(videos):
        """
        Get relevant variables for video files and crate empty ones to be filled in later on
        """
        # Get first frame of first video for future processing and number of frames in each video
        videos_data = {'Background': None, 'Frame rate': [], 'Number frames': []}

        for idx, videofile in enumerate(videos):
            cap = cv2.VideoCapture(videofile)
            videos_data['Frame rate'].append(cap.get(cv2.CAP_PROP_FPS))
            videos_data['Number frames'].append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

            if idx == 0:
                _, bg = cap.read()
                videos_data['Background'] = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)

        videos_data['Cumu. Num Frames'] = np.cumsum(videos_data['Number frames'])
        return videos_data

    def process_list_of_sessions(sessions):
        """
        loops over a list of dictionary with the info for each session and gets all the metadata

        :param sessions:
        :return:
        """
        if not isinstance(sessions, list):
            sessions = list(sessions)

        for line in sessions:
            session_id = line['Sess.ID']
            # If we loaded and empty line, stop
            if not session_id:
                continue

            session_name = '{}_{}_{}'.format(line['Sess.ID'], line['Date'], line['MouseID'])

            # if we are updating a pre-existing database, check if the session corrisponding to this line
            # already exists in the database. If so, skip the line
            print('       ... Session {}'.format(session_name))
            if database is not None:
                if session_name in database.index:
                    print('           ... session already in database')
                    continue

            # Create the metadata
            session_metadata = Session_metadata()
            session_metadata.session_id = session_id
            session_metadata.experiment = line['Experiment']
            session_metadata.date = line['Date']
            session_metadata.mouse_id = line['MouseID']
            session_metadata.created = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
            session_metadata.software = line['Software']

            # If the paths are correct, loop over each subfolder again but this time load the data
            for recording in line['Recordings']:
                path = os.path.join(line['Base fld'], line['Exp fld'], recording)

                # If the path to the recording doesn't exist the program will crash
                # To avoid loosing data, create a database and save it
                if not os.path.exists(path):
                    print('-- !! Something went wrong!\nThis path doesnt exist: {}'.format(path))
                    print('Saving the database created so far as {}'.format(os.path.join(savelogpath,
                                                                                         save_name + '_emergency_save')))
                    # Create or update a database and save it so that we don't loos data
                    if database is None:
                        # Create a new database from the metadata we have collected so far [ will only run once ]
                        db = generate_database_from_metadatas(sessions_dict)
                    else:
                        # create a new database with the newly added sessions and concatenate it with the preivious one
                        db = generate_database_from_metadatas(sessions_dict)
                        # db = pd.concat(database, justaddedd)

                    save_data(savelogpath, save_name, name_modifier='_emergency_save', object=db)
                    sys.exit('Closing application....')

                # If the program exists, get all .avi and .tdms files in the folder
                videopaths = []
                for f in os.listdir(path):
                    if '.avi' in f:
                        videopaths.append(os.path.join(path, f))
                    elif '.tdms' == f[-5:]:
                        tdmspath = os.path.join(path, f)

                # add file paths to metadata
                session_metadata.video_file_paths.append(videopaths)
                session_metadata.tdms_file_paths.append(tdmspath)

                # Loop over each video and get the relevant data [e.g., number of frames, fps...]
                session_metadata.videodata.append(get_session_videodata(videopaths))

                # Loop over each .tdms file and extract stimuli frames
                """
                TODO: make the following bit of code behave differently for data originated from the behaviour software
                or from mantis. 
                TODO: allow the possibility of extracting stimuli in TIME instead of FRAME
                """
                try:
                    # Try to load a .tdms
                    print('           ... loading metadata from .tdms')
                    tdms = TdmsFile(tdmspath)
                    if session_metadata.software == 'behaviour':
                        visual_rec_stims, audio_rec_stims, digital_rec_stims = [], [], []
                        # loop over tdms groups and extract frames of stimuli
                        # TODO extract stim metadata
                        for group in tdms.groups():
                            for obj in tdms.group_channels(group):
                                if 'stimulis' in str(obj).lower():
                                    for idx in obj.as_dataframe().loc[0].index:
                                        if '  ' in idx:
                                            framen = int(idx.split('  ')[1].split('-')[0])
                                        else:
                                            framen = int(idx.split(' ')[2].split('-')[0])
                                        if 'visual' in str(obj).lower():
                                            visual_rec_stims.append(framen)
                                        elif 'audio' in str(obj).lower():
                                            audio_rec_stims.append(framen)
                                        elif 'digital' in str(obj).lower():
                                            digital_rec_stims.append(framen)
                                        else:
                                            print('                  ... couldnt load stim correctly')

                        session_metadata.stimuli['visual'].append(visual_rec_stims)
                        session_metadata.stimuli['audio'].append(audio_rec_stims)
                        session_metadata.stimuli['digital'].append(digital_rec_stims)

                    else:
                        # TODO add mantis tdms reading stuff
                        """
                        HERE IS WERE THE CODE TO GET THE STIM TIMES IN MANTIS WILL HAVE TO BE ADDEDD
                        """
                        pass

                except:
                    print('                  ... could not load .tdms ')

            # Add to dictionary (or update entry)
            sessions_dict[session_name] = session_metadata
        return sessions_dict

    ####################################################################################################################
    ####################################################################################################################

    # Load excel spreadsheet
    if database is None:
        print('========================\nCreating database from datalog.csv')
    else:
        print('========================\nUpdating database from datalog.csv')

    try:
        loaded_excel = pyexcel.get_records(file_name=datalogpath)
    except:
        print('Could not load datalog, these are the excel files in the folder:')
        counter = 0
        files = []
        directory, _ = os.path.split(datalogpath)  # path.split returns [path, filename.abc]
        for f in os.listdir(directory):
            if 'csv' in f or 'xls' in f:
                print('({})  -  {}'.format(counter, f))
                counter += 1
                files.append(f)
        selected = input('Enter number of the file to be loaded:  ')
        selected = files[int(selected)]
        loaded_excel = pyexcel.get_records(file_name=os.path.join(directory, selected))

    # <-- Create a dictionary with each session's name as key and its metadata as value
    sessions_dict = {}

    # Read each line in the excel spreadsheet and load info
    all_metadata = []
    for line in loaded_excel:
        temp = {
            'Sess.ID': line['Sess.ID'],
            'Date':line['Date'],
            'MouseID':line['MouseID'],
            'Experiment':line['Experiment'],
            'Software':line['Software'],
            'Base fld':line['Base fld'],
            'Exp fld': line['Exp fld'],
            'Recordings': line['Sub Folders'].split('; ')
        }
        all_metadata.append(temp)

    # Loop over each recordings subfolder and check that the paths are correct [fast check that everything
    # is okay before loading the data]
    for line in all_metadata:
        for recording in line['Recordings']:
            path = os.path.join(line['Base fld'], line['Exp fld'], recording)
            if not os.path.exists(path):
                raise ValueError('Folder not found\n{}'.format(path))
    print('     Excel spreadsheet loaded correctly. Now loading metadata')

    # Use loaded metadata to create the database. Threadpooled for faster execution
    num_parallel_processes = 3
    splitted_all_metadata = [all_metadata[i::num_parallel_processes] for i in range(num_parallel_processes)]
    pool = ThreadPool(num_parallel_processes)
    _ = pool.map(process_list_of_sessions, splitted_all_metadata)

    if database is None:
        return generate_database_from_metadatas(sessions_dict)
    else:
        new_database = generate_database_from_metadatas(sessions_dict)
        frames = [database, new_database]
        return pd.concat(frames, sort=True)





