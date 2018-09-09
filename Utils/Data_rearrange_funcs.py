import pandas as pd
import numpy as np
import datetime

from Utils.utils_classes import Trial, Cohort

from Config import cohort_options


def check_session_selected(metadata, selector_type: str, selector: list) -> bool:
    """
    Checks if a session should be processed (it's metadata contains an entry of type selector type and
    values selector. 
    
    :param metadata: metadata of session obj
    :param selector_type: name of metadata entry we are selecting by
    :param selector: list of permitted values for sessions to be processed
    :return: yes/no
    """
    if selector_type == 'experiment' and metadata.experiment not in selector:
        return False
    elif selector_type == 'date' and str(metadata.date) not in selector:
        return False
    elif selector_type == 'session' and metadata.session_id not in selector:
        return False
    else:
        return True


def restructure_trial_data(trial, stim_type, idx: int, vid_num: int):
    """
    Gets trial data from tracking and restructures it for a cleaner output
    :param trial: trial obj
    :param stim_type: name of type of stimulus
    :param idx: trial number
    :param vid_num: recording number
    :return: class
    """

    new_trial = Trial()
    new_trial.std_tracking = trial.data
    new_trial.name = '{}_{}_{}'.format(stim_type, vid_num, idx)
    return new_trial


def create_cohort(db):
    """
    Create a cohort class and add it to the database

    :param db: database object
    :return: updated database
    """
    # Check if we are overwriting a pre-existing cohort in the dabatase
    if cohort_options['name'] in db.index:
        overwrite = input('A cohort with this name already exists in the database.'
                          '\nOverwrite? [y/n]')
        if overwrite != 'y':
            print('Process interrupted to avoid overwriting data')
            return

    # Create a pandas DF to append to the database
    tempDF = pd.DataFrame(index=[cohort_options['name']], columns=db.keys())

    # Create a cohort class instantiation and fill in metadata
    ch = Cohort()
    ch.name = cohort_options['name']
    ch.metadata = {
        'created': datetime.datetime.now().strftime("%y-%m-%d-%H-%M"),
        'selector type': cohort_options['selector type'],
        'selector': cohort_options['selector'],
        'sessions in cohort': []
    }
    print('=======================\n=======================')
    print('Creating cohort: {}\nSelecting: {} - {}\n'.format(ch.name, ch.metadata['selector type'],
                                                            ch.metadata['selector']))

    # Extract data: loop over all sessions in db and get data from selected ones
    for sess_name, session in db.iterrows():
        # Check if we should add the session to the cohort
        selected = check_session_selected(session.Metadata, ch.metadata['selector type'], ch.metadata['selector'])
        if selected:
            print('-----------------\n  ... adding session {}'.format(sess_name))
            ch.metadata['sessions in cohort'].append(sess_name)
            for data_modality in cohort_options['data to pool']:
                if data_modality == 'tracking':
                    print('         ... getting tracking data')
                    try:
                        for k, val  in session['Tracking']:
                            if k == 'Exploration':
                                ch.tracking_data['explorations'].append(val)
                            elif k == 'Whole Session':
                                ch.tracking_data['whole sessions'].append(val)
                            else:   # trial
                                ch.tracking_data['trials'].append(val)
                    except:
                        from warnings import warn
                        warn('        ... no tracking data was found. Make sure you are loading the correct database!')
                else:
                    print('Currently cohort formation supports only tracking pooling, not {}'.format(data_modality))
                    # TODO support other modalities

    # Fill in the temp dataframe with the cohort data
    tempDF.iloc[0]['Metadata'] = ch.metadata
    tempDF.iloc[0]['Tracking'] = ch.tracking_data

    # Append the temp DF to the database
    db = pd.concat([db, tempDF])

    print('       Cohort created. Got Tracking data from sessions {}'.format(ch.metadata['sessions in cohort']))
    return db


def arrange_dlc_data(df):
    """
    Restructure DLC dataframe
    """
    scorer = df.keys()[0][0]
    bodyparts = df[scorer].columns.levels[0]
    data = {}
    for bp in bodyparts:
        data[bp] = df[scorer][bp]
    return data
