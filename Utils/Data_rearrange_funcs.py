import pandas as pd
import numpy as np
import datetime

from Utils.utils_classes import Trial, All_trials, Cohort

from Config import cohort_options


def restructure_trial_data(trial, start_frame, stop_frame, stim_type, idx, vid_num):
    new_trial = Trial()

    new_trial.std_tracking['x'] = trial.x[0][start_frame:stop_frame-1]
    new_trial.std_tracking['y'] = trial.y[0][start_frame:stop_frame-1]
    new_trial.std_tracking['orientation'] = trial.orientation[0][start_frame:stop_frame-1]
    new_trial.std_tracking['direction'] = trial.direction[0][start_frame:stop_frame-1]
    new_trial.std_tracking['velocity'] = trial.velocity[0][start_frame:stop_frame-1]

    new_trial.name = '{}_{}_{}'.format(stim_type, vid_num, idx)

    return new_trial


def create_cohort(db):
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

        sel, seltype = ch.metadata['selector'], ch.metadata['selector type']
        # If we don't want to add the session to the cohort just skip it
        if seltype == 'experiment' and session.Metadata.experiment not in sel:
            continue
        elif seltype == 'date' and str(session.Metadata.date) not in sel:
            continue
        elif seltype == 'session' and session.Metadata.session_id not in sel:
            continue

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
                        else:
                            # tracking data from a trial
                            ch.tracking_data['trials'].append(val)
                except:
                    print('           ... no tracking data was found. Make sure you are loading the correct database!')

    # Now fill in the temp dataframe with the cohort data
    tempDF.iloc[0]['Metadata'] = ch.metadata
    tempDF.iloc[0]['Tracking'] = ch.tracking_data

    # Now append the temp DF to the database
    db = pd.concat([db, tempDF])

    print('       Cohort created. Got Tracking data from sessions {}'.format(ch.metadata['sessions in cohort']))

    return db