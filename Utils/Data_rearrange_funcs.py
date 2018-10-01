from Utils.imports import *


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
        query = input(colored('A cohort with this name already exists in the database.'
                          '\nuse that? [y/n]', 'yellow'))
        if query != 'y':
            a = 1    # TODO extract session from database
            return db, a
        else:
            pass   # create it from scratch

    # Create a pandas DF to append to the database
    keys = list(db.keys())
    keys.append('Processing')
    tempDF = pd.DataFrame(index=[cohort_options['name']], columns=keys)

    # Create a cohort class instantiation and fill in metadata
    ch = Cohort()
    ch.name = cohort_options['name']
    ch.metadata = ch.metadata(datetime.datetime.now().strftime("%y-%m-%d-%H-%M"), cohort_options['selector type'],
                              cohort_options['selector'], [])

    print('=======================\n=======================')
    print(colored('Creating cohort: {}\nSelecting: {} - {}\n'.format(ch.name, cohort_options['selector type'],
                                                                     cohort_options['selector']), 'green', attrs=['bold']))

    # Extract data: loop over all sessions in db and get data from selected ones
    ch_tracking_data = dict(explorations=[], wholesessions=[], trials=[])
    for sess_name, session in db.iterrows():
        # Check if we should add the session to the cohort
        selected = check_session_selected(session.Metadata, ch.metadata.selector_type, ch.metadata.selector)
        if selected:
            print(colored('-----------------\n  ... adding session {}'.format(sess_name), 'green'))
            ch.metadata.sessions_in_cohort.append((sess_name, session.Metadata))
            for data_modality in cohort_options['data to pool']:
                if data_modality == 'tracking':
                    print('         ... getting tracking data')
                    try:
                        for k, val in session['Tracking'].items():
                            if 'exploration' in k.lower():
                                ch_tracking_data['explorations'].append(val)
                            elif k == 'Whole Session':
                                ch_tracking_data['wholesessions'].append(val)
                            else:   # trial
                                ch_tracking_data['trials'].append(val)
                    except:
                        from warnings import warn
                        warn("""        ... no tracking data was found for session {}.'
                              Make sure you are loading the correct database!""".format(sess_name))
                        a = 1
                else:
                    print('Currently cohort formation supports only tracking pooling, not {}'.format(data_modality))
                    # TODO support other modalities

    # Fill in the temp dataframe with the cohort data
    ch.tracking_data = ch.tracking_data(ch_tracking_data['explorations'], ch_tracking_data['wholesessions'],
                                        ch_tracking_data['trials'])
    tempDF.loc[cohort_options['name']].Metadata = ch.metadata
    tempDF.loc[cohort_options['name']].Tracking = ch.tracking_data
    tempDF.loc[cohort_options['name']].Processing = {}

    # Append the temp DF to the database
    db = pd.concat([db, tempDF], sort=False)

    print('       Cohort created. Got Tracking data from sessions {}'.format(ch.metadata.sessions_in_cohort))
    return db, tempDF


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
