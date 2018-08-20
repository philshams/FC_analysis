import numpy as np

from Utils.maths import twod_distance, calc_velocity


def from_dlc_to_single_bp(session, settings):
    """
    Currently it extracts position of velocity of the mouse given a user defined tag for a body part

    :param session:
    :param settings:
    :return:
    """
    # Body part to use
    bp_tag = settings['center of mass bodypart']

    data = []
    for tr_name, tr_data in session['Tracking'].items():
        if tr_name == 'Exploration' or tr_name == 'Whole Session':
            continue  # we only care about the trials here

        else:
            if 'Posture' not in tr_data.dlc_tracking.keys():
                continue # Trial was not analysed using DLC

            else:
                # check that the user selected bp is one of the ones present in the data
                bodyparts = tr_data.dlc_tracking['Posture'].keys()
                if not bp_tag in bodyparts:
                    print('The body part selected [{}] is not present in the data.\n'
                          'Please select an alternative among {}'.format(bp_tag, bodyparts))
                    bp_tag = input()
                    if not bp_tag in bodyparts:
                        raise ValueError('Couldnt find a body part that matched your selection')

                data.append(tr_data.dlc_tracking['Posture'][bp_tag])
    return data


def single_bp_calc_stuff(data):
    for trial in data:
        # Calculate
        trial_distance = twod_distance((trial['x'].values, trial['y'].values))
        trial_velocity = calc_velocity(trial_distance)

        # Add to dataframe
        trial['Distance'] = trial_distance
        trial['Velocity'] = trial_velocity

    return data

