import matplotlib.pyplot as plt
import numpy as np
import math
import time

from Utils.maths import twod_distance, calc_velocity, calc_angle_2d


def from_dlc_to_single_bp(data, bp_tag):
    """
    Currently it extracts position of velocity of the mouse given a user defined tag for a body part

    :param data: dlc tracking data
    :param bp_tag: name of the body part to extract
    :return:
    """

    if 'Posture' not in data.dlc_tracking.keys():
        return  # Trial was not analysed using DLC

    else:
        # check that the user selected bp is one of the ones present in the data
        bodyparts = data.dlc_tracking['Posture'].keys()
        if not bp_tag in bodyparts:
            print('The body part selected [{}] is not present in the data.\n'
                  'Please select an alternative among {}'.format(bp_tag, bodyparts))
            bp_tag = input()
            if not bp_tag in bodyparts:
                raise ValueError('Couldnt find a body part that matched your selection')

        output = data.dlc_tracking['Posture'][bp_tag]
    return output, bp_tag


def get_tracking_velocity(session, settings, std=False, dlc=False, dlc_single_bp=True):
    """
    Calculates the velocity of the mouse from tracking data.
    Get get std tracking (i.e. center of mass) or DLC tracking (many bodyparts)
    Can give the velocity in px/frame, px/sec or bodylengths/s

    :param session: session entry of the database
    :param std - extract velocity from STD data
    :param dlc - extract velocity from DLC data
    :param dlc_single_bp - use only center of mass bodypart

    :return:
    """

    # Prepare DLC data


    # Loop over each trial
    for tr_name, tr_data in session.Tracking.items():
        if tr_name == 'Exploration' or tr_name == 'Whole Session':
            continue  # we only care about the trials here

        if std:
            coordinates = [c for c in zip(tr_data.std_tracking['x'], tr_data.std_tracking['y'])]
            distance = twod_distance((tr_data.std_tracking['x'], tr_data.std_tracking['y']))
            velocity = calc_velocity(distance)

        if dlc:
            if dlc_single_bp:
                body_data = from_dlc_to_single_bp(session, settings['center of mass bodypart'])

    a = 1


def single_bp_calc_stuff(data):
    for trial in data:
        # Calculate
        trial_distance = twod_distance((trial['x'].values, trial['y'].values))
        trial_velocity = calc_velocity(trial_distance)

        # Add to dataframe
        trial['Distance'] = trial_distance
        trial['Velocity'] = trial_velocity

    return data


def pose_reconstruction(head, body, tail, debug=False):
    """
    Give DLC data for head, body and tail features reconstructs the body and head absolute and relative angles

    :param head:
    :param body:
    :param tail:
    :return:
    """

    if debug:
        f, axarr = plt.subplots(2,1)
        pol = plt.subplot(2, 1, 1, projection='polar')
        plt.ion()

    for tr_num, trial in enumerate(zip(head, body, tail)):
        head_body_angle = []

        tr_head = (trial[0]['x'].values, trial[0]['y'].values)
        tr_body = (trial[1]['x'].values, trial[1]['y'].values)
        tr_tail = (trial[2]['x'].values, trial[2]['y'].values)

        for idx in np.arange(0, len(tr_body[0])):
            hp = (tr_head[0][idx], tr_head[1][idx])
            bp = (tr_body[0][idx], tr_body[1][idx])
            tp = (tr_tail[0][idx], tr_tail[1][idx])

            head_angle = calc_angle_2d(bp, hp)
            body_angle = calc_angle_2d(bp, tp)

            head_body_angle.append(abs(int(head_angle-body_angle)))

            if debug:
                axarr[1].scatter(hp[0], hp[1], color='r')
                axarr[1].scatter(bp[0], bp[1], color='b')
                axarr[1].scatter(tp[0], tp[1], color='g')

                pol.scatter(math.radians(head_angle), 1, color='r')
                pol.scatter(math.radians(body_angle), 1.25, color='g')

                time.sleep(0.5)
                axarr[1].cla()
                pol.cla()

        body[tr_num]['HeadBodyAngle'] = head_body_angle

    return body
