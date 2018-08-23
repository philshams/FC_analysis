import matplotlib.pyplot as plt
import numpy as np
import math
import time

from Utils.maths import calc_distance_2d, calc_velocity, calc_angle_2d


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


def get_average_bodylength(data, tail_tag='', head_tag=''):
    if not tail_tag or not head_tag:
        print('Need to have the name of the head and tail bodyparts to extract bodylength from DLC data')
        return False
    else:
        # Calculate the average length
        head_position, _ = from_dlc_to_single_bp(data, head_tag)
        tail_position, _ = from_dlc_to_single_bp(data, tail_tag)

        numframes = head_position.shape[0]
        lengths = np.zeros((numframes, 1))
        for idx in range(numframes):
            headpoint = (head_position['x'][idx], head_position['y'][idx])
            tailpoint = (tail_position['x'][idx], tail_position['y'][idx])

            lengths[idx] = calc_distance_2d((headpoint, tailpoint), vectors=False)

        return np.mean(lengths)


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


def get_shelter_location(tag, data):
    # Get the position of the centre of the shelter
    if tag == 'roi':
        # Get centre of shelter roi
        roi = data['Metdata']['User ROIs']['shelter']
        shelter_location = (int(roi[0] + (roi[2] / 2)),
                                int(roi[1] + (roi[3] / 2)))
    else:
        # Get it from DLC tracking
        shelter_location, _ = from_dlc_to_single_bp(data, tag)
        shelter_location = (int(np.mean(shelter_location['x'].values)),
                            int(np.mean(shelter_location['y'].values)))
    return shelter_location


def calc_position_relative_point(data, point):
    """
    Gets touple of vectors as DATA and returns a touple of vectors with the position relative to the given point
    """
    numframe = len(data[0])
    x_adj, y_adj = np.zeros((numframe, 1)), np.zeros((numframe, 1))
    for idx in range(numframe):
        x,y = data[0][idx], data[1][idx]
        x_adj[idx] = point[0] - x
        y_adj[idx] = point[1] - y
    return x_adj, y_adj
