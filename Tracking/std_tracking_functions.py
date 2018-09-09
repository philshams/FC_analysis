import cv2
import numpy as np
import math
from Tracking.Tracking_utils import calc_distance, check_orientation_errors, get_tail_position


def get_body_orientation(f, cnt, bg, display, frame, start_frame, orientation, arena_floor, tail_threshold_scaling):
    """ Get orentation of mouse for STD tracking
    --- OBSOLETE ---"""
    # Fit ellipse on mouse contour
    (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
    angle -= 90
    ellipse = cv2.fitEllipse(cnt)
    cv2.ellipse(frame, ellipse, (0, 0, 255), 1)

    # Get tail position
    tail_pos = get_tail_position(arena_floor, bg, frame, display, tail_threshold_scaling)

    # get vector to tail position and of ellipse
    vectors_dot = 1
    while vectors_dot > 0:
        tail_vec = [tail_pos[0]-x, tail_pos[1]-y]
        theta = math.radians(angle)
        ellipse_vec = [ma*math.cos(theta), ma*math.sin(theta)]

        # get dot product between vectors
        vectors_dot = np.dot(tail_vec, ellipse_vec)

        # Check if we have the correct direction
        if vectors_dot > 0:
            angle += 180

    cv2.line(frame, (int(x), int(y)), (int(x + ellipse_vec[0]), int(y + ellipse_vec[1])), (0, 0, 255), 2)
    real_angle = np.rad2deg(np.arctan2(-ellipse_vec[1], ellipse_vec[0]))
    if real_angle < 0:
        real_angle = 360 + real_angle

    corrected_angle = check_orientation_errors(orientation, real_angle, f, start_frame)

    return corrected_angle


def get_velocity(fps, coord_l):
    """ -- obsolete --
    get velocity for std tracking"""
    vel = 0
    if len(coord_l)>1:
        prev_pos = coord_l[-2]
        curr_pos = coord_l[-1]
        ds = calc_distance(prev_pos, curr_pos)
        dt = 1/fps[0]

        vel = ds/dt
    return vel


def get_mvmt_direction(coord_l):
    """ -- obsolete --
    get direction of movement for std tracking"""
    ang = 0
    if len(coord_l)>1:
        prev_pos = coord_l[-2]
        curr_pos = coord_l[-1]
        dx = curr_pos[0]-prev_pos[0]
        dy = curr_pos[1]-prev_pos[1]

        if dx == 0 or dy == 0:
            return -1
        else:
            ang = -math.atan2(dy, dx)/math.pi*180
            if ang < 0:
                ang = 360 + ang
    return ang
