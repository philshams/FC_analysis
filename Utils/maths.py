from Utils.imports import *

from Utils.decorators import clock_noself

def calc_distance_2d(data, vectors = True):
    """
    Calculates the euclidean distance between point, or each pair of points in vectors
    """
    # TODO testing
    if not vectors:
        return distance.euclidean(data[0], data[1])
    else:
        dist = []
        if isinstance(data[0], list) or isinstance(data[0], dict):
            raise Warning('This case needs to be dealt with')
        else:
            try:
                data = (data[0].values, data[1].values)
            except:
                pass
        for n, pos in enumerate(zip(data[0], data[1])):
            # Get a pair of points
            if n == 0:
                p0 = pos
                dist.append(0)
            else:
                p1 = pos

                # Calc distance
                try:
                    dist.append(distance.euclidean(p0, p1))
                except:
                    if np.isnan(p1).any():
                        dist.append(np.nan)

                # Prepare for next iteration
                p0 = p1

        return dist


def calc_acceleration(d, unit: str=False, fps: int = False, bodylength: float = False):
    """  Calculates the acceleration (1st derivative of velocity). different options for output format """
    if not unit or unit == 'pxperframe':
        # Return the velocity in px per frame
        return np.insert(np.diff(d), 0, 0)
    else:
        # Scale the velocity from px per frame depending on the unit used
        velocity = np.insert(np.diff(d), 0, 0)
        if not fps:
            print('No FPS was available when calculating velocity\n FPS set as 30 frames per second')
            fps = 30
        else:
            if isinstance(fps, list):
                fps = fps[0]

        if unit == 'pxpersec':
            return velocity*fps

        if unit =='blpersec':
            if not bodylength:
                print('No body length was found when calculating velocity as bodylengths per second\n'
                      'Using px per second instead')
                return velocity*fps
            else:
                velocity = velocity * fps
                velocity = velocity / bodylength
    return velocity


def calc_angle_2d(p1, p2, vectors: bool=False):
    """ calculates the angle of a line going through two points, or sets of points in two vectors"""
    def angle(a, b):
        radang = atan2(b[1] - a[1], b[0] - a[0])
        degang = degrees(radang)
        if degang < 0:
            return 360 + degang
        else:
            return degang

    if not vectors:
        # Calc for just two points
        return angle(p1, p2)
    else:
        # calc for two vectors of points
        if isinstance(p1, pd.DataFrame):
            p1 = np.vstack((p1['y'].values, p1['x'].values))
            p2 = np.vstack((p2['y'].values, p2['x'].values))

            deltas = np.subtract(p1.T, p2.T)
            angs = np.degrees(np.arctan2(deltas[:, 0], deltas[:, 1]))
            negs = np.where(angs < 0)[0]
            angs[negs] += 360
            angs += 90
        # angles = []
        # frames = len(p1['x'])
        # for idx in range(frames):
        #     angles.append(angle((p1.loc[idx]['x'], p1.loc[idx]['y']),
        #                         (p2.loc[idx]['x'], p2.loc[idx]['y'])))
            return angs


def calc_ang_velocity(orientation, fps: int=False):
    """
    Given a vector of orientation (degrees) per frame, calculates the velocity as either degrees per frame
    or degrees per second (if fps != False).

    :param orientation: vector of angle values
    :param fps:  frame rate of video the orientation was extracted from
    :return: angular velocity as either deg per sec or deg per frame.
    """
    rad_ori = np.radians(orientation.values)
    rad_ang_vel = np.insert(np.diff(np.unwrap(rad_ori)), 0, 0)

    if not fps:    # return and vel as degrees per frame
        return np.degrees(rad_ang_vel)
    else:          # return and vel as degrees per sec
        return np.degrees(np.multiply(rad_ang_vel, fps))


def calc_ang_acc(velocity):
    """ calculates the angular acceleration given a angular velocity vector"""
    return np.insert(np.diff(velocity), 0, 0)


def line_smoother(y, window_size=31, order=3, deriv=0, rate=1):
    # Apply a Savitzy-Golay filter to smooth traces
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with values taken from the signal itself
    try:
        firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve(m[::-1], y, mode='valid')
    except:
        y = np.array(y)
        firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve(m[::-1], y, mode='valid')

















