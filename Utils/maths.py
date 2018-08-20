from scipy.spatial import distance
import numpy as np
from math import factorial


def twod_distance(data, vectors = True):
    """
    Calculates the euclidean distance between each consecutive pair point in two points or vectors
    """

    if not vectors:
        return distance.euclidean(data[0], data[1])
    else:
        dist = []
        for n, pos in enumerate(zip(data[0], data[1])):
            # Get a pair of points
            if n == 0:
                p0 = pos
                dist.append(0)
            else:
                p1 = pos

                # Calc distance
                dist.append(distance.euclidean(p0, p1))

                # Prepare for next iteration
                p0 = p1
        return  dist


def calc_velocity(d):
    return np.insert(np.diff(d), 0, 0)


def line_smoother(y, window_size, order, deriv=0, rate=1):
    # Apply a Savitzy-Golay filter to smooth traces
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
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
