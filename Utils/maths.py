from scipy.spatial import distance
import numpy as np


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
