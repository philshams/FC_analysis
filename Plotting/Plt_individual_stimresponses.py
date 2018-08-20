import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from Utils.maths import line_smoother


def plotter(data, smoothed=True, padding=[30, 120]):
    """
    Plot the aligned traces of position, velocity, angle ...
    For a single mouse in a single session, all trials

    :param data:
    :return:
    """
    midpoint = int(len(data[0]['x']) / 2)
    padding = [midpoint - padding[0], midpoint + padding[1]]  # zoom on X axis

    titles = ['Position', 'velocity']

    f, axarr = plt.subplots(2, len(titles))
    axarr = axarr.flatten()

    # Set up axes
    # Position axis
    pos_ax = axarr[0]
    pos_ax.set(title=titles[0])

    # Velocity axis
    vel_ax = axarr[1]
    vel_ax.set(title=titles[1], xlim=padding, ylim=[-0.75, 0.75])
    vel_ax.xaxis.set_major_locator(ticker.MultipleLocator(150))

    # Plot individual trials
    vel_l = []
    for trial in data:
        # Prep data
        if not smoothed:
            vel = trial['Velocity'].values
        else:
            vel = trial['Velocity'].values
            vel = line_smoother(vel, 51, 3)
        vel_l.append(vel[0:padding[1]])

        # Plot position
        pos_ax.plot(trial['x'], trial['y'], color=[0.4, 0.4, 0.4])

        # Plot velocity
        XX = np.linspace(0, len(vel), len(vel))
        vel_ax.plot(XX, vel, color=[0.4, 0.4, 0.4])

        vel_ax.axvline(x=midpoint, color='k')

    # plot means
    vel_ax.plot(np.mean(vel_l, 0), color='r')

    plt.show()



