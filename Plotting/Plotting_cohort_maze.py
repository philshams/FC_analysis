import numpy as np
import warnings
import matplotlib.pyplot as plt


def plot_status_at_timepoint(trial_data, timepoint='stim', stim_type=['visual'], align_to='Threat'):
    """
    Plot mouse position, orientation, velocity ... at a specified timepoint.

    timepoint = 'stim' --> at stimulus presentation [midpoint of trial data]
    timepoint = 'rt' --> at reaction time

    stim_type = for which set of trials we want to plot (e.g. looms, audio, all...)

    align_to: X and Y position is normalised to some point in the arena to correct for difference in position
    relative to camera frame inbetween experiments. Select which of the user defined ROIs to use for this.

    """
    ###############################################################################################
    # Data processing

    # Select the datapoint corresponding to the stimulus of interest
    if timepoint == 'stim':
        if trial_data.visual:
            datapoint = int(np.floor(len(trial_data.visual[0].x )/2))
        else:
            try:
                datapoint = int(np.floor(len(trial_data.audio[0].x )/2))
            except:
                warnings.warn('Couldnt determine the time point to align data')
                return

    # Extract the data from trials
    status_at_timepoint = {'position':[], 'orientation':[], 'velocity':[]}
    for stim, trials in trial_data.__dict__.items():
        if stim in stim_type:
            for trial in trials:
                if not trial.x.any():
                    continue

                x_normaliser = int(trial.rois[align_to][0]+(trial.rois[align_to][2]/2))
                y_normaliser = int(trial.rois[align_to][1]+(trial.rois[align_to][3]/2))

                status_at_timepoint['position'].append((trial.x[datapoint]-x_normaliser,
                                                        trial.y[datapoint]-y_normaliser))

                if trial.x[datapoint]-x_normaliser<-100 or  trial.y[datapoint]-y_normaliser<-100:
                    a = 1

                status_at_timepoint['orientation'].append(trial.orientation[datapoint])
                status_at_timepoint['velocity'].append(trial.velocity[datapoint])

    ##################################################################################################
    # Plotting starts here
    num_columns = 3
    num_plots = len(status_at_timepoint.keys())
    num_rows = int(np.ceil(num_plots/num_columns))

    fig_title = 'Staus at reaction - all trials - {} stim'.format(stim_type)

    f = plt.figure()

    for idx, (key, val) in enumerate(status_at_timepoint.items()):
        if not key == 'orientation':
            ax = plt.subplot(num_rows, num_columns, idx+1)
        else:
            ax = plt.subplot(num_rows,  num_columns, idx+1, projection='polar')

        ax.set(title=key)

        if key != 'position':
            if key == 'velocity':
                xx = np.linspace(0, len(status_at_timepoint['velocity']), len(status_at_timepoint['velocity']))
                ax.scatter(xx, sorted(status_at_timepoint[key]))

            else:
                xx = np.ones((1, len(status_at_timepoint['orientation'])))
                ax.scatter(sorted(status_at_timepoint[key]), xx)
        else:
            ax.scatter([x for x,y in status_at_timepoint['position']],
                       [y for x,y in status_at_timepoint['position']])
            ax.set(xlim=[-75, 75], ylim=[-75, 75])

    plt.show()
    a = 1




