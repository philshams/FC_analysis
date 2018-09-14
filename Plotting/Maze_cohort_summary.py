import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('axes', edgecolor=[0.8, 0.8, 0.8])
matplotlib.rcParams['text.color'] = [0.8, 0.8, 0.8]
matplotlib.rcParams['axes.labelcolor'] = [0.8, 0.8, 0.8]
matplotlib.rcParams['axes.labelcolor'] = [0.8, 0.8, 0.8]
matplotlib.rcParams['xtick.color'] = [0.8, 0.8, 0.8]
matplotlib.rcParams['ytick.color'] = [0.8, 0.8, 0.8]
params = {'legend.fontsize': 16,
          'legend.handlelength': 2}
plt.rcParams.update(params)

import numpy as np
import seaborn as sns
import pandas as pd
import os
from termcolor import colored

from Plotting.Plotting_utils import make_legend
from Utils.Messaging import send_email_attachments
from Utils.loadsave_funcs import load_yaml
from Utils.decorators import clock

from Config import plotting_options, send_messages, cohort_options


class MazeCohortPlotter:
    """
    Plots a summary of a whole cohort (or a whole cohort).  It plots:
        * A heatmap of the outcomes of the trials

    # TODO check that it works with cohorts too
    """
    def __init__(self, cohort):
        print(colored('\n      Plotting whole session summary', 'green'))
        # Prep up useful variables
        self.colors = [[.2, .2, .2], [.8, .4, .4], [.4, .8, .4], [.4, .4, .8]]
        self.cohort = cohort
        self.settings = load_yaml(plotting_options['cfg'])

        # Create figure and set up axes
        self.set_up_fig()

        # Plot
        self.cohort_plot()

        # Save or display the figure
        if self.settings['save figs']:
            path = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\data\\z_TrialImages'
            name = '{}'.format(cohort_options['name'])
            print(colored('             ... saving figure {}'.format(name), 'grey'))
            self.f.savefig(os.path.join(path, name), facecolor=[0.45, 0.45, 0.45])
            if send_messages:
                send_email_attachments(name, os.path.join(path, name))
            plt.close('all')
        else:
            plt.show()

    def set_up_fig(self):
        """ creates the matpltlib figure and axes """
        self.f = plt.figure(figsize=(35, 15), facecolor=[0.1, 0.1, 0.1])
        grid = (4, 4)

        # Trials heatmap
        self.origins_heatmap = plt.subplot2grid(grid, (0, 0), rowspan=1, colspan=1)
        self.origins_heatmap.set(title='origin arms',
                                 facecolor=[0.2, 0.2, 0.2], xlim=[300, -300], ylim=[-500, 100], xlabel='trials')

        self.escapes_heatmap = plt.subplot2grid(grid, (0, 1), rowspan=1, colspan=1)
        self.escapes_heatmap.set(title='escape arms',
                                 facecolor=[0.2, 0.2, 0.2], xlim=[300, -300], ylim=[-500, 100], xlabel='trials')
        self.f.tight_layout()

    @clock
    def cohort_plot(self):
        self.plot_outcomes_heatmap()

    def plot_outcomes_heatmap(self):
        """ Creates a heatmap of all trials color coded by the path taken.
            First row is the origins, bottom row is the scapes """
        origins = self.cohort['Processing'][0]['Origins']
        escapes = self.cohort['Processing'][0]['Escapes']
        arr_length = 40
        escs = np.zeros((len(escapes), arr_length))
        for idx, vals in enumerate(escapes.values()):
            escs[idx, :len(vals)] = np.asarray(vals)

        ors = np.zeros((len(origins), arr_length))
        for idx, vals in enumerate(origins.values()):
            ors[idx, :len(vals)] = np.asarray(vals)

        # Plot and return results
        sns.heatmap(ors, ax=self.origins_heatmap, cmap=self.colors, vmin=0, vmax=3, cbar=False, annot=False,
                    linewidths=.5, yticklabels=False)
        self.origins_heatmap.set(xlabel='Trials', ylabel='Mice')

        sns.heatmap(escs, ax=self.escapes_heatmap, cmap=self.colors, vmin=0, vmax=3, cbar=False, annot=False,
                    linewidths=.5, yticklabels=False)
        self.escapes_heatmap.set(xlabel='Trials', ylabel='Mice')

