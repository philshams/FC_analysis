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

from Config import plotting_options, send_messages


class MazeSessionPlotter:
    """
    Plots a summary of a whole session (or a whole cohort).  It plots:
        * A heatmap of the outcomes of the trials
        * The probability of taking different paths
        * The probability of taking different paths as a function of:
                - horizontal position on the threat platform

    # TODO check that it works with cohorts too
    """
    def __init__(self, session):
        print(colored('\n      Plotting whole session summary', 'green'))
        # Prep up useful variables
        self.colors = [[.2, .2, .2], [.8, .4, .4], [.4, .8, .4], [.4, .4, .8]]
        self.session = session
        self.settings = load_yaml(plotting_options['cfg'])

        # Create figure and set up axes
        self.set_up_fig()

        # Plot
        self.session_plot()

        # Save or display the figure
        if self.settings['save figs']:
            path = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\data\\z_TrialImages'
            name = '{}'.format(self.session.name)
            print(colored('             ... saving figure {}'.format(name), 'grey'))
            self.f.savefig(os.path.join(path, name), facecolor=[0.3, 0.3, 0.3])
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
        self.arms_heatmap = plt.subplot2grid(grid, (0, 0), rowspan=1, colspan=1)
        self.arms_heatmap.set(title='origin and escape arms',
                              facecolor=[0.2, 0.2, 0.2], xlim=[300, -300], ylim=[-500, 100], xlabel='trials')

        # Probability of taking each arm
        self.arms_probs = plt.subplot2grid(grid, (0, 1), rowspan=1, colspan=1)
        self.arms_probs.set(title='Probability per arm', facecolor=[0.2, 0.2, 0.2])

        # Probability of escaping on a arm given hor. position
        self.x_pos_arm_prob = plt.subplot2grid(grid, (1, 0), rowspan=1, colspan=1)
        self.x_pos_arm_prob.set(title='Probability per arm per x pos', facecolor=[0.2, 0.2, 0.2],
                                xlabel='X position (binned)', ylabel='Probability', xticks=([]))
        self.f.tight_layout()

    @clock
    def session_plot(self):
        self.origins, self.escapes, self.origins_escapes_legends = self.plot_outcomes_heatmap()
        self.plot_probability_per_arm()
        self.plot_probs_as_func_x_pos()

    def plot_outcomes_heatmap(self):
        """ Creates a heatmap of all trials color coded by the path taken.
            First row is the origins, bottom row is the scapes """
        ors = self.session.processing['Origins']
        escs = self.session.processing['Escapes']
        legends = np.vstack((np.asarray(ors), np.asarray(escs)))

        # Go from string encoding of outcomes to numerical for plotting
        ors = [1 if x=='Left' else 2 if x=='Central' else 3 if x=='Right' else 0 for x in ors]
        escs = [1 if x=='Left' else 2 if x=='Central' else 3 if x=='Right' else 0 for x in escs]
        arms = np.vstack((np.array(ors), np.array(escs)))

        # Plot and return results
        sns.heatmap(arms, ax=self.arms_heatmap, cmap=self.colors, vmin=0, vmax=3, cbar=False, annot=True,
                    linewidths=.5, yticklabels=False)
        self.arms_heatmap.set(xlabel='Trials', ylabel='Escapes - Origins')
        return ors, escs, legends

    def plot_probability_per_arm(self):
        """ Plot the probability of having an origin or an escape along each arm """
        classes = np.array(list(('origin', 'escape')*4))
        possibilites_s = np.array([item for pair in zip( ['None', 'Left', 'Centre', 'Right'],
                                                         ['None', 'Left', 'Centre', 'Right'] + [0]) for item in pair])
        if self.session.processing['Origin probabilities'] is not None:
            probs = np.array([item for pair in zip(self.session.processing['Origin probabilities'].per_arm,
                                                   self.session.processing['Escape probabilities'].per_arm
                                                   + [0]) for item in pair])
            data = pd.DataFrame({'class':classes, 'types':possibilites_s, 'probs':probs})

            sns.factorplot(x="types", y='probs', hue='class',  data=data, ax=self.arms_probs, kind='bar')
            make_legend(self.arms_probs,[0.1, .1, .1], [0.8, 0.8, 0.8])

    def plot_probs_as_func_x_pos(self):
        """ Plot the probability of escaping through each path as a function of hor. position """
        l, c, r = [], [], []
        if self.session.processing['Origin probabilities'] is not None:
            for prob in self.session.processing['Escape probabilities'].per_x:
                l.append(prob.left)
                c.append(prob.central)
                r.append(prob.right)

            self.x_pos_arm_prob.bar([0.00, 1.00, 2.00, 3.00], l, width=0.25, color=self.colors[1], label='Left')
            self.x_pos_arm_prob.bar([0.25, 1.25, 2.25, 3.25], c, width=0.25, color=self.colors[2], label='Center')
            self.x_pos_arm_prob.bar([0.50, 1.50, 2.50, 3.50], r, width=0.25, color=self.colors[3], label='Right')

            n1 = [0, 1, 2, 3]
            [self.x_pos_arm_prob.axvline(x-.125, color='white', linewidth=0.5, label=None) for x in n1]
            [self.x_pos_arm_prob.axvline(x+.5+.125, color='white', linewidth=0.5, label=None) for x in n1]
        make_legend(self.arms_probs,[0.1, .1, .1], [0.8, 0.8, 0.8])
