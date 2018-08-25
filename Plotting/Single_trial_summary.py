import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('axes',edgecolor=[0.8, 0.8, 0.8])
matplotlib.rcParams['text.color'] = [0.8, 0.8, 0.8]
matplotlib.rcParams['axes.labelcolor'] = [0.8, 0.8, 0.8]
matplotlib.rcParams['axes.labelcolor'] = [0.8, 0.8, 0.8]
matplotlib.rcParams['xtick.color'] = [0.8, 0.8, 0.8]
matplotlib.rcParams['ytick.color'] = [0.8, 0.8, 0.8]

import numpy as np
import math
import os

from Utils.maths import line_smoother


class Plotter():
    def __init__(self, session):
        """
        This class plots all the data for a single trial:
            - std tracking
            - dlc tracking
            - velocity
            - orientation
        """
        plt.ion()
        if not session is None:

            self.plot_pose = True
            self.save_figs = True

            self.session = session

            # Define stuff used for extracting data
            self.sel_trial = 0
            self.prestim_frames = 5
            self.poststim_frames = 180

            # Define stuff used for plotting
            self.colors = {
                'Rear': [0.4, 0.8, 0.4],
                'Lear': [0.4, 0.8, 0.4],
                'snout': [0.5, 0.6, 0.5],
                'neck': [0.6, 0.4, 0.4],
                'body': [0.8, 0.3, 0.3],
                'tail': [0.4, 0.4, 0.8],
            }

            self.get_trials_data()
            self.main()
        else:
            plt.show()

########################################################################################################################
    def get_trials_data(self):
        tracking = self.session.Tracking

        trials_names = tracking.keys()
        self.trials = {}
        for trial in trials_names:
            self.trials[trial] = tracking[trial]

    def setup_figure(self):
        self.f = plt.figure(figsize=(35,15), facecolor=[0.1, 0.1, 0.1])

        grid = (6, 9)

        self.twod_track = plt.subplot2grid(grid, (0, 0), rowspan=4, colspan=3)
        self.twod_track.set(title='Tracking relative to shelter',
                            facecolor=[0.2, 0.2, 0.2], xlim=[300, -300], ylim=[-500, 100])


        self.std = plt.subplot2grid(grid, (0, 3), rowspan=1, colspan=2)
        self.std.set(title='DLC - X-Y displacement', facecolor=[0.2, 0.2, 0.2])

        self.dlc = plt.subplot2grid(grid, (2,  3), rowspan=1, colspan=2, sharex=self.std)
        self.dlc.set(title='STD - X-Y displacement', facecolor=[0.2, 0.2, 0.2])


        self.std_vel_plot = plt.subplot2grid(grid, (1, 3), rowspan=1, colspan=2, sharex=self.std)
        self.std_vel_plot.set(title='STD - Velocity ', facecolor=[0.2, 0.2, 0.2])

        self.dlc_vel_plot = plt.subplot2grid(grid, (3,  3), rowspan=1, colspan=2, sharex=self.std)
        self.dlc_vel_plot.set(title='DLC - Velocity', facecolor=[0.2, 0.2, 0.2])


        self.tracking_on_maze = plt.subplot2grid(grid, (0, 5), rowspan=2, colspan=2)
        self.tracking_on_maze.set(title='Tracking on maze', facecolor=[0.2, 0.2, 0.2], xlim=[0, 600], ylim=[600, 0])

        self.reaction_polar = plt.subplot2grid(grid, (2, 5), rowspan=2, colspan=2, projection='polar')
        self.reaction_polar.set(title='Reaction - orientation', theta_zero_location='N', facecolor=[0.2, 0.2, 0.2],
                                theta_direction=-1)

        self.pose = plt.subplot2grid(grid, (4, 0), rowspan=2, colspan=9)
        self.pose.set(title='Pose reconstruction', facecolor=[0.2, 0.2, 0.2], ylim=[600, -150])

        self.f.tight_layout()


        # Set up figure for trials
        # if self.sel_trial == 0:
        #     self.f_trials, self.tr_axarr = plt.subplots(len(self.trials), 1, figsize=(15, 10))

########################################################################################################################

    def get_tr_data_to_plot(self, trial):
        self.stim = int(len(trial.std_tracking) / 2)
        self.wnd = 600

        stim = self.stim
        wnd = self.wnd

        # STD
        self.std_x_adj = trial.std_tracking['adjusted x'].values[self.stim - wnd:stim + wnd]
        self.std_y_adj = trial.std_tracking['adjusted y'].values[stim - wnd:stim + wnd]
        self.std_x = trial.std_tracking['x'].values[stim - wnd:stim + wnd]
        self.std_y = trial.std_tracking['y'].values[stim - wnd:stim + wnd]
        self.std_vel = trial.std_tracking['Velocity'].values[stim - wnd:stim + wnd]

        # DLC
        for bp in trial.dlc_tracking['Posture'].keys():
            if bp == 'body':
                self.dlc_x_adj = trial.dlc_tracking['Posture'][bp]['adjusted x'].values[stim - wnd:stim + wnd]
                self.dlc_y_adj = trial.dlc_tracking['Posture'][bp]['adjusted y'].values[stim - wnd:stim + wnd]
                self.dlc_x = trial.dlc_tracking['Posture'][bp]['x'].values[stim - wnd:stim + wnd]
                self.dlc_y = trial.dlc_tracking['Posture'][bp]['y'].values[stim - wnd:stim + wnd]
                self.dlc_vel = trial.dlc_tracking['Posture'][bp]['Velocity'].values[stim - wnd:stim + wnd]
                self.dlc_ori = trial.dlc_tracking['Posture'][bp]['Orientation'].values[stim - wnd:stim + wnd]
                break

    def get_dlc_pose(self, trial, stim):
        frames = np.linspace(stim-self.prestim_frames, stim+self.poststim_frames,
                             self.prestim_frames+self.poststim_frames+1)

        poses = {}
        for frame in frames:
            pose = {}
            for bp in trial.dlc_tracking['Posture'].keys():
                pose[bp] = trial.dlc_tracking['Posture'][bp].loc[int(frame)]
                if bp == 'body':
                    pose['zero'] = trial.dlc_tracking['Posture'][bp].loc[int(frame)]['x']
            poses[str(frame)] = pose
        return poses

    def get_outcome(self, x, y,  window, ax):
        pre = x[0:window-1]
        post = x[window:-1]
        post_y = y[window:-1]

        # Get frame at which the mouse is the most distant from midline, ang get the X position at that frame
        pre_peak = pre[np.where(np.abs(pre)==np.max(np.abs(pre)))]
        post_peak = post[np.where(np.abs(post)==np.max(np.abs(post)))]

        self.at_shelter = np.where(post_y>30)[0][0]

        text_x, text_y, text_bg_col = -280, 75, [0.6, 0.6, 0.6]

        if pre_peak<0:
            ax.text(-text_x, text_y, 'Origin RIGHT', bbox={'facecolor':text_bg_col, 'alpha':0.5, 'pad':10})
        else:
            ax.text(-text_x, text_y, 'Origin LEFT', bbox={'facecolor':text_bg_col, 'alpha':0.5, 'pad':10})

        if post_peak<0:
            ax.text(-text_x, text_y-50, 'Escape RIGHT', bbox={'facecolor':text_bg_col, 'alpha':0.5, 'pad':10})
        else:
            ax.text(-text_x, text_y-50, 'Escape LEFT', bbox={'facecolor':text_bg_col, 'alpha':0.5, 'pad':10})

########################################################################################################################

    def plot_skeleton_time(self, poses, ax):
        x = np.linspace(1, 101 * (len(poses.keys()) / 2), len(poses.keys()) + 1)
        for idx, (fr, pose) in enumerate(sorted(poses.items())):
            fr = x[idx]
            if idx == self.prestim_frames-1:
                ax.axvline(fr, color='r', linewidth=3)

            # Plot the skeleton
            self.plot_skeleton_lines(ax, pose, self.colors, fr)

            # Plot the location of the bodyparts
            self.plot_skeleton_single_pose(pose, ax, shift=fr)

            # Mark the frame
            ax.axvline(fr, color=[0.6, 0.6, 0.6], linewidth=0.25)
        return x

    def plot_skeleton_space(self, poses, ax):
        for idx, (fr, pose) in enumerate(sorted(poses.items())):
            if idx<self.prestim_frames-1 or idx%10:
                continue

            # Plot the skeleton
            self.plot_skeleton_lines(ax, pose, self.colors, False)

            # Plot the location of the bodyparts
            self.plot_skeleton_single_pose(pose, ax)

    def plot_skeleton_single_pose(self, pose, ax, shift=False):
        for bpname, bp in pose.items():
            if bpname == 'zero':
                continue
            if not bpname in self.colors:
                if shift:
                    continue
                else:
                    ax.plot(bp['x'], bp['y'], 'o', markersize=7, color=[0.8, 0.8, 0.8])
            else:
                if shift:
                    ax.plot(bp['x'] + shift - pose['zero'], bp['y'], 'o', markersize=7, color=self.colors[bpname])
                else:
                    ax.plot(bp['x'], bp['y'], 'o', markersize=15, color=self.colors[bpname], alpha=0.5)

    def plot_skeleton_lines(self, ax, pose, colors, fr):
        def plot_line_skeleton(ax, p1, p2, pose, colors, shift):
            if shift:
                ax.plot([pose[p1]['x'] + shift - pose['zero'], pose[p2]['x'] + shift - pose['zero']],
                        [pose[p1]['y'], pose[p2]['y']],
                        color=colors[p1], linewidth=4)
            else:
                ax.plot([pose[p1]['x'], pose[p2]['x']], [pose[p1]['y'], pose[p2]['y']],
                        color=colors[p1], linewidth=6)

        # Plot body
        p1, p2 = 'body', 'tail'
        plot_line_skeleton(ax, p1, p2, pose, colors, fr)

        # Plot body-neck
        p1, p2 = 'neck', 'body'
        plot_line_skeleton(ax, p1, p2, pose, colors, fr)

        # Plot body-ears
        p1, p2 = 'neck', 'Rear'
        plot_line_skeleton(ax, p1, p2, pose, colors, fr)
        p1, p2 = 'neck', 'Lear'
        plot_line_skeleton(ax, p1, p2, pose, colors, fr)

        # Plot ears-nose
        p1, p2 = 'Lear', 'snout'
        plot_line_skeleton(ax, p1, p2, pose, colors, fr)
        p1, p2 = 'Rear', 'snout'
        plot_line_skeleton(ax, p1, p2, pose, colors, fr)

########################################################################################################################

    def plot_trial(self, trialidx):
        """
        plot stuff for a single trial
        """

        self.setup_figure()

        print('         ... plotting trial {} of {}: {}'.format(
            trialidx, len(list(self.trials.keys()))-1, list(self.trials.keys())[trialidx]))

        trial = self.trials[list(self.trials.keys())[trialidx]]

        # Get tracking data for plotting
        self.get_tr_data_to_plot(trial)

        # Get pose
        poses = self.get_dlc_pose(trial, self.stim)

        # Get outcome
        self.get_outcome(self.dlc_x_adj, self.dlc_x_adj, self.wnd, self.twod_track)

        # Plot 2D tracking from std and dlc data
        shelter = plt.Rectangle((-50,-30),100,60,linewidth=1,edgecolor='b',facecolor=[0.4, 0.4, 0.8], alpha=0.5)
        self.twod_track.add_artist(shelter)
        # self.twod_track.plot(std_x_adj, std_y_adj, '-o', color='g')
        self.twod_track.plot(self.dlc_x_adj[0:self.wnd], self.dlc_y_adj[0:self.wnd],
                             '-', color=[0.4, 0.4, 0.4], linewidth=2)
        self.twod_track.plot(self.dlc_x_adj[self.wnd:], self.dlc_y_adj[self.wnd:],
                             '-', color=[0.6, 0.6, 0.6], linewidth=4)
        self.twod_track.plot(self.dlc_x_adj[self.wnd], self.dlc_y_adj[self.wnd],
                             'o', color=[0.8, 0.2, 0.2], markersize=20, alpha = 0.75)

        # Plot 1D stuff for std [x, y, velocity...]
        self.std.plot(self.std_x_adj, color=[0.2, 0.8, 0.2], linewidth=3)
        self.std.plot(self.std_y_adj, color=[0.2, 0.6, 0.2], linewidth=3)
        self.std.axvline(self.wnd, color='w', linewidth=1)

        self.std_vel_plot.plot(self.std_vel, color=[0.4, 0.4, 0.4], linewidth=3)
        self.std_vel_plot.plot(line_smoother(self.std_vel, 51, 3), color=[0.4, 0.8, 0.4], linewidth=5)
        self.std_vel_plot.axvline(self.wnd, color='w', linewidth=1)

        # Plot 1D stuff for dlc [x, y, velocity...]
        self.dlc.plot(self.dlc_x_adj, color=[0.8, 0.2, 0.2], linewidth=3)
        self.dlc.plot(self.dlc_y_adj, color=[0.6, 0.2, 0.2], linewidth=3)
        self.dlc.axvline(self.wnd, color='w', linewidth=1)

        self.dlc_vel_plot.plot(self.dlc_vel, color=[0.4, 0.4, 0.4], linewidth=5)
        self.dlc_vel_plot.plot(line_smoother(self.dlc_vel, 51, 3), color=[0.8, 0.4, 0.4], linewidth=3)
        self.dlc_vel_plot.axvline(self.wnd, color='w', linewidth=1)

        # Show mouse tracking over the maze
        self.tracking_on_maze.imshow(self.session.Metadata.videodata[0]['Background'], cmap='Greys')
        self.tracking_on_maze.plot(self.dlc_x, self.dlc_y, '-', color='r')

        # Plot orientation at reaction
        yy = np.linspace(self.prestim_frames, self.poststim_frames, self.prestim_frames+self.poststim_frames)
        theta = self.dlc_ori[self.wnd-self.prestim_frames:self.wnd+self.poststim_frames]
        theta = [math.radians(x) for x in theta]
        self.reaction_polar.scatter(theta, yy, c=yy, cmap='Oranges', s=50)

        # Show pose reconstruction
        if self.plot_pose:
            # TIME pose reconstruction
            self.pose.axhline(0, color='w', linewidth=1, alpha=0.75)
            x = self.plot_skeleton_time(poses, self.pose)

            # Plot stuff over pose reconstruction
            self.pose.fill_between(x, 0, self.std_x_adj[self.wnd-5:self.wnd+self.poststim_frames+2],
                                   facecolor=[0.8, 0.8, 0.8], alpha=0.5)
            self.pose.fill_between(x, 0, self.std_vel[self.wnd-5:self.wnd+self.poststim_frames+2]*10,
                                   facecolor=[0.6, 0.6, 0.6], alpha=0.5)

        if self.save_figs:
            path = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\data\\dlc_trialImgs'
            name = '{}'.format(list(self.trials.keys())[self.sel_trial])
            plt.savefig(os.path.join(path, name), facecolor=[0.1, 0.1, 0.1])
            plt.close('all')


########################################################################################################################

    def main(self):
        """
        Loop, plot stuff and allow user to select other trials

        :return:
        """
        num_trials = len(list(self.trials.keys()))
        print('         ... found {} trials, displaying the first one: {}'.format(
            num_trials, list(self.trials.keys())[0]))

        while True:
            try:
                self.plot_trial(self.sel_trial)
            except:
                break

            self.sel_trial += 1
            if self.sel_trial > num_trials:
                print('         ... displayed all trials.')
                break

        plt.show()
