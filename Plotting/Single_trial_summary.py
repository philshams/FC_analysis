import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('axes',edgecolor=[0.8, 0.8, 0.8])
matplotlib.rcParams['text.color'] = [0.8, 0.8, 0.8]
matplotlib.rcParams['axes.labelcolor'] = [0.8, 0.8, 0.8]
matplotlib.rcParams['axes.labelcolor'] = [0.8, 0.8, 0.8]
matplotlib.rcParams['xtick.color'] = [0.8, 0.8, 0.8]
matplotlib.rcParams['ytick.color'] = [0.8, 0.8, 0.8]
params = {'legend.fontsize': 16,
          'legend.handlelength': 2}
plt.rcParams.update(params)

import numpy as np
import math
import os

from Utils.maths import line_smoother
from Utils.Messaging import slack_chat_messenger, slack_chat_attachments

from Config import use_slack


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

            self.exploration_maxmin = 10

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

        self.twod_track = plt.subplot2grid(grid, (0, 1), rowspan=3, colspan=2)
        self.twod_track.set(title='Tracking relative to shelter',
                            facecolor=[0.2, 0.2, 0.2], xlim=[300, -300], ylim=[-500, 100])

        self.std = plt.subplot2grid(grid, (0, 3), rowspan=1, colspan=2)
        self.std.set(title='STD - X-Y displacement', facecolor=[0.2, 0.2, 0.2])

        self.dlc = plt.subplot2grid(grid, (2,  3), rowspan=1, colspan=2, sharex=self.std)
        self.dlc.set(title='STD - X-Y displacement', facecolor=[0.2, 0.2, 0.2])

        self.std_vel_plot = plt.subplot2grid(grid, (1, 3), rowspan=1, colspan=2, sharex=self.std)
        self.std_vel_plot.set(title='DLC - Velocity ', facecolor=[0.2, 0.2, 0.2])

        self.dlc_vel_plot = plt.subplot2grid(grid, (3,  3), rowspan=1, colspan=2, sharex=self.std)
        self.dlc_vel_plot.set(title='DLC - Velocity', facecolor=[0.2, 0.2, 0.2])

        self.tracking_on_maze = plt.subplot2grid(grid, (0, 0), rowspan=1, colspan=1)
        self.tracking_on_maze.set(title='Tracking on maze', facecolor=[0.2, 0.2, 0.2], xlim=[0, 600], ylim=[600, 0])

        self.absolute_angle_plot = plt.subplot2grid(grid, (0, 5), rowspan=2, colspan=2, projection='polar')
        self.absolute_angle_plot.set(title='Orientation (body green)', theta_zero_location='N', facecolor=[0.2, 0.2, 0.2],
                                     theta_direction=-1)

        self.pose = plt.subplot2grid(grid, (4, 0), rowspan=2, colspan=9)
        self.pose.set(title='Pose reconstruction', facecolor=[0.2, 0.2, 0.2], ylim=[635, -150])

        self.pose_space = plt.subplot2grid(grid, (2, 0), rowspan=1, colspan=1)
        self.pose_space.set(title='Pose at stim', facecolor=[0.2, 0.2, 0.2], xlim=[150, 450], ylim=[650, 350])

        self.exploration_plot = plt.subplot2grid(grid, (1, 0), rowspan=1, colspan=1)
        self.exploration_plot.set(title='Eploration', facecolor=[0.2, 0.2, 0.2], xlim=[0, 600], ylim=[600, 0])

        self.react_time_plot =  plt.subplot2grid(grid, (3, 0), rowspan=1, colspan=3)
        self.react_time_plot.set(title='Reaction Time', facecolor=[0.2, 0.2, 0.2])

        self.head_rel_angle = plt.subplot2grid(grid, (2, 5), rowspan=2, colspan=2, projection='polar')
        self.head_rel_angle.set(title='Head Relative Angle', theta_zero_location='N', facecolor=[0.2, 0.2, 0.2],
                                theta_direction=-1)

        self.f.tight_layout()

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
                self.dlc_head_ori = trial.dlc_tracking['Posture'][bp]['Head angle'].values[stim - wnd:stim + wnd]
                self.dlc_bodylength = trial.dlc_tracking['Posture'][bp]['Body length'].values[stim - wnd:stim + wnd]
                break
        avgbdlength = trial.metadata['avg body length']
        self.dlc_bodylength = np.array([x/avgbdlength for x in self.dlc_bodylength])

        # Exploration
        fps = self.session.Metadata.videodata[0]['Frame rate'][0]
        exploration_maxfr = int(self.exploration_maxmin*60*fps)
        expl_len = int(len(self.session.Tracking['Exploration']))

        if expl_len>exploration_maxfr:
            self.exp_heatmap = True
            self.exploration = self.session.Tracking['Exploration'][expl_len-exploration_maxfr:]
        else:
            self.exp_heatmap = False
            self.exploration = self.session.Tracking['Exploration']

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

        self.post_y = y[window:-1]
        self.post_vel = self.dlc_vel[window:-1]
        self.post_ori = self.dlc_ori[window:-1]
        self.post_bl = self.dlc_bodylength[window:-1]

        self.mean_pre_xvel, self.sdev_pre_xacc = np.mean(np.diff(x[0:window - 31])), np.std(np.diff(x[0:window - 31]))
        self.mean_pre_yvel, self.sdev_pre_yacc = np.mean(np.diff(y[0:window - 31])), np.std(np.diff(y[0:window - 31]))
        self.mean_pre_vel, self.sdev_pre_vel = np.mean(self.dlc_vel[0:window-31]), np.std(self.dlc_vel[0:window-31])
        self.mean_pre_bl, self.sdev_pre_bl = np.mean(self.dlc_bodylength[0:window-31]),np.std(self.dlc_bodylength[0:window-31])

        # Get frame at which the mouse is the most distant from midline, ang get the X position at that frame
        pre_peak = pre[np.where(np.abs(pre)==np.max(np.abs(pre)))]
        post_peak = post[np.where(np.abs(post)==np.max(np.abs(post)))]

        # Get position and orientation at time of stimulus
        x_stim, y_stim, ori_stim, vel_stim, bodylenfth_stim = self.dlc_x_adj[window], self.dlc_y_adj[window],\
                                                              self.dlc_ori[window],\
                                                              self.dlc_vel[window], self.dlc_bodylength[window]

        # Get REACTION TIME
        # Get point of max Y distance from shelet
        self.y_diff = np.diff(self.post_y)
        self.x_diff = np.diff(post)

        try:
            self.at_shelter = np.where(self.post_y>0)[0][0]
        except:
            self.at_shelter = len(self.post_y)

        # Adjust ax limits and mark time in which mouse reached shelter
        self.head_rel_angle.set(ylim=[0, self.at_shelter])
        self.absolute_angle_plot.set(ylim=[0, self.at_shelter])
        self.std.axvline(self.at_shelter+window, color=[0.8, 0.2, 0.8], linewidth=1, label=None)
        self.std_vel_plot.axvline(self.at_shelter+window, color=[0.8, 0.2, 0.8], linewidth=1, label=None)
        self.dlc.axvline(self.at_shelter+window, color=[0.8, 0.2, 0.8], linewidth=1, label=None)
        self.dlc_vel_plot.axvline(self.at_shelter+window, color=[0.8, 0.2, 0.8], linewidth=1, label=None)
        self.react_time_plot.axvline(self.at_shelter, color=[0.8, 0.2, 0.8], linewidth=1, label=None)
        self.twod_track.plot(self.dlc_x_adj[self.wnd+self.at_shelter], self.dlc_y_adj[self.wnd+self.at_shelter],
                             'o', color=[0.8, 0.2, 0.8], markersize=20, alpha=0.75, label='At shelter')
        # Show the results
        text_x, text_y, text_bg_col = -280, 75, [0.1, 0.1, 0.1]

        if pre_peak<0:
            ax.text(-text_x, text_y, 'Origin RIGHT', bbox={'facecolor':text_bg_col, 'alpha':0.5, 'pad':10})
        else:
            ax.text(-text_x, text_y, 'Origin LEFT', bbox={'facecolor':text_bg_col, 'alpha':0.5, 'pad':10})

        if post_peak<0:
            ax.text(-text_x, text_y-50, 'Escape RIGHT', bbox={'facecolor':text_bg_col, 'alpha':0.5, 'pad':10})
        else:
            ax.text(-text_x, text_y-50, 'Escape LEFT', bbox={'facecolor':text_bg_col, 'alpha':0.5, 'pad':10})

        ax.text(-text_x, text_y - 100, 'Stim X: {}'.format(round(x_stim, 2)),
                bbox={'facecolor': text_bg_col, 'alpha': 0.5, 'pad': 10})

        ax.text(-text_x, text_y - 150, 'Stim Y: {}'.format(round(y_stim, 2)),
                bbox={'facecolor': text_bg_col, 'alpha': 0.5, 'pad': 10})

        ax.text(-text_x, text_y - 200, 'Stim Ori: {}'.format(round(360+ori_stim, 2)),
                bbox={'facecolor': text_bg_col, 'alpha': 0.5, 'pad': 10})

        ax.text(-text_x, text_y - 250, 'Stim Vel: {}'.format(round(vel_stim, 2)),
                bbox={'facecolor': text_bg_col, 'alpha': 0.5, 'pad': 10})

        ax.text(-text_x, text_y - 300, 'Stim BL: {}'.format(round(bodylenfth_stim, 2)),
                bbox={'facecolor': text_bg_col, 'alpha': 0.5, 'pad': 10})

        ax.text(-text_x, text_y - 350, 'At shelt: {}'.format(round(self.at_shelter, 2)),
                bbox={'facecolor': text_bg_col, 'alpha': 0.5, 'pad': 10})

########################################################################################################################

    def plot_skeleton_time(self, poses, ax):
        x = np.linspace(1, 101 * (len(poses.keys()) / 2), len(poses.keys()) + 1)
        for idx, (fr, pose) in enumerate(sorted(poses.items())):
            fr = x[idx]
            # Mark the frame
            if idx == self.prestim_frames-1:
                ax.axvline(fr, color='r', linewidth=3)
                # Plot pose over maze edges at react time
                self.plot_skeleton_single_pose(pose, self.pose_space)
                self.plot_skeleton_lines(self.pose_space, pose, self.colors, False)

                maze_outline = self.session.Metadata.videodata[0]['Maze Edges']
                self.pose_space.imshow(maze_outline, cmap='gray')

            elif not (idx+self.prestim_frames+1)%10:
                ax.axvline(fr, color=[0.4, 0.4, 0.4], linewidth=2)
                ax.text(fr-20, 600, '{}'.format(idx-self.prestim_frames+1),
                        bbox={'facecolor': [0.1, 0.1, 0.1], 'alpha': 0.5, 'pad': 10})

            elif (idx-self.prestim_frames) == self.at_shelter:
                ax.axvline(fr, color=[0.8, 0.2, 0.8], linewidth=3, label=None)

            else:
                ax.axvline(fr, color=[0.6, 0.6, 0.6], linewidth=0.25)

            # Plot the skeleton
            self.plot_skeleton_lines(ax, pose, self.colors, fr)

            # Plot the location of the bodyparts
            self.plot_skeleton_single_pose(pose, ax, shift=fr)
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
                    ax.plot(bp['x'], bp['y'], 'o', markersize=5, color=[0.8, 0.8, 0.8], label=None)
            else:
                if shift:
                    # Plot bodyparts as points in the pose-time plot
                    ax.plot(bp['x'] + shift - pose['zero'], bp['y'], 'o', markersize=6, color=self.colors[bpname],
                            label=None)
                else:
                    # Plot bodyparts as points in the pose-reaction (space) plot
                    ax.plot(bp['x'], bp['y'], 'o', markersize=7, color=self.colors[bpname], alpha=0.5, label=None)

    def plot_skeleton_lines(self, ax, pose, colors, fr):
        def plot_line_skeleton(ax, p1, p2, pose, colors, shift):
            if shift:
                ax.plot([pose[p1]['x'] + shift - pose['zero'], pose[p2]['x'] + shift - pose['zero']],
                        [pose[p1]['y'], pose[p2]['y']],
                        color=colors[p1], linewidth=4, label=None)
            else:
                ax.plot([pose[p1]['x'], pose[p2]['x']], [pose[p1]['y'], pose[p2]['y']],
                        color=colors[p1], linewidth=2, label=None)

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

    def make_legend(self, ax, c1, c2, changefont=False):
        if not changefont:
            legend = ax.legend(frameon=True)
        else:
            legend = ax.legend(frameon=True, prop={'size': changefont})

        frame = legend.get_frame()
        frame.set_facecolor(c1)
        frame.set_edgecolor(c2)

    def plot_trial(self, trialidx):
        """
        plot stuff for a single trial
        """

        trialname = list(self.trials.keys())[trialidx]
        if 'Exploration' in trialname or 'Whole Session' in trialname:
            return

        self.setup_figure()

        print('         ... plotting trial {} of {}: {}'.format(
            trialidx, len(list(self.trials.keys()))-2, trialname))

        trial = self.trials[list(self.trials.keys())[trialidx]]

        # Get tracking data for plotting
        self.get_tr_data_to_plot(trial)

        # Get pose
        poses = self.get_dlc_pose(trial, self.stim)

        # Get outcome
        self.get_outcome(self.dlc_x_adj, self.dlc_y_adj, self.wnd, self.twod_track)

        # Plot 2D tracking from std and dlc data
        shelter = plt.Rectangle((-50,-30),100,60,linewidth=1,edgecolor='b',facecolor=[0.4, 0.4, 0.8], alpha=0.5)
        self.twod_track.add_artist(shelter)
        # self.twod_track.plot(std_x_adj, std_y_adj, '-o', color='g')
        self.twod_track.plot(self.dlc_x_adj[0:self.wnd], self.dlc_y_adj[0:self.wnd],
                             '-', color=[0.4, 0.4, 0.4], linewidth=2, label='Outward')
        self.twod_track.plot(self.dlc_x_adj[self.wnd:], self.dlc_y_adj[self.wnd:],
                             '-', color=[0.6, 0.6, 0.6], linewidth=4, label='Escape')
        self.twod_track.plot(self.dlc_x_adj[self.wnd], self.dlc_y_adj[self.wnd],
                             'o', color=[0.8, 0.2, 0.2], markersize=20, alpha = 0.75, label='Stim location')
        self.make_legend(self.twod_track, [0.1, .1, .1], [0.8, 0.8, 0.8])

        # Plot 1D stuff for std [x, y, velocity...]
        self.std.plot(self.std_x_adj, color=[0.2, 0.8, 0.2], linewidth=3, label='X pos')
        self.std.plot(self.std_y_adj, color=[0.2, 0.6, 0.2], linewidth=3, label='Y pos')
        self.std.axvline(self.wnd, color='w', linewidth=1, label=None)
        self.make_legend(self.std, [0.1, .1, .1], [0.8, 0.8, 0.8])

        self.std_vel_plot.plot(self.std_vel, color=[0.4, 0.4, 0.4], linewidth=3, label='Vel')
        self.std_vel_plot.plot(line_smoother(self.std_vel, 51, 3), color=[0.4, 0.8, 0.4], linewidth=5,
                               label='Smoothed')
        self.std_vel_plot.axvline(self.wnd, color='w', linewidth=1, label=None)
        self.make_legend(self.std_vel_plot, [0.1, .1, .1], [0.8, 0.8, 0.8])

        # Plot 1D stuff for dlc [x, y, velocity...]
        self.dlc.plot(self.dlc_x_adj, color=[0.8, 0.2, 0.2], linewidth=3, label='X pos')
        self.dlc.plot(self.dlc_y_adj, color=[0.6, 0.2, 0.2], linewidth=3, label='Y pos')
        self.dlc.axvline(self.wnd, color='w', linewidth=1, label=None)
        self.make_legend(self.dlc, [0.1, .1, .1], [0.8, 0.8, 0.8])

        self.dlc_vel_plot.plot(self.dlc_vel, color=[0.4, 0.4, 0.4], linewidth=5, label='Vel')
        self.dlc_vel_plot.plot(line_smoother(self.dlc_vel, 51, 3), color=[0.8, 0.4, 0.4], linewidth=3,
                               label='Smoothed')
        self.dlc_vel_plot.axvline(self.wnd, color='w', linewidth=1, label=None)
        self.make_legend(self.dlc_vel_plot, [0.1, .1, .1], [0.8, 0.8, 0.8])

        # Show mouse tracking over the maze
        self.tracking_on_maze.imshow(self.session.Metadata.videodata[0]['Background'], cmap='Greys')
        self.tracking_on_maze.plot(self.dlc_x, self.dlc_y, '-', color='r')

        # Show exploration heatmap
        cmap = plt.cm.bone
        if self.exp_heatmap:
            self.exploration_plot.hexbin(self.exploration['x'].values, self.exploration['y'].values,
                                         bins='log', gridsize=50, cmap=cmap)
        else:
            pass
            # self.exploration_plot.plot(self.exploration['x'].values, self.exploration['y'].values,
            #                            color=[0.6, 0.6, 0.6], linewidth=3)

        # Plot orientation at reaction
        yy = np.linspace(self.prestim_frames, self.poststim_frames, self.prestim_frames+self.poststim_frames)
        theta = self.dlc_ori[self.wnd-self.prestim_frames:self.wnd+self.poststim_frames]
        head_theta = self.dlc_head_ori[self.wnd-self.prestim_frames:self.wnd+self.poststim_frames]

        theta = np.array([math.radians(x) for x in theta])
        head_theta = np.array([math.radians(x) for x in head_theta])
        head_rel_angle = head_theta-theta

        self.absolute_angle_plot.scatter(theta, yy, c=yy, cmap='Oranges', s=50, alpha=0.5, label='Body')
        self.absolute_angle_plot.scatter(head_theta, yy, c=yy, cmap='Greens', s=50, alpha=0.5, label='Head')
        self.head_rel_angle.scatter(head_rel_angle, yy, c=yy, cmap='Paired', s=50, alpha=0.5)

        try:
            # Show pose reconstruction
            if self.plot_pose:
                # TIME pose reconstruction
                self.pose.axhline(0, color='w', linewidth=1, alpha=0.75)
                x = self.plot_skeleton_time(poses, self.pose)

                # Plot stuff over pose reconstruction
                self.pose.fill_between(x, 0, self.std_x_adj[self.wnd-5:self.wnd+self.poststim_frames+2],
                                       facecolor=[0.8, 0.8, 0.8], alpha=0.5, label='X position')
                self.pose.fill_between(x, 0, self.std_vel[self.wnd-5:self.wnd+self.poststim_frames+2]*10,
                                       facecolor=[0.8, 0.4, 0.4], alpha=0.5, label='Velocity')

                self.pose.plot(x, self.dlc_bodylength[self.wnd - 5:self.wnd + self.poststim_frames + 2] * 100,
                                       color=[0.4, 0.4, 0.8], alpha=0.5, linewidth=4, label='Body length')

                self.make_legend(self.pose, [0.1, .1, .1], [0.8, 0.8, 0.8])

        except:
            pass

        # Plot Reaction Time stuff
        self.react_time_plot.plot(self.y_diff, color=[0.6, 0.2, 0.2], linewidth=3, label='Y vel')
        self.react_time_plot.axhline(self.mean_pre_yvel, color=[0.6, 0.2, 0.2], linewidth=1, label=None)

        self.react_time_plot.plot(self.x_diff, color=[0.8, 0.2, 0.2], linewidth=3, label='X vel')
        self.react_time_plot.axhline(self.mean_pre_xvel, color=[0.8, 0.2, 0.2], linewidth=1, label=None)

        self.react_time_plot.plot(self.post_vel, color=[0.6, 0.6, 0.6], linewidth=3, label='Vel')
        self.react_time_plot.axhline(self.mean_pre_vel, color=[0.6, 0.6, 0.6], linewidth=1, label=None)

        self.react_time_plot.plot(self.post_bl * 10, color=[0.4, 0.4, 0.8], linewidth=3, label='BL')
        self.react_time_plot.axhline(self.mean_pre_bl * 10, color=[0.4, 0.4, 0.8], linewidth=1, label=None)

        self.react_time_plot.axhline(0, color='w', linewidth=1, label=None)
        self.react_time_plot.set(xlim=[0, self.at_shelter + 120], ylim=[-20, 20])
        self.make_legend(self.react_time_plot, [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=8)

        if self.save_figs:
            path = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\data\\dlc_trialImgs'
            name = '{}'.format(list(self.trials.keys())[self.sel_trial])
            print('             ... saving figure {}'.format(name))
            plt.savefig(os.path.join(path, name), facecolor=[0.1, 0.1, 0.1])

            if use_slack:
                slack_chat_attachments(os.path.join(path, name))

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
