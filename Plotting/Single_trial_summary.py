import matplotlib.pyplot as plt
import numpy as np

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
            self.session = session
            self.sel_trial = 0

            self.get_trials_data()
            self.main()
        else:
            plt.show()

    def get_trials_data(self):
        tracking = self.session.Tracking

        trials_names = tracking.keys()
        self.trials = {}
        for trial in trials_names:
            self.trials[trial] = tracking[trial]

    def setup_figure(self):
        self.f = plt.figure(figsize=(35,15))

        grid = (6, 7)

        self.twod_track = plt.subplot2grid(grid, (0, 0), rowspan=4, colspan=4)
        self.twod_track.set(title='Tracking relative to shelter', facecolor=[0.2, 0.2, 0.2], xlim=[-300, 300], ylim=[-500, 100])


        self.std = plt.subplot2grid(grid, (0, 4), rowspan=1, colspan=2)
        self.std.set(title='DLC - X-Y displacement', facecolor=[0.2, 0.2, 0.2])

        self.dlc = plt.subplot2grid(grid, (2,  4), rowspan=1, colspan=2, sharex=self.std)
        self.dlc.set(title='STD - X-Y displacement', facecolor=[0.2, 0.2, 0.2])


        self.std_vel = plt.subplot2grid(grid, (1, 4), rowspan=1, colspan=2, sharex=self.std)
        self.std_vel.set(title='STD - Velocity ', facecolor=[0.2, 0.2, 0.2])

        self.dlc_vel = plt.subplot2grid(grid, (3,  4), rowspan=1, colspan=2, sharex=self.std)
        self.dlc_vel.set(title='DLC - Velocity', facecolor=[0.2, 0.2, 0.2])


        self.tracking_on_maze = plt.subplot2grid(grid, (0, 6), rowspan=1, colspan=1)
        self.tracking_on_maze.set(title='Tracking on maze', facecolor=[0.2, 0.2, 0.2], xlim=[0, 600], ylim=[0, 600])

        self.reaction_polar = plt.subplot2grid(grid, (1, 6), rowspan=1, colspan=1, projection='polar')
        self.reaction_polar.set(title='Reaction - orientation', theta_zero_location='N', facecolor=[0.2, 0.2, 0.2])

        self.pose = plt.subplot2grid(grid, (4, 0), rowspan=2, colspan=7)
        self.pose.set(title='Pose reconstruction', facecolor=[0.2, 0.2, 0.2])

    def get_dlc_pose(self, trial, stim, everyframe=5):
        frames = np.linspace(stim-everyframe, stim+30, int(30/everyframe))

        poses = {}
        for frame in frames:
            pose = {}
            for bp in trial.dlc_tracking['Posture'].keys():
                pose[bp] = trial.dlc_tracking['Posture'][bp].loc[int(frame)]
                if bp == 'body':
                    bp['zero'] = trial.dlc_tracking['Posture'][bp].loc[int(frame)]['x']
            poses[str(frame)] = pose
        return poses


    def plot_trial(self, trialidx):
        """
        plot stuff for a single trial
        """

        print('         ... plotting trial {} of {}: {}'.format(
            trialidx, len(list(self.trials.keys())), list(self.trials.keys())[trialidx]))

        trial = self.trials[list(self.trials.keys())[trialidx]]
        stim = int(len(trial.std_tracking)/2)
        wnd = 600

        # Prep data
        std_x_adj = trial.std_tracking['adjusted x'].values[stim-wnd:stim+wnd]
        std_y_adj = trial.std_tracking['adjusted y'].values[stim-wnd:stim+wnd]
        std_x = trial.std_tracking['x'].values[stim-wnd:stim+wnd]
        std_y = trial.std_tracking['y'].values[stim-wnd:stim+wnd]
        std_vel = trial.std_tracking['Velocity'].values[stim-wnd:stim+wnd]

        for bp in trial.dlc_tracking['Posture'].keys():
            if bp == 'body':
                dlc_x_adj = trial.dlc_tracking['Posture'][bp]['adjusted x'].values[stim-wnd:stim+wnd]
                dlc_y_adj = trial.dlc_tracking['Posture'][bp]['adjusted y'].values[stim - wnd:stim + wnd]
                dlc_x = trial.dlc_tracking['Posture'][bp]['x'].values[stim-wnd:stim+wnd]
                dlc_y = trial.dlc_tracking['Posture'][bp]['y'].values[stim - wnd:stim + wnd]
                dlc_vel = trial.dlc_tracking['Posture'][bp]['Velocity'].values[stim - wnd:stim + wnd]
                break

        # Get pose
        poses = self.get_dlc_pose(trial, stim)

        # Plot 2D tracking from std and dlc data
        circle1 = plt.Circle((0, 0), 10, color='m')
        self.twod_track.add_artist(circle1)
        # self.twod_track.plot(std_x_adj, std_y_adj, '-o', color='g')
        self.twod_track.plot(dlc_x_adj[0:wnd], dlc_y_adj[0:wnd], '-x', color=[0.4, 0.2, 0.2], linewidth=2)
        self.twod_track.plot(dlc_x_adj[wnd:], dlc_y_adj[wnd:], '-x', color=[0.8, 0.2, 0.2], linewidth=4)
        self.twod_track.plot(dlc_x_adj[wnd], dlc_y_adj[wnd], 'o', color=[0.8, 0.2, 0.2], markersize=20, alpha = 0.75)

        # Plot 1D stuff for std [x, y, velocity...]
        self.std.plot(std_x_adj, color=[0.2, 0.8, 0.2], linewidth=3)
        self.std.plot(std_y_adj, color=[0.2, 0.6, 0.2], linewidth=3)
        self.std.axvline(wnd, color='w', linewidth=1)

        self.std_vel.plot(std_vel, color=[0.4, 0.4, 0.4], linewidth=3)
        self.std_vel.plot(line_smoother(std_vel, 51, 3), color=[0.4, 0.8, 0.4], linewidth=5)
        self.std_vel.axvline(wnd, color='w', linewidth=1)

        # Plot 1D stuff for dlc [x, y, velocity...]
        self.dlc.plot(dlc_x_adj, color=[0.8, 0.2, 0.2], linewidth=3)
        self.dlc.plot(dlc_y_adj, color=[0.6, 0.2, 0.2], linewidth=3)
        self.dlc.axvline(wnd, color='w', linewidth=1)

        self.dlc_vel.plot(dlc_vel, color=[0.4, 0.4, 0.4], linewidth=5)
        self.dlc_vel.plot(line_smoother(dlc_vel, 51, 3), color=[0.8, 0.4, 0.4], linewidth=3)
        self.dlc_vel.axvline(wnd, color='w', linewidth=1)

        # Show mouse tracking over the maze
        self.tracking_on_maze.imshow(self.session.Metadata.videodata[0]['Background'])
        self.tracking_on_maze.plot(dlc_x, dlc_y, '-x', color='r')

        # Show pose reconstruction
        for fr, pose in poses.items():
            fr = int(float(fr))
            for bpname, bp in pose.items():
                if bpname == 'zero':
                    continue

                self.pose.plot(bp['x']+fr-bp['zero'], bp['y'], 'o', markersize=15)
                self.pose.axvline(fr, color='w', linewidth=1)

    def main(self):
        """
        Loop, plot stuff and allow user to select other trials

        :return:
        """
        num_trials = len(list(self.trials.keys()))
        print('         ... found {} trials, displaying the first one: {}'.format(
            num_trials, list(self.trials.keys())[0]))

        while True:
            self.setup_figure()
            try:
                self.plot_trial(self.sel_trial)
            except:
                break

            plt.show()
            q = input('\n\nNext [n] or prev [m] trial or EXIT [e]')
            if q == 'm':
                self.sel_trial -= 1
                if self.sel_trial < 0:
                    self.sel_trial = 0
            elif q == 'n':
                self.sel_trial += 1
                if self.sel_trial > num_trials:
                    print('         ... displayed all trials.')
                    break
            elif q == 'e':
                break




if __name__=="__main__":
    Plotter(None)