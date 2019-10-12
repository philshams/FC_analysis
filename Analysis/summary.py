import cv2
import numpy as np
import os
import pickle
import imageio
import pandas as pd
from sklearn import linear_model
from sklearn.decomposition import PCA
import scipy.stats
from math import factorial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
plt.rcParams.update({'font.size': 18})
import seaborn as sb
from scipy.ndimage import gaussian_filter1d
from Utils.registration_funcs import get_arena_details, model_arena
import tqdm


def summary_plots(save_folder):
    '''
    GET POPULATION SPEED TRACES AND THE LIKE
    '''


    '''
    INITIALIZE VARIABLES
    '''
    #initialize variables
    save_file = os.path.join(save_folder, 'proper_analysis')

    # load and initialize dictionary
    with open(save_file, 'rb') as dill_file:
        analysis_dictionary = pickle.load(dill_file)


    '''
    LOOP ACROSS ALL EXPERIMENTS AND CONDITIONS
    '''
    experiments = ['Circle wall down (dark)']
    experiments = ['Circle wall down', 'Circle wall up']
    experiments = ['Circle wall down', 'Circle wall down (no baseline)', 'Circle wall down (no shelter)']
    experiments = []
    # experiments = ['Circle void (shelter on side)', 'Circle wall (shelter on side)']
    # for experiment in analysis_dictionary:
    for experiment in experiments:
        for condition in analysis_dictionary[experiment]:
            '''
            AVERAGE THE EXPLORATION PLOTS PER CONDITION
            '''

            # initialize array to fill in with each mouse's data
            exploration_shape = analysis_dictionary[experiment]['obstacle']['shape']
            _, _, obstacle_type, _, _, _ = get_arena_details(experiment)
            arena, _, shelter_roi = model_arena(exploration_shape, True, False, obstacle_type, simulate=True)
            exploration_image = np.zeros(exploration_shape)
            bin_size = analysis_dictionary[experiment]['obstacle']['scale']
            bins_per_side = int(exploration_shape[0] / bin_size)
            exploration = np.zeros((bins_per_side, bins_per_side, len(analysis_dictionary[experiment][condition]['exploration'])))

            # fill array with each mouse's data
            for i, mouse in enumerate(analysis_dictionary[experiment][condition]['exploration']):
                exploration[:,:,i] = analysis_dictionary[experiment][condition]['exploration'][mouse]

            if not exploration.size: continue

            # average all mice data
            exploration_all = np.mean(exploration, 2)

            # gaussian blur
            exploration_blur = cv2.GaussianBlur(exploration_all, ksize=(201, 201), sigmaX=11, sigmaY=11)

            # normalize
            exploration_blur = (exploration_blur / np.percentile(exploration_blur, 93) * 255)
            exploration_blur[exploration_blur > 255] = 255
            exploration_all = (exploration_all / np.percentile(exploration_all, 98) * 255)
            exploration_all[exploration_all > 255] = 255

            # change color map
            exploration_blur = cv2.applyColorMap(exploration_blur.astype(np.uint8), cv2.COLORMAP_OCEAN)
            exploration_all = cv2.cvtColor(exploration_all.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            # exploration_blur[arena == 0] = 0

            # make composite image
            exploration_image = ((exploration_all.astype(float)+exploration_blur.astype(float))/2)
            exploration_image[exploration_image > 255] = 255
            exploration_image = exploration_image.astype(np.uint8)
            cv2.imshow('heat map', exploration_image)

            # save results
            experiment_save_folder = os.path.join(save_folder, experiment)
            if not os.path.isdir(experiment_save_folder):
                os.makedirs(experiment_save_folder)

            imageio.imwrite(os.path.join(experiment_save_folder, experiment + '_exploration_' + condition + '.tif'), exploration_image[:,:,::-1])

            '''
            AVERAGE THE DIRECTIONALITY PLOTS PER CONDITION
            '''

            # initialize array to fill in with each mouse's data
            bins_per_side = analysis_dictionary[experiment]['obstacle']['direction scale']
            scale = int(analysis_dictionary[experiment]['obstacle']['shape'][0] / bins_per_side)
            direction = np.zeros((bins_per_side**2, bins_per_side**2, len(analysis_dictionary[experiment][condition]['direction'])))

            # fill array with each mouse's data
            for i, mouse in enumerate(analysis_dictionary[experiment][condition]['direction']):
                direction[:, :, i] = analysis_dictionary[experiment][condition]['direction'][mouse]

            # average all mice data
            direction_all = np.mean(direction, 2)

            # visualize the directionality
            direction_image = np.zeros(analysis_dictionary[experiment]['obstacle']['shape'], np.uint8)
            for x in range(bins_per_side):
                for y in range(bins_per_side):

                    from_idx = x + y * bins_per_side

                    origin = int((x + .5) * scale), int((y + .5) * scale)
                    cv2.circle(direction_image, origin, 5, 60, -1)

                    # if (not x and not y) or (not x and y==(bins_per_side-1)) or (not y and x==(bins_per_side-1)) or (x==(bins_per_side-1) and y==(bins_per_side-1)):
                    #     continue

                    for x_move in range(-1,2):
                        for y_move in range(-1,2):
                            to_idx = (x + x_move) + (y + y_move) * bins_per_side

                            try:
                                endpoint = int(origin[0] + .1*scale * x_move * direction_all[from_idx, to_idx]), \
                                           int(origin[1] + .1*scale * y_move * direction_all[from_idx, to_idx])
                                cv2.arrowedLine(direction_image, origin, endpoint, 255, 2, tipLength = .15)
                            except: pass

                    cv2.imshow('angles', direction_image)


            # merge direction and exploration
            direction_merge = cv2.cvtColor(direction_image, cv2.COLOR_GRAY2BGR)
            alpha = .6
            cv2.addWeighted(direction_merge, alpha, exploration_image, 1 - alpha, 0, direction_merge)
            cv2.imshow('angles', direction_merge)

            # save results
            imageio.imwrite(os.path.join(experiment_save_folder, experiment + '_direction_' + condition + '.tif'), direction_merge[:, :, ::-1])

    '''
    LOOP ACROSS ALL EXPERIMENTS AND CONDITIONS
    '''
    # for experiment in analysis_dictionary:
    for experiment in experiments:
        for condition in analysis_dictionary[experiment]:
            # make sure this experiment was analyzed
            if not 'speed' in analysis_dictionary[experiment][condition]: continue

            # folder to save results
            experiment_save_folder = os.path.join(save_folder, experiment)

            # get the number of trials
            number_of_trials = 0
            scaling_factor = 100 / analysis_dictionary[experiment]['obstacle']['shape'][0]


            for i, mouse in enumerate(analysis_dictionary[experiment][condition]['speed']):
                number_of_trials += len(analysis_dictionary[experiment][condition]['speed'][mouse])
            if not number_of_trials: continue

            # initialize array to fill in with each trial's data
            time_axis = np.arange(-4, 10, 1/30)
            speed_traces = np.zeros((14 * 30, number_of_trials)) * np.nan
            subgoal_speed_traces = np.zeros((14 * 30, number_of_trials))  * np.nan
            HD_traces = np.zeros((14 * 30, number_of_trials))  * np.nan
            escape = np.zeros(number_of_trials)
            time = np.zeros(number_of_trials)
            end_idx = np.zeros(number_of_trials)
            RT = np.zeros(number_of_trials)

            # fill in with each trial's data
            trial_num = 0
            for i, mouse in enumerate(analysis_dictionary[experiment][condition]['speed']):
                for trial in range(len(analysis_dictionary[experiment][condition]['speed'][mouse])):
                    trial_speed = [s* scaling_factor * 30  for s in analysis_dictionary[experiment][condition]['speed'][mouse][trial] ]
                    trial_subgoal_speed = [s* scaling_factor * 30  for s in analysis_dictionary[experiment][condition]['geo speed'][mouse][trial] ]
                    trial_HD = analysis_dictionary[experiment][condition]['HD'][mouse][trial]

                    speed_traces[:len(trial_speed), trial_num] = gaussian_filter1d(trial_speed, 2)
                    subgoal_speed_traces[:len(trial_speed), trial_num] = gaussian_filter1d(trial_subgoal_speed, 2)
                    HD_traces[:len(trial_speed), trial_num] = gaussian_filter1d(trial_HD, 2)

                    escape[trial_num] = analysis_dictionary[experiment][condition]['escape'][mouse][trial]
                    time[trial_num] = analysis_dictionary[experiment][condition]['time'][mouse][0][trial]

                    end_idx[trial_num] = analysis_dictionary[experiment][condition]['time'][mouse][1][trial]

                    # intial_fast_speed = np.where(-subgoal_speed_traces[:len(trial_speed), trial_num] > 10)[0]
                    # if intial_fast_speed.size: RT[trial_num] = np.min(intial_fast_speed[intial_fast_speed > (4*30)]) / 30 - 4
                    # else: RT[trial_num] = np.nan

                    RT[trial_num] = analysis_dictionary[experiment][condition]['RT'][mouse][trial]

                    trial_num += 1


            order = np.argsort(time)
            order = order[::-1]

            '''
            PLOT THE SPEED TRACES
            '''
            # generate 2 2d grids for the x & y bounds
            fig, ax = plt.subplots(figsize=(14, 6))
            x, y = np.meshgrid(time_axis, np.arange(0, number_of_trials+1))
            # S = speed_traces[:, order]
            z = -subgoal_speed_traces[:, order].T

            c = ax.pcolormesh(x, y, z, cmap='magma', vmin=0, vmax=80)
            ax.set_title('Escape speed traces: ' + experiment + ' - ' + condition)
            ax.set_xlabel('time since stimulus onset (s)')
            ax.set_ylabel('time since session start (mins)')
            yticks = np.arange(1,number_of_trials,6)
            ax.set_yticks(yticks)
            ax.set_yticklabels(time[order[yticks]].astype(int))
            fig.colorbar(c, ax=ax, label='speed along best path to shelter (cm/s)') #

            ax.plot([0, 0], [0, number_of_trials], color='white', linewidth = 3)
            for i, trial in enumerate(order):
                try:
                    shelter_time = time_axis[int(end_idx[trial])] - time_axis[0]
                    reaction_time = RT[trial]
                    ax.plot([reaction_time, reaction_time], [i + .1, i + 1 - .1], color=[.5,.5,.5], linewidth=2)
                except:
                    ax.plot([0,0], [i, i+1], color='black', linewidth = 3)
                    continue

                ax.plot([shelter_time, shelter_time], [i+.1, i+1-.1], color='white', linewidth = 2)

            plt.savefig(os.path.join(experiment_save_folder, experiment + '_speed_' + condition + '.tif'))
            plt.close('all')

            '''
            PLOT HD TRACES
            '''
            # generate 2 2d grids for the x & y bounds
            fig, ax = plt.subplots(figsize=(12, 6))
            x, y = np.meshgrid(time_axis, np.arange(0, number_of_trials + 1))
            HD = abs(HD_traces[:, order])**(1/2)*(180**(1/2))

            ax.set_title('Escape HD traces: ' + experiment + ' - ' + condition)
            ax.set_xlabel('time since stimulus onset (s)')
            ax.set_ylabel('body angle relative to the shelter (deg)')

            for i, trial in enumerate(order):
                    plt.plot(time_axis, HD[:, trial], color='green', alpha=.6)

            ax.plot(time_axis, np.median(HD, axis = 1), color = 'white', linewidth = 5, alpha = .7)
            ax.plot(time_axis, np.percentile(HD, 75, axis=1), color='white', linewidth=2, alpha=.7, linestyle = '--')
            ax.plot(time_axis, np.percentile(HD, 25, axis=1), color='white', linewidth=2, alpha=.7, linestyle = '--')
            ax.plot([0, 0], [0, 180], color='gray', linewidth=2, linestyle = '--')

            # plt.savefig(os.path.join(experiment_save_folder, experiment + '_HD_' + condition + '.tif'))


            '''
            PLOT A STATE SPACE PLOT
            '''
            # generate 2 2d grids for the x & y bounds
            fig = plt.figure(figsize=(14, 10))
            ax = plt.axes(projection='3d')
            ax.view_init(elev=25, azim=45)

            S = speed_traces[:, order]
            HD = abs(HD_traces[:, order])

            epochs = [np.arange(0,120), np.arange(120, 150), np.arange(150, 180), np.arange(180, 270), np.arange(270, 420)]
            colors = [ [.7,.7,.7], [.9,.1,.1], [.7,0,.9], [.3,.1,.9], [0,0,.8]]
            for i, epoch in enumerate(epochs):
                ax.plot3D(S[epoch, :], HD[epoch, :], epoch/30 - 4, color = colors[i], alpha = .6)

            ax.set_title('State space plot: ' + experiment + ' - ' + condition)
            ax.set_xlabel('speed (cm/s)')
            ax.set_ylabel('body angle relative to shelter (deg)')
            ax.set_zlabel('time since stimulus onset (s)')
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.grid(False)

            # plt.savefig(os.path.join(experiment_save_folder, experiment + '_state space_' + condition + '.tif'))

            plt.close('all')

    experiment = 'Circle wall down (no shelter)'
    condition = 'no obstacle'
    mouse = 'CA7160'
    trial = 0


    # experiments = ['Circle wall up', 'Circle wall (shelter on side)', 'Circle void (shelter on side)']
    experiments = ['Circle wall down (no baseline)', 'Circle wall down']
    # experiments = ['Circle wall down (dark)','Circle wall down']
    # experiments = ['Circle wall up', 'Circle wall down (no shelter)', 'Circle wall down (no baseline)', 'Circle wall down']
    # experiments = ['Circle wall down', 'Circle wall down (dark)']

    # experiments = ['Circle wall up', 'Circle wall down', 'Circle wall down (dark)']
    # experiments = ['Circle wall up', 'Circle wall down (no shelter)', 'Circle wall down (no baseline)', 'Circle wall down','Circle wall down']
    # experiments = ['Circle wall down (no baseline)'] #'Circle wall down (no shelter)']
    # experiments = ['Circle wall down', 'Circle wall down (no baseline)']

    # conditions = ['no obstacle', 'obstacle', 'obstacle']
    conditions = ['no obstacle', 'no obstacle', 'no obstacle', 'obstacle']
    # conditions = ['no obstacle', 'no obstacle', 'obstacle', 'no obstacle', 'obstacle']


    # x_labels = ['No obstacle ever', 'Obstacle removed', 'Obstacle']
    # x_labels = ['No obstacle - early', 'Obstacle removed', 'Obstacle - early', 'Obstacle added']
    #
    # x_labels = ['Lights on', 'Lights off']
    # x_labels = ['Nothing', 'Wall', 'Hole']
    # x_labels = ['Wall + Shelter 1', 'Wall + Shelter 2', 'Wall + no shelter']
    # x_labels = ['No obstacle', 'OR (no shelter)', 'OR (w/ shelter)', 'Obstacle']
    # x_labels = ['Obstacle removed (no baseline trials)', 'Obstacle removed (baseline without shelter)', 'Obstacle in dark', 'No obstacle']
     #, 'Obstacle (dark)']
    x_labels = ['Obstacle removed (no baseline trials)', 'Obstacle removed (no baseline trials)']# 'Obstacle removed (3 baseline trials)',  'Obstacle removed (baseline w/o shelter)']

    # x_labels = ['Lights on', 'Lights off']
    # x_labels = ['Obstacle removed (no baseline trials)', 'Obstacle removed (baseline without shelter)']
    # x_labels = ['No obstacle', 'Obstacle']



    all_data = {}

    # metrics = ['path length', 'speed', 'RT', 'linearity']
    # metrics = ['speed', 'peak speed', 'RT', 'linearity']
    metrics = ['linearity'] #, 'OM linearity']#, 'edginess']
    # metrics = ['OM linearity']
    # metrics = ['SR']
    # metrics = ['exploration']

    # experiments = ['Circle wall down', 'Circle wall up', 'Circle wall down', 'Circle wall up']
    # conditions = ['no obstacle', 'no obstacle', 'obstacle', 'obstacle']
    mode = 'path' #lunge or path
    traj_loc = 44.75
    # PM = 8
    ETD = 7  # 7
    RT_speed = 15

    time_limit = [0, 70]
    scaling_factor = 100/720


    # colors = [[1, .1, .3], [.8, .1, .85], [.1, .6, 1], 'green']
    colors = [ [.1, .6, 1], 'cyan', 'green', [.8, .1, .85]]



    '''
    PLOT TIME TO SHELTER ACROSS SESSION PLOTS
    '''
    for plot_num, what_to_plot in enumerate(metrics):

        # X LABELS
        # x_labels = ['No obstacle', 'OR (no shelter)', 'OR (no BL)', 'OR (3BL)', 'Obstacle']


        # name things things
        if what_to_plot == 'escape speed':
            title = 'Homing speed'
            y_label = 'length of optimal path to shelter / time to shelter (cm/s)'
        elif what_to_plot == 'path length':
            title = 'Path efficiency'
            y_label = '1 / path length relative to optimal path'
        elif what_to_plot == 'RT':
            title = 'Reaction time'
            y_label = 'reaction time (s)'
        elif what_to_plot == 'speed':
            title = 'Speed during escape'
            y_label = 'speed (cm/s)'
        elif what_to_plot == 'peak speed':
            title = 'Peak speed during escape'
            y_label = 'speed (cm/s)'
        elif what_to_plot == 'linearity':
            title = 'Escape strategy'
            y_label = 'Closer to homing vector (cm)                               Closer to wall-edge vector (cm)'
            y_label = 'edginess'

        elif what_to_plot == 'OM linearity':
            title = 'Sub-goal vs. homing-vector escape strategy (lunge)'
            y_label = 'Distance closer to wall-edge vector than to homing vector (cm)'
        elif what_to_plot == 'edginess':
            title = 'Straightness of escape'
            y_label = 'deviation from subgoal - deviation from homing vector, 5 cm before the wall (cm)'
        elif what_to_plot == 'gravity':
            title = 'Deviation toward center'
            y_label = 'path bias toward center of arena, normalized to greatest possible bias'
        elif what_to_plot == 'exploration':
            title = 'Place preference - central object'
            y_label = 'time spent near object relative to time spent exploring elsewhere'
        elif what_to_plot == 'SR':
            title = 'Edginess of all homings'
            y_label = 'deviation from subgoal - deviation from homing vector, 5 cm before the wall (cm)'


        # set up plot for escape duration across session
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_title(title)
        ax1.set_xlabel('time since session start (mins)')
        ax1.set_ylabel(y_label)

        # set up plot for escape duration boxplot
        fig2, ax2 = plt.subplots(figsize=(7, 11))
        ax2.set_title(title)# + ' (' + str(time_limit[0]) + ' - ' + str(time_limit[1]) + ' mins)')
        ax2.set_ylabel(y_label)
        ax2.set_xticks(np.arange(len(experiments)))
        ax2.set_xticklabels(x_labels)
        ax2.set_xlim([-1,len(experiments)])
        plt.xticks(rotation=30)
        ax2.set_ylim([-22, 32])
        ax2.set_ylim([-.01,1.01])

        # # set up plot for escape duration across session
        fig3, ax3 = plt.subplots(figsize=(15, 15))
        ax3.set_title('Correlation between previous and current homing trajectories') # + ' (' + mode + ')')
        ax3.set_xlabel('prior homing edginess')
        ax3.set_ylabel('current homing edginess')



        legend = []

        c = 0
        x = 0

        for exp, (experiment, condition) in enumerate(zip(experiments, conditions)):

            # make sure this experiment was analyzed
            if not 'speed' in analysis_dictionary[experiment][condition]:
                continue
            legend.append(experiment + ' - ' + condition)

            # folder to save results
            experiment_save_folder = os.path.join(save_folder, experiment)

            # get the number of trials
            number_of_trials = 0
            bins_per_side = analysis_dictionary[experiment]['obstacle']['direction scale']
            scale = int(analysis_dictionary[experiment]['obstacle']['shape'][0] / bins_per_side)
            scaling_factor = 100 / analysis_dictionary[experiment]['obstacle']['shape'][0]

            for i, mouse in enumerate(analysis_dictionary[experiment][condition]['speed']):
                number_of_trials += len(analysis_dictionary[experiment][condition]['speed'][mouse])
            if not number_of_trials:
                continue

            # initialize array to fill in with each trial's data
            time_axis = np.arange(-4, 10, 1/30)
            escape = np.zeros(number_of_trials)
            time = np.zeros(number_of_trials)
            quantity = np.zeros(number_of_trials)
            quantity_2 = np.zeros(number_of_trials)
            path_length = np.zeros(number_of_trials)

            # fill in with each trial's data
            trial_num = 0
            for i, mouse in enumerate(analysis_dictionary[experiment][condition]['speed']):
                mouse_trials = 0

                for trial in range(len(analysis_dictionary[experiment][condition]['speed'][mouse])):



                    '''
                    FILTER BY TRIALS
                    '''
                    # only do first trial after obstacle change
                    # if 'down' in experiment and condition == 'no obstacle' and trial > 2:
                    # if analysis_dictionary[experiment][condition]['escape'][mouse][trial]:
                    #     if trial > np.where(analysis_dictionary[experiment][condition]['escape'][mouse])[0][0]:
                    # if trial > 3:
                    if mouse_trials > 3: # or len(analysis_dictionary[experiment][condition]['SR'][mouse][trial][0]) < 1: # or ('baseline' in experiment and mouse_trials > 2):
                        quantity[trial_num] = np.nan
                        trial_num += 1
                        mouse_trials += 1
                        continue

                    # get the stim onset time
                    time[trial_num] = analysis_dictionary[experiment][condition]['time'][mouse][0][trial]
                    if analysis_dictionary[experiment][condition]['time'][mouse][0][0] < 7: time[trial_num] += 7

                    time[trial_num] = trial


                    if (not analysis_dictionary[experiment][condition]['escape'][mouse][trial] or not analysis_dictionary[experiment][condition]['path'][mouse][trial][1].size) \
                        and (not what_to_plot == 'exploration'):
                        quantity[trial_num] = np.nan
                    else:

                        if not what_to_plot == 'exploration':
                            # filter by stim position
                            initial_y_pos = analysis_dictionary[experiment][condition]['path'][mouse][trial][1][0] * scaling_factor
                        else: initial_y_pos  = 0

                        '''
                        MAKE SURE HIGH ENOUGH IN ARENA (AND/OR FAST ENOUGH TO SHELTER)
                        '''
                        print(initial_y_pos)
                        if initial_y_pos > 23 or len(analysis_dictionary[experiment][condition]['path'][mouse][trial][1]) > (30 * 7): #25 / 7 needed for certain analyses!
                            quantity[trial_num] = np.nan
                        else:

                            # get the reaction time
                            trial_subgoal_speed = [s * scaling_factor * 30 for s in analysis_dictionary[experiment][condition]['geo speed'][mouse][trial]]
                            subgoal_speed_trace = gaussian_filter1d(trial_subgoal_speed, 2)
                            intial_speed = np.where(-subgoal_speed_trace > RT_speed)[0] #15
                            # intial_fast_speed = np.where(-subgoal_speed_trace > 10)[0]

                            if intial_speed.size and not what_to_plot == 'exploration':
                                RT = ( np.min(intial_speed[intial_speed > (4 * 30)]) / 30 - 4 )
                                # RT_fast = (np.min(intial_fast_speed[intial_fast_speed > (4 * 30)]) / 30 - 4)
                            else: RT = np.nan

                            if what_to_plot == 'escape speed':
                                quantity[trial_num] = analysis_dictionary[experiment][condition]['optimal path length'][mouse][trial]\
                                                      * scaling_factor \
                                                      / (analysis_dictionary[experiment][condition]['time'][mouse][1][trial] / 30 - RT)

                            if what_to_plot == 'speed':
                                quantity[trial_num] = analysis_dictionary[experiment][condition]['actual path length'][mouse][trial] \
                                                      * scaling_factor  \
                                                      / (analysis_dictionary[experiment][condition]['time'][mouse][1][trial] / 30 - RT)

                            if what_to_plot == 'peak speed':
                                quantity[trial_num] = np.max(analysis_dictionary[experiment][condition]['speed'][mouse][trial]) * scaling_factor * 30

                            elif what_to_plot == 'path length':
                                quantity[trial_num] = 1 / ((analysis_dictionary[experiment][condition]['actual path length'][mouse][trial]+10) / \
                                                      analysis_dictionary[experiment][condition]['optimal path length'][mouse][trial])
                                if quantity[trial_num] > 1:
                                    print('path: ' + experiment + mouse + ' ' + str(trial))

                                print('path: ' + experiment + mouse + ' ' + str(trial) + ' ' + str(quantity[trial_num]))

                            elif what_to_plot == 'RT':
                                if analysis_dictionary[experiment][condition]['escape'][mouse][trial]: quantity[trial_num] = RT #analysis_dictionary[experiment][condition]['RT'][mouse][trial] / 30
                                else: quantity[trial_num] = np.nan

                            elif what_to_plot == 'linearity' or (what_to_plot == 'SR' and mode == 'path'):

                                # skip if no previous homings
                                SR = np.array(analysis_dictionary[experiment][condition]['SR'][mouse][trial][0])
                                if len(SR) <= 2:
                                    quantity[trial_num] = np.nan
                                    trial_num += 1
                                    mouse_trials += 1
                                    continue
                                    # quantity[trial_num] = 0

                                # get position for the trial
                                x_pos = analysis_dictionary[experiment][condition]['path'][mouse][trial][0][int(RT*30):] * scaling_factor
                                y_pos = analysis_dictionary[experiment][condition]['path'][mouse][trial][1][int(RT*30):] * scaling_factor
                                x_pos_start = analysis_dictionary[experiment][condition]['path'][mouse][trial][0][int(RT*30)] * scaling_factor
                                y_pos_start = analysis_dictionary[experiment][condition]['path'][mouse][trial][1][int(RT*30)] * scaling_factor
                                # x_pos_start = analysis_dictionary[experiment][condition]['lunge'][mouse][trial][0][0] * scaling_factor
                                # y_pos_start = analysis_dictionary[experiment][condition]['lunge'][mouse][trial][0][1] * scaling_factor

                                # do line from starting position to ending position (or to shelter...)
                                y_pos_shelter = 86.5
                                x_pos_shelter = 50
                                # y_pos_shelter = y_pos[-1]
                                # x_pos_shelter = x_pos[-1]

                                slope = (y_pos_shelter - y_pos_start) / (x_pos_shelter - x_pos_start)
                                intercept = y_pos_start - x_pos_start * slope
                                distance_to_line = abs(y_pos - slope * x_pos - intercept) / np.sqrt( (-slope)**2 + (1)**2 )

                                # # get index at center point (wall location)
                                mouse_at_center = np.argmin(abs(y_pos - (traj_loc)))
                                # mouse_at_center0 = np.argmin( abs(y_pos - (traj_loc - PM)) ) #43
                                # mouse_at_center1 = np.argmin( abs(y_pos - (traj_loc + PM)) ) #was 46
                                # if mouse_at_center0 > mouse_at_center1: mouse_at_center0 = np.min((mouse_at_center0, mouse_at_center1)); mouse_at_center1 = mouse_at_center0 + 1

                                # linear_offset = np.mean(distance_to_line[mouse_at_center0:mouse_at_center1])#:mouse_at_center1])
                                linear_offset = distance_to_line[mouse_at_center]  #:mouse_at_center1])

                                # # get line to the closest edge
                                # mouse_at_center0 = np.argmin(abs(y_pos - 43))
                                y_edge = 50
                                if x_pos[mouse_at_center] > 50: x_edge = 75
                                else: x_edge = 25

                                # do line from starting position to edge position
                                slope = (y_edge - y_pos_start) / (x_edge - x_pos_start)
                                intercept = y_pos_start - x_pos_start * slope
                                distance_to_line = abs(y_pos - slope * x_pos - intercept) / np.sqrt( (-slope)**2 + (1)**2 )

                                if what_to_plot == 'linearity':
                                    # get index at center point (wall location)
                                    # quantity[trial_num] = linear_offset - np.mean(distance_to_line[mouse_at_center0:mouse_at_center1])
                                    quantity[trial_num] = (np.min((25, linear_offset - np.mean(distance_to_line[mouse_at_center]))) + 25) / 50

                                    if np.isnan(quantity[trial_num]):
                                        quantity_2[trial_num] = np.nan
                                        continue

                                    print('edginess: ' + experiment + mouse + ' ' + str(trial) + ' ' + str(quantity[trial_num]))
                                elif what_to_plot == 'SR':
                                    # quantity_2[trial_num] = linear_offset - np.mean(distance_to_line[mouse_at_center0:mouse_at_center1])
                                    quantity_2[trial_num] = (np.min((25, linear_offset - np.mean(distance_to_line[mouse_at_center]))) + 25) / 50

                                    if np.isnan(quantity_2[trial_num]):
                                        quantity[trial_num] = np.nan
                                        continue

                                    print('edginess: ' + experiment + mouse + ' ' + str(trial) + ' ' + str(quantity_2[trial_num]))

                            elif what_to_plot == 'OM linearity' or (what_to_plot == 'SR' and mode == 'lunge'):

                                # get position for the trial
                                x_pos_start = analysis_dictionary[experiment][condition]['lunge'][mouse][trial][0][0] * scaling_factor
                                y_pos_start = analysis_dictionary[experiment][condition]['lunge'][mouse][trial][0][1] * scaling_factor
                                x_pos_end = analysis_dictionary[experiment][condition]['lunge'][mouse][trial][1][0] * scaling_factor
                                y_pos_end = analysis_dictionary[experiment][condition]['lunge'][mouse][trial][1][1] * scaling_factor

                                # wait til gone far enough
                                slope = (y_pos_end - y_pos_start) / (x_pos_end - x_pos_start + .001)
                                intercept = y_pos_start - x_pos_start * slope
                                x_OM = (traj_loc - intercept) / slope

                                # do line from starting position to ending position (or to shelter...)
                                y_pos_shelter = 86.5
                                x_pos_shelter = 50
                                # x_pos_shelter = analysis_dictionary[experiment][condition]['path'][mouse][trial][0][-1] * scaling_factor
                                # y_pos_shelter = analysis_dictionary[experiment][condition]['path'][mouse][trial][1][-1] * scaling_factor

                                slope = (y_pos_shelter - y_pos_start) / (x_pos_shelter - x_pos_start)
                                intercept = y_pos_start - x_pos_start * slope
                                distance_to_line = abs(traj_loc - slope * x_OM - intercept) / np.sqrt( (-slope)**2 + (1)**2 )

                                # get index at center point (wall location)
                                linear_offset = distance_to_line

                                # get line to the closest edge
                                # mouse_at_center0 = np.argmin(abs(y_pos - 43))
                                y_edge = 50
                                if x_OM > 50: x_edge = 75
                                else: x_edge = 25

                                # do line from starting position to edge position
                                slope = (y_edge - y_pos_start) / (x_edge - x_pos_start)
                                intercept = y_pos_start - x_pos_start * slope
                                distance_to_line = abs(traj_loc - slope * x_OM - intercept) / np.sqrt( (-slope)**2 + (1)**2 )


                                if what_to_plot == 'OM linearity':
                                    # get index at center point (wall location)
                                    quantity[trial_num] = np.min((25, linear_offset - distance_to_line))

                                    if np.isnan(quantity[trial_num]):
                                        quantity_2[trial_num] = np.nan
                                        continue


                                    print('edginess: ' + experiment + mouse + ' ' + str(trial) + ' ' + str(quantity[trial_num]))

                                elif what_to_plot == 'SR':
                                    # get index at center point (wall location)
                                    quantity_2[trial_num] = np.min((25, linear_offset - distance_to_line))

                                    if np.isnan(quantity_2[trial_num]):
                                        quantity[trial_num] = np.nan
                                        continue

                                    print('edginess: ' + experiment + mouse + ' ' + str(trial) + ' ' + str(quantity_2[trial_num]))

                            elif what_to_plot == 'gravity':

                                # get position for the trial
                                x_pos = analysis_dictionary[experiment][condition]['path'][mouse][trial][0][int(RT*30):] * scaling_factor
                                y_pos = analysis_dictionary[experiment][condition]['path'][mouse][trial][1][int(RT*30):] * scaling_factor

                                # must no be right at obstacle
                                if abs(x_pos[0] - 50) < 8 or abs(x_pos[-1] - 50) > 8:
                                    quantity[trial_num] = np.nan
                                    print(mouse + ' - ' + str(trial))

                                else:

                                    # get straight line prediction
                                    slope = (y_pos[-1] - y_pos[0]) / (x_pos[-1] - x_pos[0])
                                    intercept = y_pos[0] - x_pos[0] * slope
                                    x_line = np.linspace(np.round(x_pos[0]), np.round(x_pos[-1]), int(abs(np.round(x_pos[0]) - np.round(x_pos[-1]))+1))
                                    y_line = x_line * slope + intercept
                                    # y_line = np.arange(np.round(y_pos[0]), np.round(y_pos[-1]))
                                    # x_line = (y_line - intercept) / slope

                                    # get the y-coord where the center is crossed - line
                                    center_offset = 7
                                    x_cross_line_idx = np.where(  (abs(np.round(x_line - 50)) <= (center_offset + 1)) * \
                                                                  (abs(np.round(x_line - 50)) >= (center_offset - 1)) )[0][0]
                                    y_cross_line = y_line[x_cross_line_idx]


                                    # get the y-coord where the center is crossed - data
                                    x_cross_data_idx = np.where(  (abs(np.round(x_pos - 50)) <= (center_offset + 1)) * \
                                                                  (abs(np.round(x_pos - 50)) >= (center_offset - 1)) )[0][0]
                                    y_cross_data = y_pos[x_cross_data_idx]
                                    x_cross_data = x_pos[x_cross_data_idx]

                                    # get discrepancy between line and data
                                    straight_line_length = np.sqrt((y_pos[-1] - y_pos[0]) ** 2 + (x_pos[-1] - x_pos[0]) ** 2)
                                    # quantity[trial_num] = (y_cross_line - y_cross_data) / straight_line_length

                                    # normalize to max possible:
                                    if np.sign(y_cross_line - y_cross_data) > 0:
                                        quantity[trial_num] = (y_cross_line - y_cross_data) / (y_cross_line - 5)
                                    else:
                                        quantity[trial_num] = (y_cross_line - y_cross_data) / (95 - y_cross_line)

                                    print('gravity: ' + experiment + mouse + ' ' + str(trial) + ' ' + str(quantity[trial_num]))

                            elif what_to_plot == 'exploration':

                                # only take the first trial
                                if trial:
                                    quantity[trial_num] = np.nan
                                    print(mouse + ' ' + str(trial))
                                else:
                                    # get the stored exploration plot - proportion of time at each location
                                    exploration = analysis_dictionary[experiment][condition]['exploration'][mouse]
                                    exploration = exploration / np.sum(exploration)

                                    # get relevant set and subset
                                    exploration_set = exploration[0:int(70/scaling_factor) , :]
                                    exploration_subset = exploration[int(15/scaling_factor):int(70/scaling_factor) , int(40/scaling_factor):int(60/scaling_factor)]

                                    # exploration_set = exploration[0:int(70 / scaling_factor), :]
                                    # exploration_subset = exploration[int(45 / scaling_factor):int(55 / scaling_factor),int(15 / scaling_factor):int(85 / scaling_factor)]

                                    proportion_in_subset = np.sum(exploration_subset) / np.sum(exploration_set)
                                    relative_size = exploration_subset.size / exploration_set.size

                                    # return place preference index
                                    # quantity_2[trial_num] = proportion_in_subset / relative_size
                                    quantity[trial_num] = proportion_in_subset / relative_size

                                    print('PP: ' + experiment + mouse + ' ' + str(trial) + ' ' + str(quantity[trial_num]))

                            if what_to_plot == 'SR':

                                # get the stored exploration plot - proportion of time at each location
                                SR = np.array(analysis_dictionary[experiment][condition]['SR'][mouse][trial][0])
                                edge_proximity = 50 - np.array(analysis_dictionary[experiment][condition]['SR'][mouse][trial][1])
                                thru_center = np.array(analysis_dictionary[experiment][condition]['SR'][mouse][trial][2])
                                SR_time = np.array(analysis_dictionary[experiment][condition]['SR'][mouse][trial][3])

                                # get position for the trial
                                x_pos = analysis_dictionary[experiment][condition]['path'][mouse][trial][0][int(RT * 30):] * scaling_factor
                                y_pos = analysis_dictionary[experiment][condition]['path'][mouse][trial][1][int(RT * 30):] * scaling_factor

                                # get the sidedness of the escape
                                mouse_at_center = np.argmin( abs(y_pos - traj_loc) )

                                # get line to the closest edge
                                y_edge = 50
                                if x_pos[mouse_at_center] > 50: x_edge = 75
                                else: x_edge = 25

                                # only use recent escapes
                                if len(SR) >= (ETD+1):
                                    SR = SR[-ETD:]
                                    thru_center = thru_center[-ETD:]
                                    edge_proximity = edge_proximity[-ETD:]


                                # get line to the closest edge, exclude escapes to other edge
                                MOE = 10.2 # 10.1
                                if x_pos[mouse_at_center] > 50: edge_proximity = edge_proximity[SR > 25+MOE]; thru_center = thru_center[SR > 25+MOE]; SR = SR[SR > 25+MOE]  #35
                                else: edge_proximity = edge_proximity[SR < 75-MOE]; thru_center = thru_center[SR < 75-MOE]; SR = SR[SR < 75-MOE]  #65


                                if SR.size:
                                    # if previously went to that edge, then predict going again to that edge
                                    #  if (abs(SR[-1] - x_edge) < 6) * (edge_proximity[-1] < 4) and False: #TEMPORARY
                                    #     x_repetition = SR[np.argmin(abs(SR - x_edge))]
                                    #  else:
                                        # get mean x position
                                    x_repetition = np.mean(SR)

                                # NOW DO THE EDGINESS ANALYSIS, WITH REPETITION AS THE REAL DATA
                                # do line from starting position to ending position (or to shelter...)
                                y_pos[-1] = 86.5
                                x_pos[-1] = 50

                                if mode == 'lunge':
                                    x_pos_start = analysis_dictionary[experiment][condition]['lunge'][mouse][trial][0][0] * scaling_factor
                                    y_pos_start = analysis_dictionary[experiment][condition]['lunge'][mouse][trial][0][1] * scaling_factor
                                elif mode == 'path':
                                    x_pos_start = x_pos[0]
                                    y_pos_start = y_pos[0]
                                else: print('select proper mode')

                                slope = (y_pos[-1] - y_pos_start) / (x_pos[-1] - x_pos_start)
                                intercept = y_pos_start - x_pos_start * slope
                                if x_repetition:
                                    distance_to_line = abs(traj_loc - slope * x_repetition - intercept) / np.sqrt((-slope) ** 2 + (1) ** 2)


                                # do line from starting position to edge position
                                slope = (y_edge - y_pos[0]) / (x_edge - x_pos[0])
                                intercept = y_pos[0] - x_pos[0] * slope
                                distance_to_edge = abs(traj_loc - slope * x_repetition - intercept) / np.sqrt((-slope) ** 2 + (1) ** 2)

                                # get index at center point (wall location)
                                quantity[trial_num] = (np.max((-25, np.min((25, distance_to_line - distance_to_edge)))) + 25) / 50

                                if np.isnan(quantity[trial_num]):
                                    quantity_2[trial_num] = np.nan
                                    continue

                                # # TEMPORARY
                                if len(SR) <= 1 and 'up' in experiment:
                                #     quantity[trial_num] = np.nan
                                #     trial_num += 1
                                #     mouse_trials += 1
                                #     continue
                                    quantity[trial_num] = 0

                                print('SR edginess: ' + experiment + mouse + ' ' + str(trial) + ' ' + str(quantity[trial_num]))


                            mouse_trials += 1
                    trial_num += 1

            '''
            BOX PLOT COMPARING CONDITIONS
            '''
            # plot escapes within time limit a la box plot
            in_time_idx = (time >= time_limit[0]) * (time <= time_limit[1])
            quantity_for_box_plot =  quantity[~np.isnan(quantity) * in_time_idx]

            print(str(len(quantity_for_box_plot)) + ' total trials')
            print(str(np.sum(quantity_for_box_plot < 0)) + ' homing vectors')
            print(str(np.sum(quantity_for_box_plot > 0)) + ' sub goals')

            # put in the all-data dict
            all_data[experiment+condition] = quantity_for_box_plot

            # make a boxplot
            median = np.median(quantity_for_box_plot)
            lower = np.percentile(quantity_for_box_plot, 25)
            upper = np.percentile(quantity_for_box_plot, 75)

            # plot the median and IQR
            ax2.errorbar(x, median, yerr = np.array([[median - lower], [upper - median]]), color = colors[c],
                         capsize = 12, capthick = 3, alpha = .6, linewidth = 3)
            ax2.scatter(x, median, color=colors[c], s=300, alpha=.8)

            # plot each trial
            ax2.scatter(np.ones_like(quantity_for_box_plot) * x, quantity_for_box_plot, color=colors[c], s = 20, alpha=.9, edgecolors= 'black') #[.2,.2,.2])



            '''
            CORRELATION PLOT
            '''
            # plot correlation of linearity and peak speed
            if colors[c] == 'cyan' or colors[c] == [1, .1, .3]: alpha = .4
            else: alpha = .9

            # if experiment == 'Circle wall down (dark)' and condition == 'obstacle':
                # exclude_idx = (quantity > 120) * (quantity_2 < 0) # (quantity_2 < -16)
                # legplot = ax3.scatter(quantity[in_time_idx], quantity_2[in_time_idx], 20, color=colors[c], alpha=alpha)
            ax3.scatter(quantity[in_time_idx], quantity_2[in_time_idx], 20, color=colors[c], alpha=alpha)

            # do linear regression
            q1 = quantity[~np.isnan(quantity)]
            q2 = quantity_2[~np.isnan(quantity)]
            order = np.argsort(q1)
            q1 = q1[order]
            q2 = q2[order]
            r, p = scipy.stats.pearsonr(q1, q2)
            print(r, p)
            if p > .0005:
                x_labels[exp] = x_labels[exp] + '    r = ' + str(np.round(r,2)) + ', p = ' +  str(np.round(p,3)) + ''
            else:
                if 'baseline' in experiment:
                    x_labels[exp] = x_labels[exp] + '         r = ' + str(np.round(r, 2)) + ', p < .001'
                elif not 'shelter' in experiment:
                    x_labels[exp] = x_labels[exp] + '           r = ' + str(np.round(r, 2)) + ', p < .001'

            LR = LRPI(t_value=1)
            LR.fit(q1, q2)
            prediction = LR.predict(q1)

            # if not (experiment == 'Circle wall down (dark)' and condition == 'obstacle'):
            ax3.plot(q1, prediction['Pred'].values, color=colors[c], linewidth=1, linestyle='--', alpha=.5)
            ax3.fill_between(q1, prediction['lower'].values, prediction['upper'].values, color=colors[c], alpha=alpha/8)#6

            '''
            ALL-SESSION SCATTER PLOT AND LINEAR REGRESSION
            '''
            # plot all escapes over time
            ax1.scatter(time, quantity, color = colors[c], alpha = .5)

            # plot a line - do linear regression and error bars?
            time_of_stim = time[~np.isnan(quantity)]
            quantity_to_plot = quantity[~np.isnan(quantity)]
            length_to_shelter = path_length[~np.isnan(quantity)]

            # put in order
            order = np.argsort(time_of_stim)
            time_of_stim = time_of_stim[order].reshape(-1, 1)
            quantity_to_plot = quantity_to_plot[order].reshape(-1, 1)

            # do linear regression
            LR = LRPI(t_value=1)
            LR.fit(time_of_stim, quantity_to_plot)
            prediction = LR.predict(time_of_stim)

            ax1.plot( time_of_stim, prediction['Pred'].values, color=colors[c], linewidth=2, linestyle = ':', alpha = .5)
            ax1.fill_between( time_of_stim[:,0], prediction['lower'].values, prediction['upper'].values, color=colors[c], alpha = .1)

            c+=1
            x += 1

        leg1 = ax1.legend(legend)
        leg1.draggable(True)

        if 'linearity' in what_to_plot or what_to_plot == 'gravity' or what_to_plot == 'SR': ax2.plot(np.arange(-1, len(experiments) + 1), np.zeros(len(experiments) + 2)+.5, color='white', linestyle='--', alpha=.4)
        if what_to_plot == 'exploration': ax2.plot(np.arange(-1, len(experiments) + 1), np.ones(len(experiments) + 2), color='white', linestyle='--', alpha=.4)
        if what_to_plot == 'SR':
            ax3.plot([-100, 100], [0.5,0.5], color='white', linestyle='--', alpha=.2); ax3.plot([0.5,0.5], [-100, 100], color='white', linestyle='--', alpha=.2)
            ax3.set_ylim([-.01,1.1]); ax3.set_xlim([-.01,1.1])
        # leg1 = ax3.legend((legplot,), (x_labels[-1],))
        # leg1.draggable(True)
        leg1 = ax3.legend(x_labels)
        leg1.draggable(True)

        # do statistical test
        # number_of_tests = factorial(len(experiments)) / (factorial(2) * factorial(len(experiments) - 2))
        # for i, (experiment1, condition1) in enumerate(zip(experiments, conditions)):
        #     for j, (experiment2, condition2) in enumerate(zip(experiments, conditions)):
        #
        #         if i < j:
        #             # do t test
        #             t, p = scipy.stats.ttest_ind(all_data[experiment1+condition1], all_data[experiment2+condition2], equal_var=False)
        #
        #             # do Mann-Whitney
        #             # t, p = scipy.stats.mannwhitneyu(all_data[experiment1+condition1], all_data[experiment2+condition2])
        #
        #             # Bony Ferroni correction
        #             p *= number_of_tests
        #
        #             # plot line
        #             y_limit = ax2.get_ylim()[1]
        #             if p < 0.05:
        #                 ax2.plot([i, j], [y_limit, y_limit], color = 'white', alpha = .8, linewidth = 3)
        #                 ax2.scatter(np.mean([i,j]),1.01*y_limit, color = 'white', marker = '*' )
        #                 if p < 0.001:
        #                     ax2.scatter(np.mean([i, j])-.1, 1.01 * y_limit, color='white', marker='*')
        #                     if p < 0.0001:
        #                         ax2.scatter(np.mean([i, j])+.1, 1.01 * y_limit, color='white', marker='*')


    plt.show()
    print('hi')
    plt.close('all')



    '''
    Do barplot with chi squared test
    '''
    # x_label = ['No obstacle - all', 'Obstacle - exp', 'Obstacle - naive','Obstacle - exp (lunge)','Obstacle - naive (lunge)','Obstacle in dark - all']
    x_label = ['No obstacle', 'Obstacle - exp', 'Obstacle - exp (lunge)', 'Obstacle - naive', 'Obstacle - naive (lunge)', 'Obstacle in dark']

    # data = np.array([[3, 19, 7, 17, 5, 18],
    #                 [19, 2, 1, 4, 3, 14]])

    # data = np.array([[3, 18, 7, 12, 5, 18],
    #                 [19, 1, 1, 1, 3, 14]])

    data = np.array([[3, 18, 13, 7, 4, 18],
                    [19, 1, 1, 1, 3, 14]])

    colors1 = [[.8, 0, 0],[0,.8,0],[0,.8,0],[0,.8,0],[0,.8,0],[0, 0, .8]]
    colors2 = [[.4, 0.1, 0.1], [.1,.4,.1], [.1,.4,.1], [.1,.4,.1], [.1,.4,.1],[.1,.1,.4]]

    fig4, ax4 = plt.subplots(figsize=(20, 11))
    ax4.set_title('Choice of strategy')
    ax4.set_ylabel('Proportion homing vector ----------------------- Proportion wall-edge vector   ')
    ax4.set_xticks(np.arange(len(x_label)))
    ax4.set_xticklabels(x_label)
    ax4.set_xlim([-1,len(x_label)])
    ax4.set_ylim([0,1.1])
    plt.xticks(rotation=0)

    # plot the data
    x = np.arange(len(x_label))
    normed_data = data / np.sum(data, axis = 0)
    ax4.bar(x, normed_data[0, :], .5, color = colors1)
    ax4.bar(x, normed_data[1, :], .5, bottom=normed_data[0, :], color=colors2)





    # do statistical test
    number_of_tests = factorial(len(x_label)) / (factorial(2) * factorial(len(x_label) - 2))
    for i, label1 in enumerate(x_label):
        for j, label2 in enumerate(x_label):

            if i < j:

                # just use 2-way comparison
                cur_data = np.zeros((2,2))
                cur_data[0, :] = data[:, i]
                cur_data[1, :] = data[:, j]

                # do chi sq test
                o, p = scipy.stats.fisher_exact(cur_data)

                # Bony Ferroni correction
                p *= number_of_tests

                # plot line
                y_limit = ax4.get_ylim()[1]
                if p < 0.05:
                    ax4.plot([i, j], [y_limit, y_limit], color = 'white', alpha = .8, linewidth = 3)
                    ax4.scatter(np.mean([i,j]),1.01*y_limit, color = 'white', marker = '*' )
                    if p < 0.001:
                        ax4.scatter(np.mean([i, j])-.1, 1.01 * y_limit, color='white', marker='*')
                        if p < 0.0001:
                            ax4.scatter(np.mean([i, j])+.1, 1.01 * y_limit, color='white', marker='*')





class LRPI:
    def __init__(self, normalize=False, n_jobs=1, t_value=2.13144955):
        self.normalize = normalize
        self.n_jobs = n_jobs
        self.LR = linear_model.LinearRegression(normalize=self.normalize, n_jobs=self.n_jobs)
        self.t_value = t_value

    def fit(self, X_train, y_train):
        self.X_train = pd.DataFrame(X_train)
        self.y_train = pd.DataFrame(y_train)

        self.LR.fit(self.X_train, self.y_train)
        X_train_fit = self.LR.predict(self.X_train)
        self.MSE = np.power(self.y_train.subtract(X_train_fit), 2).sum(axis=0) / (self.X_train.shape[0] - self.X_train.shape[1] - 1)
        self.X_train.loc[:, 'const_one'] = 1
        self.XTX_inv = np.linalg.inv(np.dot(np.transpose(self.X_train.values), self.X_train.values))

    def predict(self, X_test):
        self.X_test = pd.DataFrame(X_test)
        self.pred = self.LR.predict(self.X_test)
        self.X_test.loc[:, 'const_one'] = 1
        SE = [np.dot(np.transpose(self.X_test.values[i]), np.dot(self.XTX_inv, self.X_test.values[i])) for i in range(len(self.X_test))]
        results = pd.DataFrame(self.pred, columns=['Pred'])

        results.loc[:, "lower"] = results['Pred'].subtract((self.t_value) * (np.sqrt(self.MSE.values + np.multiply(SE, self.MSE.values))), axis=0)
        results.loc[:, "upper"] = results['Pred'].add((self.t_value) * (np.sqrt(self.MSE.values + np.multiply(SE, self.MSE.values))), axis=0)

        return results



# # loop across square bins
# for x_bin in range(bins_per_side):
#     for y_bin in range(bins_per_side):
#         exploration_image[int(bin_size * (y_bin + .5)), int(bin_size * (x_bin + .5))] = \
#             exploration_avg[y_bin, x_bin]

# increase contrast
# exploration_image = exploration_image * 10
# exploration_image[exploration_image > 255] = 255