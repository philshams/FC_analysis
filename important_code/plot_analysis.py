import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24})
import os
import scipy
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from matplotlib.colors import ListedColormap
import statsmodels.api as sm
import pandas as pd
from sklearn import linear_model
from helper_code.registration_funcs import model_arena, get_arena_details
from helper_code.processing_funcs import speed_colors


def plot_traversals(analysis_object):
    '''     plot all traversals across the arena        '''
    # initialize parameters
    experiments = analysis_object.traversal_parameters['experiments']
    conditions = analysis_object.traversal_parameters['conditions']
    sides = ['back', 'front']
    types = ['spontaneous', 'evoked']
    fast_color = np.array([.5, 1, .5])
    slow_color = np.array([1, .9, .9])
    # loop over spontaneous vs evoked
    for t, type in enumerate(types):
        # loop over experiments and conditions
        for c, (experiment, condition) in enumerate(zip(experiments, conditions)):
            # extract experiments from nested list
            sub_experiments, sub_conditions = extract_experiments(experiment, condition)
            # initialize the arena
            shape = analysis_object.analysis[sub_experiments[0]]['obstacle']['shape']
            obstacle_type = analysis_object.analysis[sub_experiments[0]]['obstacle']['type']
            arena, _, _ = model_arena(shape, False, False, obstacle_type, simulate=False, dark = analysis_object.dark_theme)
            scaling_factor = 100 / arena.shape[0]
            arena_color = cv2.cvtColor(arena, cv2.COLOR_GRAY2RGB)
            # loop over each experiment and condition
            for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
                # loop over back and front sides
                for s, start in enumerate(sides):
                    # loop over each mouse in the experiment
                    for i, mouse in enumerate(analysis_object.analysis[experiment][condition][start + ' traversal']):
                        # find all the paths across the arena
                        path = np.array(analysis_object.analysis[experiment][condition][start + ' traversal'][mouse][t*2])
                        # loop over all paths
                        for trial in range(path.shape[0]):
                            # get the x and y coordinates of the path
                            x_idx = path[trial][0].astype(int); y_idx = path[trial][1].astype(int)
                            # get the duration and distance traveled, to compute net speed
                            time = len(x_idx) / 30
                            dist = np.sum( np.sqrt( np.diff(x_idx )**2 + np.diff(y_idx )**2 ) ) * scaling_factor
                            speed = np.min((40, dist / time))
                            # choose a color accordingly
                            speed_color = ((40 - speed) * slow_color + speed * fast_color) / 40
                            # initialize a mask array
                            mask_arena = np.zeros_like(arena)
                            # loop over each point, drawing line segments on the mask array
                            for j in range(len(x_idx ) -1):
                                x1, y1 = x_idx[j], y_idx[j]
                                x2, y2 = x_idx[ j +1], y_idx[ j +1]
                                cv2.line(mask_arena, (x1, y1), (x2, y2), 1, thickness = 1 + 1* (speed > 15) + 2 * (speed > 25) + 1 * (speed > 35))
                            # draw on the actual array
                            arena_color[mask_arena.astype(bool)] = arena_color[mask_arena.astype(bool)] * speed_color
                            # display the traversals
                            cv2.imshow('traversals - ' + type, arena_color)
                            cv2.waitKey(1)

def plot_speed_traces(analysis_object, speed = 'absolute'):
    '''     plot the speed traces       '''
    # initialize parameters
    experiments = analysis_object.escapes_parameters['experiments']
    conditions = analysis_object.escapes_parameters['conditions']
    # loop over experiments and conditions
    for c, (experiment, condition) in enumerate(zip(experiments, conditions)):
        # extract experiments from nested list
        sub_experiments, sub_conditions = extract_experiments(experiment, condition)
        # get the number of trials
        number_of_trials = get_number_of_trials(sub_experiments, sub_conditions, analysis_object.analysis)
        number_of_mice = get_number_of_mice(sub_experiments, sub_conditions, analysis_object.analysis)
        # initialize array to fill in with each trial's data
        time_axis = np.arange(-10, 15, 1 / 30)
        speed_traces = np.zeros((25 * 30, number_of_trials)) * np.nan
        subgoal_speed_traces = np.zeros((25 * 30, number_of_trials)) * np.nan
        time = np.zeros(number_of_trials)
        end_idx = np.zeros(number_of_trials)
        RT = np.zeros(number_of_trials)
        trial_num = 0
        shape = analysis_object.analysis[sub_experiments[0]]['obstacle']['shape']
        scaling_factor = 100 / shape[0]
        # create custom colormap # colormap = 'seismic'
        colormap = speed_colormap(scaling_factor, n_bins=256, v_min=0, v_max=65)
        # loop over each experiment and condition
        for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
            # loop over each mouse
            for i, mouse in enumerate(analysis_object.analysis[experiment][condition]['speed']):
                # loop over each trial
                for trial in range(len(analysis_object.analysis[experiment][condition]['speed'][mouse])):
                    # extract data
                    trial_speed = [s* scaling_factor * 30  for s in analysis_object.analysis[experiment][condition]['speed'][mouse][trial] ]
                    trial_subgoal_speed = [s* scaling_factor * 30  for s in analysis_object.analysis[experiment][condition]['geo speed'][mouse][trial] ]
                    # filter and add data to the arrays
                    speed_traces[:len(speed_traces), trial_num] = gaussian_filter1d(trial_speed, 2)[:len(speed_traces)]
                    subgoal_speed_traces[:len(speed_traces), trial_num] = gaussian_filter1d(trial_subgoal_speed, 2)[:len(speed_traces)]
                    # add additional data to the arrays
                    time[trial_num] = analysis_object.analysis[experiment][condition]['start time'][mouse][trial]
                    end_idx[trial_num] = analysis_object.analysis[experiment][condition]['end time'][mouse][trial]
                    RT[trial_num] = analysis_object.analysis[experiment][condition]['RT'][mouse][trial]
                    trial_num += 1
        # print out metrics
        print('Number of sessions: ' + str(number_of_mice))
        print('Number of trials: ' + str(number_of_trials))
        num_escapes = np.sum(~np.isnan(end_idx))
        print('percent escape: ' + str(num_escapes / number_of_trials))
        RT_for_quartiles = RT.copy()
        RT_for_quartiles[np.isnan(RT)] = np.inf
        RT_quartiles = np.percentile(RT_for_quartiles, 25), np.percentile(RT_for_quartiles, 50), np.percentile(RT_for_quartiles, 75)
        print('RT quartiles: ' + str(RT_quartiles))
        end_idx_for_quartiles = end_idx.copy()
        end_idx_for_quartiles[np.isnan(end_idx)] = np.inf
        end_quartiles = np.percentile(end_idx_for_quartiles, 25), np.percentile(end_idx_for_quartiles, 50), np.percentile(end_idx_for_quartiles, 75)
        print('to-shelter quartiles: ' + str(end_quartiles))
        # order the data chronologically or by RT
        order = np.argsort(end_idx)
        order = order[::-1]
        # format speed data
        if speed == 'geodesic':
            # speed toward shelter
            z = -subgoal_speed_traces[:, order].T
        else:
            # speed in general
            z = speed_traces[:, order].T
        # separate out the escapes and non-escapes (here, under 6 seconds)
        gap_size = 2
        num_non_escapes = np.sum(np.isnan(end_idx)) + np.sum(end_idx > 6 * 30)
        z_with_gap = np.ones((z.shape[0]+gap_size, z.shape[1])) * np.nan
        z_with_gap[:num_non_escapes, :] = z[:num_non_escapes, :]
        z_with_gap[num_non_escapes+gap_size:, :] = z[num_non_escapes:, :]
        # generate 2 2d grids for the x & y bounds
        fig, ax = plt.subplots(figsize=(12, 5))
        x, y = np.meshgrid(time_axis, np.arange(0, number_of_trials + gap_size))
        ax.set_title('Speed traces: ' + experiment + ' - ' + condition)
        # ax.set_xlabel('time since stimulus onset (s)')
        # ax.set_ylabel('escape trial')
        plt.axis('off')
        # plot speed data
        c = ax.pcolormesh(x, y, z_with_gap, cmap=colormap, vmin=0, vmax=65)
        fig.colorbar(c, ax=ax, label='speed along best path to shelter (cm/s)')
        # plot timing ticks for each trial
        ax.plot([0, 0], [0, number_of_trials + gap_size - 1], color='white', linewidth=2, linestyle = '--')
        ax.plot([6, 6], [0, number_of_trials + gap_size - 1], color='gray', linewidth=2, linestyle='--')

        plt.show()
        plt.savefig(os.path.join(analysis_object.summary_plots_folder, experiment + '_speed_' + condition + '.tif'))
    plt.close('all')


def plot_escape_paths_2(analysis_object):
    '''     plot the escape paths       '''
    # initialize parameters
    experiments = analysis_object.escapes_parameters['experiments']
    conditions = analysis_object.escapes_parameters['conditions']
    homing_vector_color = np.array([.9, .9, 1])
    non_escape_color = np.array([.925, .925, .925])
    edge_vector_color = np.array([.7, 1, .7])
    fps = 30
    escape_duration = 3
    min_distance_to_shelter = 30
    HV_cutoff = .75
    # loop over experiments and conditions
    for c, (experiment, condition) in enumerate(zip(experiments, conditions)):
        # extract experiments from nested list
        sub_experiments, sub_conditions = extract_experiments(experiment, condition)
        # initialize the arena
        shape = analysis_object.analysis[sub_experiments[0]]['obstacle']['shape']
        obstacle_type = analysis_object.analysis[sub_experiments[0]]['obstacle']['type']
        arena, _, _ = model_arena(shape, not sub_conditions[0]=='no obstacle', False, obstacle_type, shelter = not 'no shelter' in sub_experiments[0], simulate=False, dark = analysis_object.dark_theme)
        scaling_factor = 100 / arena.shape[0]
        arena_color = cv2.cvtColor(arena, cv2.COLOR_GRAY2RGB)
        arena_reference = arena_color.copy()
        arena_color[arena_reference == 245] = 255
        get_arena_details(analysis_object, experiment = sub_experiments[0])
        shelter_location = [s / scaling_factor / 10 for s in analysis_object.shelter_location]
        strategies = np.array([0,0,0])
        # create custom colormap # colormap = 'seismic'
        colormap = speed_colormap(scaling_factor, n_bins=256, v_min=0, v_max=65)
        # loop over each experiment and condition
        for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
            # loop over each mouse
            for i, mouse in enumerate(analysis_object.analysis[experiment][condition]['speed']):
                # find all the paths across the arena
                paths = analysis_object.analysis[experiment][condition]['path'][mouse]
                # loop over all paths
                for trial in range(len(paths)):
                    # select the trial
                    if trial > 2 and condition == 'obstacle': continue
                    # get the x and y coordinates of the path
                    x_idx = paths[trial][0][:fps * escape_duration].astype(int)
                    y_idx = paths[trial][1][:fps * escape_duration].astype(int)
                    # needs to start at top
                    if abs(y_idx[0] * scaling_factor - 50) < 25:
                        continue
                    if abs(x_idx[0] * scaling_factor-50) > 20:
                        continue
                    # categorize the escape
                    distance_in_3_secs = np.sqrt((x_idx[fps*2-1]-x_idx[0])**2+(y_idx[fps*2-1]-y_idx[0])**2)* scaling_factor
                    print(distance_in_3_secs)
                    if distance_in_3_secs < 30:
                        path_color = non_escape_color
                        strategies[2] = strategies[2] + 1
                    else:
                        if abs(analysis_object.analysis[experiment][condition]['edginess'][mouse][trial]) < HV_cutoff:
                            path_color = homing_vector_color
                            strategies[0] = strategies[0] + 1
                        else:
                            path_color = edge_vector_color
                            strategies[1] = strategies[1] + 1
                        print(mouse)
                        print(trial)
                        print(abs(analysis_object.analysis[experiment][condition]['edginess'][mouse][trial]))
                    # color by mouse
                    # path_color = homing_vector_color

                    # initialize a mask array
                    mask_arena_for_blur = np.zeros_like(arena)
                    mask_arena = np.zeros_like(arena)
                    # loop over each point, drawing line segments on the mask array
                    for j in range(len(x_idx) - 1):
                        x1, y1 = x_idx[j], y_idx[j]
                        x2, y2 = x_idx[j + 1], y_idx[j + 1]
                        cv2.line(mask_arena, (x1, y1), (x2, y2), 1, thickness=5, lineType = 16)
                        cv2.line(mask_arena_for_blur, (x1, y1), (x2, y2), 2, thickness=1, lineType=16)
                        # end if close to shelter
                        distance_to_shelter = np.sqrt((x1 - shelter_location[0])**2 + (y1 - shelter_location[1])**2)
                        # if distance_to_shelter < min_distance_to_shelter: break
                    # blur line
                    mask_arena_blur = np.ones(arena_color.shape)
                    for i in range(3):
                        if path_color[i] < 1:
                            mask_arena_blur[:, :, i] = (.5 * - gaussian_filter(mask_arena_for_blur.astype(float), sigma=.5)) + 1
                        elif i==1:
                            mask_arena_blur[:, :, i] = (.2 * - gaussian_filter(mask_arena_for_blur.astype(float), sigma=.2)) + 1
                    arena_color[mask_arena.astype(bool)] = arena_color[mask_arena.astype(bool)] * mask_arena_blur[mask_arena.astype(bool)] * path_color
                    # shelter like new
                    arena_color[arena_reference < 245] = arena_reference[arena_reference < 245]
                    # display the traversals
                    cv2.imshow('escapes ' + str(c), arena_color)
                    cv2.waitKey(1)
        #plot bar of strategies
        fig, ax = plt.subplots(figsize=(3, 6))
        normed_strategies = strategies / np.sum(strategies)
        ax.bar(1, normed_strategies[0], .5, color=[1,0,0])
        ax.bar(1, normed_strategies[1], .5, bottom=normed_strategies[0], color=[0,1,0])
        ax.bar(1, normed_strategies[2], .5, bottom=normed_strategies[1]+normed_strategies[0], color=[.5,.5,.5])
        cv2.waitKey(1000)
    plt.show()
    pass




def plot_escape_paths(analysis_object):
    '''     plot the escape paths       '''
    # initialize parameters
    experiments = analysis_object.escapes_parameters['experiments']
    conditions = analysis_object.escapes_parameters['conditions']
    homing_vector_color = np.array([.9, .9, 1])
    non_escape_color = np.array([.925, .925, .925])
    edge_vector_color = np.array([.7, 1, .7])
    fps = 30
    escape_duration = 6
    min_distance_to_shelter = 30
    HV_cutoff = .75
    # loop over experiments and conditions
    for c, (experiment, condition) in enumerate(zip(experiments, conditions)):
        # extract experiments from nested list
        sub_experiments, sub_conditions = extract_experiments(experiment, condition)
        # initialize the arena
        number_of_mice = get_number_of_mice(sub_experiments, sub_conditions, analysis_object.analysis)
        shape = analysis_object.analysis[sub_experiments[0]]['obstacle']['shape']
        obstacle_type = analysis_object.analysis[sub_experiments[0]]['obstacle']['type']
        arena, _, _ = model_arena(shape, not sub_conditions[0]=='no obstacle', False, obstacle_type, simulate=False, dark = analysis_object.dark_theme)
        scaling_factor = 100 / arena.shape[0]
        arena_color = cv2.cvtColor(arena, cv2.COLOR_GRAY2RGB)
        arena_reference = arena_color.copy()
        arena_color[arena_reference == 245] = 255
        get_arena_details(analysis_object, experiment = sub_experiments[0])
        shelter_location = [s / scaling_factor / 10 for s in analysis_object.shelter_location]
        strategies = np.array([0,0,0])
        # create custom colormap # colormap = 'seismic'
        colormap = speed_colormap(scaling_factor, n_bins=256, v_min=0, v_max=65)
        # loop over each experiment and condition
        for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
            # loop over each mouse
            for i, mouse in enumerate(analysis_object.analysis[experiment][condition]['speed']):
                # control analysis
                if analysis_object.analysis_options['control'] and not mouse=='control': continue
                if not analysis_object.analysis_options['control'] and mouse=='control': continue
                # find all the paths across the arena
                paths = analysis_object.analysis[experiment][condition]['path'][mouse]
                # loop over all paths
                for trial in range(len(paths)):
                    # select the trial
                    if (trial) and condition == 'no obstacle': continue
                    # get the x and y coordinates of the path
                    x_idx = paths[trial][0][:fps * escape_duration].astype(int)
                    y_idx = paths[trial][1][:fps * escape_duration].astype(int)
                    #
                    if 'side' in experiment:
                        if (x_idx[0] * scaling_factor) > 50: x_idx = (100/scaling_factor - x_idx).astype(int)
                        if abs(x_idx[0] * scaling_factor-50) < 25: continue
                        if abs(y_idx[0] * scaling_factor-50) > 25: continue
                    else:
                        # needs to start at top
                        if abs(y_idx[0] * scaling_factor-50) < 25: continue
                        if abs(x_idx[0] * scaling_factor-50) > 20: continue
                    # if abs(x_idx[0] * scaling_factor-50) < 25: continue
                    # if abs(y_idx[0] * scaling_factor-50) > 35: continue
                    # categorize the escape
                    time_to_shelter = analysis_object.analysis[experiment][condition]['end time'][mouse][trial]
                    if np.isnan(time_to_shelter) or time_to_shelter > (escape_duration*fps):
                        path_color = non_escape_color
                        strategies[2] = strategies[2] + 1
                        # continue
                    else:
                        if abs(analysis_object.analysis[experiment][condition]['edginess'][mouse][trial]) < HV_cutoff:
                            path_color = homing_vector_color
                            strategies[0] = strategies[0] + 1
                        else:
                            path_color = edge_vector_color
                            strategies[1] = strategies[1] + 1
                        print(mouse)
                        print(trial)
                        print(abs(analysis_object.analysis[experiment][condition]['edginess'][mouse][trial]))
                    # color by mouse
                    # path_color = homing_vector_color

                    # initialize a mask array
                    mask_arena_for_blur = np.zeros_like(arena)
                    mask_arena = np.zeros_like(arena)
                    # loop over each point, drawing line segments on the mask array
                    for j in range(len(x_idx) - 1):
                        x1, y1 = x_idx[j], y_idx[j]
                        x2, y2 = x_idx[j + 1], y_idx[j + 1]
                        cv2.line(mask_arena, (x1, y1), (x2, y2), 1, thickness=5, lineType = 16)
                        cv2.line(mask_arena_for_blur, (x1, y1), (x2, y2), 2, thickness=1, lineType=16)
                        # end if close to shelter
                        distance_to_shelter = np.sqrt((x1 - shelter_location[0])**2 + (y1 - shelter_location[1])**2)
                        if distance_to_shelter < min_distance_to_shelter: break
                    # blur line
                    mask_arena_blur = np.ones(arena_color.shape)
                    for i in range(3):
                        if path_color[i] < 1:
                            mask_arena_blur[:, :, i] = (.5 * - gaussian_filter(mask_arena_for_blur.astype(float), sigma=.5)) + 1
                        elif i==1:
                            mask_arena_blur[:, :, i] = (.2 * - gaussian_filter(mask_arena_for_blur.astype(float), sigma=.2)) + 1
                    arena_color[mask_arena.astype(bool)] = arena_color[mask_arena.astype(bool)] * mask_arena_blur[mask_arena.astype(bool)] * path_color
                    # shelter like new
                    arena_color[arena_reference < 245] = arena_reference[arena_reference < 245]
                    # display the traversals
                    cv2.imshow('escapes ' + str(c), arena_color)
                    cv2.waitKey(1)
        #plot bar of strategies
        print(np.sum(strategies))
        # fig, ax = plt.subplots(figsize=(3, 6))
        # normed_strategies = strategies / np.sum(strategies)
        # ax.bar(1, normed_strategies[0], .5, color=[1,0,0])
        # ax.bar(1, normed_strategies[1], .5, bottom=normed_strategies[0], color=[0,1,0])
        # ax.bar(1, normed_strategies[2], .5, bottom=normed_strategies[1]+normed_strategies[0], color=[.5,.5,.5])
        # print(normed_strategies)
        cv2.waitKey(100)
    # plt.show()
    pass


def random_agent():

    y_eval = 40
    HV_cutoff = .75
    starting_point = (50,15)

    # get threshold HV to obst edge angle
    threshold_point_wall = (50 + HV_cutoff * 25, 50)
    slope = (threshold_point_wall[1] - starting_point[1]) / (threshold_point_wall[0] - starting_point[0])
    intercept = starting_point[1]- starting_point[0] * slope
    x_eval = (y_eval - intercept) / slope
    angle_eval = np.angle(y_eval*1j + x_eval)

    HV = angle_eval / (np.pi/2)
    OE = (np.pi / 4 - angle_eval) / (np.pi / 2)
    N = .5
    print('HV percent:' + str(HV))
    print('OE percent:' + str(OE))
    print('N percent:' + str(N))

    fig, ax = plt.subplots(figsize=(3, 6))
    normed_strategies = strategies / np.sum(strategies)
    ax.bar(1, HV, .5, color=[1, 0, 0])
    ax.bar(1, OE, .5, bottom=HV, color=[0, 1, 0])
    ax.bar(1, N, .5, bottom=HV+OE, color=[.5, .5, .5])

    plt.show()


def plot_edginess(analysis_object):
    # initialize parameters
    experiments = analysis_object.escapes_parameters['experiments']
    conditions = analysis_object.escapes_parameters['conditions']
    fps = 30
    escape_duration = 6
    HV_cutoff = .72
    ETD = 7
    traj_loc = 40
    # initialize figure
    fig, ax = plt.subplots(figsize=(9, 9))
    title = 'Escape strategy'
    y_label = 'edginess'
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xticks(np.arange(len(experiments)))
    # ax2.set_xticklabels(x_labels)
    ax.set_xlim([-1, len(3*experiments)-1])
    # ax.set_ylim([-.03, 1.03])

    # initialize figure
    fig2, ax2 = plt.subplots(figsize=(9, 9))
    title = 'Escape strategy correlation'
    # y_label = 'edginess'
    x_label = 'previous edginess'
    # ax2.set_title(title)
    # ax2.set_ylabel(y_label)
    ax2.set_ylim([-.05, 1.19])


    colors = [[.2,.2,.2], [.2,.2,.2]]
    # loop over experiments and conditions
    for c, (experiment, condition) in enumerate(zip(experiments, conditions)):
        # extract experiments from nested list
        sub_experiments, sub_conditions = extract_experiments(experiment, condition)
        # get the number of trials
        number_of_trials = get_number_of_trials(sub_experiments, sub_conditions, analysis_object.analysis)
        number_of_mice = get_number_of_mice(sub_experiments, sub_conditions, analysis_object.analysis)
        # initialize array to fill in with each trial's data
        edginess = np.ones(number_of_trials) * np.nan
        prev_edginess = np.ones(number_of_trials) * np.nan
        time_in_center = np.ones(number_of_trials) * np.nan
        end_idx = np.ones(number_of_trials) * np.nan
        trial_num = -1
        shape = analysis_object.analysis[sub_experiments[0]]['obstacle']['shape']
        scaling_factor = 100 / shape[0]
        # create custom colormap # colormap = 'seismic'
        colormap = speed_colormap(scaling_factor, n_bins=256, v_min=0, v_max=65)
        # loop over each experiment and condition
        for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
            # loop over each mouse
            for i, mouse in enumerate(analysis_object.analysis[experiment][condition]['speed']):
                if analysis_object.analysis_options['control'] and not mouse=='control': continue
                if not analysis_object.analysis_options['control'] and mouse=='control': continue
                # loop over each trial
                for trial in range(len(analysis_object.analysis[experiment][condition]['end time'][mouse])):
                    trial_num += 1
                    if trial > 2: continue
                    if condition == 'obstacle' and trial < 2: continue
                    # if 'up' in experiment and trial < 2: continue
                    # if not trial: continue
                    end_idx[trial_num] = analysis_object.analysis[experiment][condition]['end time'][mouse][trial]
                    if np.isnan(end_idx[trial_num]): continue
                    if (end_idx[trial_num] > escape_duration * fps): continue
                    elif trial and 'quick' in experiment: continue
                    # elif condition == 'obstacle' and trial > 0: continue
                    # elif trial > 2: continue
                    # skip certain trials
                    y_start = analysis_object.analysis[experiment][condition]['path'][mouse][trial][1][0] * scaling_factor
                    x_start = analysis_object.analysis[experiment][condition]['path'][mouse][trial][0][0] * scaling_factor
                    if y_start > 23: continue
                    if abs(x_start-50) > 23: continue #22
                    # add data
                    edginess[trial_num] = analysis_object.analysis[experiment][condition]['edginess'][mouse][trial]
                    # get previous edginess
                    SH_data = analysis_object.analysis[experiment][condition]['SR'][mouse][trial]
                    SR = np.array(SH_data[0])
                    x_edge = np.array(SH_data[4])
                    SR_time = np.array(SH_data[3])
                    RT = analysis_object.analysis[experiment][condition]['RT'][mouse][trial]
                    # RT = 0
                    # get position for the trial
                    x_pos = analysis_object.analysis[experiment][condition]['path'][mouse][trial][0][int(RT * 30):] * scaling_factor
                    y_pos = analysis_object.analysis[experiment][condition]['path'][mouse][trial][1][int(RT * 30):] * scaling_factor
                    # get the sidedness of the escape
                    mouse_at_center = np.argmin(abs(y_pos - traj_loc))

                    # get line to the closest edge
                    y_edge = 50

                    # only use recent escapes
                    if len(SR) >= (ETD + 1): SR = SR[-ETD:]

                    # get line to the closest edge, exclude escapes to other edge
                    MOE = 10.2  # 20 #10.2
                    if x_edge > 50: SR = SR[SR > 25 + MOE]  # 35
                    else: edge_proximity = SR = SR[SR < 75 - MOE]  # 65

                    # take the mean of the prev homings
                    if SR.size: x_repetition = np.mean(SR)

                    # NOW DO THE EDGINESS ANALYSIS, WITH REPETITION AS THE REAL DATA
                    # do line from starting position to shelter
                    y_pos_end = 86.5
                    x_pos_end = 50
                    x_pos_start = x_pos[0]
                    y_pos_start = y_pos[0]
                    slope = (y_pos_end - y_pos_start) / (x_pos_end - x_pos_start)
                    intercept = y_pos_start - x_pos_start * slope
                    if SR.size: distance_to_line = abs(traj_loc - slope * x_repetition - intercept) / np.sqrt((-slope) ** 2 + (1) ** 2)
                    homing_vector_at_center = (traj_loc - intercept) / slope

                    # do line from starting position to edge position
                    slope = (y_edge - y_pos_start) / (x_edge - x_pos_start)
                    intercept = y_pos_start - x_pos_start * slope
                    distance_to_edge = abs(traj_loc - slope * x_repetition - intercept) / np.sqrt((-slope) ** 2 + (1) ** 2)

                    # compute the max possible deviation
                    edge_vector_at_center = (traj_loc - intercept) / slope
                    line_to_edge_offset = abs(homing_vector_at_center - edge_vector_at_center)  # + 5

                    # get index at center point (wall location)
                    prev_edginess[trial_num] = (distance_to_line - distance_to_edge + 25) / 50
                    prev_edginess[trial_num] = (distance_to_line - distance_to_edge + line_to_edge_offset) / (2 * line_to_edge_offset)

                    # print(mouse)
                    # print(prev_edginess[trial_num])
                    # print(edginess[trial_num])
                    time_in_center[trial_num] = analysis_object.analysis[experiment][condition]['in center'][mouse][trial] /30
                    if not time_in_center[trial_num]:
                        print(experiment)
                        print(mouse)


        # print out metrics
        print('Number of sessions: ' + str(number_of_mice))
        print('Number of trials: ' + str(number_of_trials))
        # plot escapes within time limit a la box plot
        data = edginess
        data = abs(edginess)
        data_for_box_plot = data[~np.isnan(data)]
        # make a boxplot
        median = np.median(data_for_box_plot)
        lower = np.percentile(data_for_box_plot, 25)
        upper = np.percentile(data_for_box_plot, 75)
        # plot the median and IQR
        # ax.errorbar(3*c - .6, median, yerr=np.array([[median - lower], [upper - median]]), color=colors[0], capsize=8, capthick=1, alpha=1, linewidth=1)
        # ax.scatter(3*c - .6, median, color=colors[0], s=150, alpha=1)
        # plot each trial
        scatter_axis = np.ones_like(data_for_box_plot) *3*c - .15
        for i in range(len(data_for_box_plot)):
            difference = abs(data_for_box_plot[i] - data_for_box_plot)
            difference[i] = np.inf
            if np.min(difference) == 0: scatter_axis[i] = np.random.normal(3*c - .15, 0.25)
            elif np.min(difference) < .008: scatter_axis[i] = np.random.normal(3*c - .15, 0.15)
        # ax.scatter(scatter_axis, data_for_box_plot, color=colors[c], s=10, alpha=.8, edgecolors='black', linewidth=1)  # [.2,.2,.2])
        ax.scatter(scatter_axis[data_for_box_plot>HV_cutoff], data_for_box_plot[data_for_box_plot>HV_cutoff], color=[0,.8,0], s=25, alpha=1, edgecolors=[0,.2,0], linewidth=1)  # [.2,.2,.2])
        ax.scatter(scatter_axis[data_for_box_plot<HV_cutoff], data_for_box_plot[data_for_box_plot<HV_cutoff], color=[1,0,0], s=25, alpha=1, edgecolors=[.2,0,0], linewidth=1)  # [.2,.2,.2])
        #do kde
        kde = fit_kde(data_for_box_plot, bw=.04)  # .04
        plot_kde(ax, kde, data_for_box_plot, z=3*c + .3, vertical=True, normto=1.5, color=colors[0], violin=False, clip=True)  # True)
        # skew = scipy.stats.skew(data_for_box_plot)
        # print('actual skew: ' + str(skew))
        # control_skew = []
        # for i in range(1000):
        #     control_dist = np.random.normal(int(c>0), np.std(data_for_box_plot), len(data_for_box_plot)) # np.mean(data_for_box_plot)
        #     control_skew.append(scipy.stats.skew(control_dist))
        # print('skew: p = ')
        # print(scipy.stats.percentileofscore(control_skew, skew))

        # plot the correlation
        # data_for_correlation = prev_edginess[~np.isnan(data)]
        data_for_correlation = time_in_center[~np.isnan(data)]

        ax2.scatter(data_for_correlation, data_for_box_plot, color=[.5, .5, .5], s=25, alpha=1, edgecolors=[.2, .2, .2], linewidth=1)

        # do linear regression
        r, p = scipy.stats.pearsonr(data_for_correlation, data_for_box_plot)
        print(r, p)

        # plot linear regression
        order = np.argsort(data_for_correlation)
        data_for_correlation = data_for_correlation[order]
        data_for_box_plot = data_for_box_plot[order]

        LR = LRPI(t_value=1)
        LR.fit(data_for_correlation, data_for_box_plot)
        prediction = LR.predict(data_for_correlation)

        ax2.plot(data_for_correlation, prediction['Pred'].values, color=[.0, .0, .0], linewidth=1, linestyle='--', alpha=.7)
        ax2.fill_between(data_for_correlation, prediction['lower'].values, prediction['upper'].values, color=[.2, .2, .2], alpha=.05)  # 6

    plt.show()
    pass


def plot_efficiency(analysis_object):
    # initialize parameters
    experiments = analysis_object.escapes_parameters['experiments']
    conditions = analysis_object.escapes_parameters['conditions']
    fps = 30
    escape_duration = 6
    HV_cutoff = .72
    ETD = 7

    colors = [[.2,.2,.2], [.2,.2,.2]]
    # loop over experiments and conditions
    for c, (experiment, condition) in enumerate(zip(experiments, conditions)):
        # extract experiments from nested list
        sub_experiments, sub_conditions = extract_experiments(experiment, condition)
        # get the number of trials
        number_of_trials = get_number_of_trials(sub_experiments, sub_conditions, analysis_object.analysis)
        number_of_mice = get_number_of_mice(sub_experiments, sub_conditions, analysis_object.analysis)
        # initialize array to fill in with each trial's data
        efficiency = np.ones(number_of_trials) * np.nan
        efficiency_RT = np.ones(number_of_trials) * np.nan
        num_prev_homings = np.ones(number_of_trials) * np.nan
        time = np.ones(number_of_trials) * np.nan
        trials = np.ones(number_of_trials) * np.nan
        end_idx = np.ones(number_of_trials) * np.nan
        trial_num = -1
        shape = analysis_object.analysis[sub_experiments[0]]['obstacle']['shape']
        scaling_factor = 100 / shape[0]
        # create custom colormap # colormap = 'seismic'
        colormap = speed_colormap(scaling_factor, n_bins=256, v_min=0, v_max=65)
        # loop over each experiment and condition
        for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
            # loop over each mouse
            for i, mouse in enumerate(analysis_object.analysis[experiment][condition]['full path length']):
                # control analysis
                if analysis_object.analysis_options['control'] and not mouse=='control': continue
                if not analysis_object.analysis_options['control'] and mouse=='control': continue
                # loop over each trial
                for trial in range(len(analysis_object.analysis[experiment][condition]['end time'][mouse])):
                    trial_num += 1
                    if trial > 2: continue
                    # if condition == 'obstacle' and trial < 2: continue
                    # if 'up' in experiment and trial < 2: continue
                    # if not trial: continue
                    end_idx[trial_num] = analysis_object.analysis[experiment][condition]['end time'][mouse][trial]
                    if (end_idx[trial_num] > escape_duration * fps): continue
                    # skip certain trials
                    y_start = analysis_object.analysis[experiment][condition]['path'][mouse][trial][1][0] * scaling_factor
                    x_start = analysis_object.analysis[experiment][condition]['path'][mouse][trial][0][0] * scaling_factor
                    if y_start > 23: continue
                    if abs(x_start-50) > 23: continue #22
                    # add data
                    efficiency[trial_num] = analysis_object.analysis[experiment][condition]['optimal path length'][mouse][trial] / \
                                            analysis_object.analysis[experiment][condition]['full path length'][mouse][trial]

                    efficiency_RT[trial_num] = analysis_object.analysis[experiment][condition]['optimal RT path length'][mouse][trial] / \
                                               analysis_object.analysis[experiment][condition]['RT path length'][mouse][trial]

                    time[trial_num] = analysis_object.analysis[experiment][condition]['start time'][mouse][trial]
                    trials[trial_num] = int(trial)
                    # compute SHs
                    # get the stored exploration plot - proportion of time at each location
                    SH_data = analysis_object.analysis[experiment][condition]['SR'][mouse][trial]
                    SR = np.array(SH_data[0])
                    x_edge = SH_data[4]
                    # only use recent escapes
                    if len(SR) >= (ETD + 1): SR = SR[-ETD:]
                    # get line to the closest edge, exclude escapes to other edge
                    num_prev_homings[trial_num] = int(np.min((2, np.sum( abs(SR - x_edge) < 5))))

                    if efficiency[trial_num] > .8 and not num_prev_homings[trial_num]:
                        print(mouse)
                        print(trial)

        # print out metrics
        print('Number of sessions: ' + str(number_of_mice))
        print('Number of trials: ' + str(number_of_trials))

        # plot data

        end_idx = np.array([e/30 for e in end_idx])
        end_idx[np.isnan(efficiency)] = np.nan
        for i, data in enumerate([efficiency, end_idx]): #efficiency_RT,
            for x_data in [num_prev_homings, trials, time]: #time, trials,
                # initialize figure
                fig, ax = plt.subplots(figsize=(9, 9))
                title = 'Escape spatial efficiency'
                y_label = 'torosity'
                # ax.set_title(title)
                # ax.set_ylabel(y_label)
                # ax.set_xticks(np.arange(len(experiments)))
                # ax2.set_xticklabels(x_labels)
                # ax.set_xlim([-1, len(3 * 3) - 1])
                if not i: ax.set_ylim([-.03, 1.03])
                else: ax.set_ylim([-.1, 6.5])

                # plot escapes within time limit a la box plot
                # data[time > 20] = np.nan
                # x_data[x_data > 20] = np.nan

                # only plot escapes
                data_for_box_plot = data[~np.isnan(data)]
                x_data = x_data[~np.isnan(data)]

                if np.max(x_data) < 3:
                    plot_data_x, plot_data = x_data, data_for_box_plot
                    ax.set_xticks([0,1,2])
                else:
                    plot_data_x, plot_data = x_data[x_data < 20], data_for_box_plot[x_data < 20]

                # get the correlation
                r, p = scipy.stats.pearsonr(plot_data_x, plot_data)
                print(r, p)

                # jitter the axis
                scatter_axis = plot_data_x.copy()
                for j in range(len(scatter_axis)):
                    difference = abs(plot_data[j] - plot_data)
                    difference[j] = np.inf
                    if np.min(difference) < (.001 * np.max(plot_data)):
                        scatter_axis[j] = scatter_axis[j]+ np.random.normal(0, 0.03)

                # plot each trial
                ax.scatter(scatter_axis, plot_data, color=[.5, .5, .5], s=25, alpha=1, edgecolors=[.2, .2, .2], linewidth=1)

                # do a linear regression
                order = np.argsort(plot_data_x)
                plot_data_x = plot_data_x[order]
                plot_data = plot_data[order]

                LR = LRPI(t_value=1)
                LR.fit(plot_data_x, plot_data)
                prediction = LR.predict(plot_data_x)

                ax.plot(plot_data_x, prediction['Pred'].values, color=[.0, .0, .0], linewidth=1, linestyle='--', alpha=.7)
                ax.fill_between(plot_data_x, prediction['lower'].values, prediction['upper'].values, color=[.2,.2,.2], alpha=.05)  # 6



    plt.show()
    pass

def plot_exploration(analysis_object):
    '''     plot the average exploration heat map       '''
    # initialize parameters
    experiments = analysis_object.exploration_parameters['experiments']
    conditions = analysis_object.exploration_parameters['conditions']
    # loop over experiments and conditions
    for c, (experiment, condition) in enumerate(zip(experiments, conditions)):
        # extract experiments from nested list
        sub_experiments, sub_conditions = extract_experiments(experiment, condition)
        # get the number of trials
        number_of_mice = get_number_of_mice(sub_experiments, sub_conditions, analysis_object.analysis)
        mouse_num = 0
        # initialize array to fill in with each mouse's data
        shape = analysis_object.analysis[sub_experiments[0]]['obstacle']['shape']
        exploration = np.zeros((shape[0], shape[1], number_of_mice))
        # loop over each experiment and condition
        for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
            # loop over each mouse
            for mouse in analysis_object.analysis[experiment][condition]['exploration']:
                # fill array with each mouse's data
                exploration[:, :, mouse_num] = analysis_object.analysis[experiment][condition]['exploration'][mouse]
                mouse_num += 1

        # average all mice data
        exploration_all = np.mean(exploration, 2)

        # make an image out of it
        exploration_image = exploration_all.copy()
        exploration_image = (exploration_image / np.percentile(exploration_image, 98) * 255)
        exploration_image[exploration_image > 255] = 255
        # median filter
        # exploration_image = scipy.signal.medfilt2d(exploration_image, kernel_size=5)
        # present it
        exploration_image = cv2.cvtColor(255 - exploration_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        # cv2.imshow('heat map', exploration_image)
        # cv2.waitKey(100)
        # save results
        scipy.misc.imsave(os.path.join(analysis_object.summary_plots_folder, experiment + '_exploration_' + condition + '.tif'), exploration_image[:,:,::-1])


        shape = analysis_object.analysis[sub_experiments[0]]['obstacle']['shape']
        obstacle_type = analysis_object.analysis[sub_experiments[0]]['obstacle']['type']
        _, _, shelter_roi = model_arena(shape, False, False, obstacle_type, simulate=False, dark=analysis_object.dark_theme)
        percent_in_shelter = []
        for m in range( exploration.shape[2]):
            mouse_exploration = exploration[:,:,m]
            percent_in_shelter.append( np.sum(mouse_exploration*shelter_roi) )





class LRPI:
    '''     linear regression you can get prediction interval from      '''
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


def extract_experiments(experiment, condition):
    '''     extract experiments from nested list     '''
    if type(experiment) == list:
        sub_experiments = experiment
        sub_conditions = condition
    else:
        sub_experiments = [experiment]
        sub_conditions = [condition]

    return sub_experiments, sub_conditions

def get_number_of_trials(sub_experiments, sub_conditions, analysis):
    '''     find out how many trials in each condition, for data initialization     '''
    # initialize the number of trials
    number_of_trials = 0
    max_number_of_trials = 99999
    # loop over each experiment/condition
    for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
        # loop over each mouse
        for i, mouse in enumerate(analysis[experiment][condition]['speed']):
            # and count up the number of trials
            number_of_trials += np.min((max_number_of_trials, len(analysis[experiment][condition]['speed'][mouse])))
        if not number_of_trials: continue
    return number_of_trials

def get_number_of_mice(sub_experiments, sub_conditions, analysis):
    '''     find out how many mice in each condition, for data initialization     '''
    # initialize the number of mice
    number_of_mice = 0
    max_number_of_mice = 99999
    list_of_mice = []
    # loop over each experiment/condition
    for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
        # and count up the number of mice
        key = list(analysis[experiment][condition].keys())[-1]
        number_of_mice += len(analysis[experiment][condition][key])
        list_of_mice.append(list(analysis[experiment][condition][key].keys()))
    print('number of mice: ' + str(len(np.unique(list(flatten(list_of_mice))))))

    return number_of_mice

def speed_colormap(scaling_factor, n_bins=256, v_min = 0, v_max = 65):
    '''     create a custom, gray-blue-green colormap       '''
    # create an empty colormap
    new_colors = np.ones((n_bins, 4))
    # loop over each color entry
    for c in range(new_colors.shape[0]):
        # get speed
        speed = (c * (v_max-v_min) / new_colors.shape[0] + v_min) / 30 / scaling_factor
        # get speed color
        _, speed_color = speed_colors(speed, plotting = True)
        # insert into color array
        new_colors[c, :3] = speed_color
    # insert into colormap
    colormap = ListedColormap(new_colors)
    return colormap

def flatten(iterable):
    '''       flatten a nested list       '''
    it = iter(iterable)
    for e in it:
        if isinstance(e, (list, tuple)):
            for f in flatten(e):
                yield f
        else:
            yield e

def fit_kde(x, **kwargs):
    """ Fit a KDE using StatsModels.
        kwargs is useful to pass stuff to the fit, e.g. the binwidth (bw)"""
    x = np.array(x).astype(np.float)
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit(**kwargs)  # Estimate the densities
    return kde

def plot_shaded_withline(ax, x, y, z=None, label=None, violin=True, **kwargs):
    """[Plots a curve with shaded area and the line of the curve clearly visible]

    Arguments:
        ax {[type]} -- [matplotlib axis]
        x {[np.array, list]} -- [x data]
        y {[np.array, list]} -- [y data]

    Keyword Arguments:
        z {[type]} -- [description] (default: {None})
        label {[type]} -- [description] (default: {None})
        alpha {float} -- [description] (default: {.15})
    """
    # if z is not None:
    fill_alpha = .2
    line_alpha = .4

    redness = kwargs['color'][0]
    if type(redness) != str:
        if redness > .5: fill_alpha += .1

    ax.fill_betweenx(y, z + x, z, alpha=fill_alpha, **kwargs)
    ax.plot(z + x, y, alpha=line_alpha, label=label, **kwargs)
    if violin:
        ax.fill_betweenx(y, z - x, z, alpha=fill_alpha, **kwargs)
        ax.plot(z - x, y, alpha=line_alpha, label=label, **kwargs)
    # else:
    #     ax.fill_between(x, y, alpha=alpha, **kwargs)

def plot_kde(ax, kde, data, z, vertical=False, normto=None, label=None, violin=True, clip=False, **kwargs):
    """[Plots a KDE distribution. Plots first the shaded area and then the outline.
       KDE can be oriented vertically, inverted, normalised...]

    Arguments:
        ax {[plt.axis]} -- [ax onto which to plot]
        kde {[type]} -- [KDE fitted with statsmodels]
        z {[type]} -- [value used to shift the plotted curve. e.g for a horizzontal KDE if z=0 the plot will lay on the X axis]

    Keyword Arguments:
        invert {bool} -- [mirror the KDE plot relative to the X or Y axis, depending on ortentation] (default: {False})
        vertical {bool} -- [plot KDE vertically] (default: {False})
        normto {[float]} -- [normalise the KDE so that the peak of the distribution is at a certain value] (default: {None})
        label {[string]} -- [label for the legend] (default: {None})

    Returns:
        ax, kde
    """
    if vertical:
        x = kde.density
        y = kde.support
    else:
        x, y = kde.support, kde.density

    if clip:

        if np.max(data) > 1:
            x = x[(y > 0)]  # * (y < 1)]
            y = y[(y > 0)]  # * (y < 1)]
        else:
            x = x[(y > 0) * (y < 1)]
            y = y[(y > 0) * (y < 1)]

    if normto is not None:
        if not vertical:
            y = y / np.max(y) * normto
        else:
            x = x / np.max(x) * normto

    plot_shaded_withline(ax, x, y, z=z, violin=violin, **kwargs)

    return ax, kde


