import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import scipy
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import statsmodels.api as sm
import pandas as pd
from sklearn import linear_model
from helper_code.registration_funcs import model_arena, get_arena_details
from helper_code.processing_funcs import speed_colors
from helper_code.analysis_funcs import *
plt.rcParams.update({'font.size': 30})

def plot_traversals(self):
    '''     plot all traversals across the arena        '''
    # initialize parameters
    sides = ['back', 'front']
    types = ['spontaneous']
    fast_color = np.array([.5, 1, .5])
    slow_color = np.array([1, .9, .9])
    edge_vector_color = np.array([0, .7, .99])**5
    homing_vector_color = np.array([.95, 0, .85])**5

    edge_vector_color_bar = np.array([.65, .85, 1])
    homing_vector_color_bar = np.array([1, .85, .95])
    non_escape_color_bar = np.array([0, 0, 0])
    p = 0
    HV_cutoff = .75
    # initialize figures
    fig, _, _, ax, _, _ = initialize_figures(self, traversals = 2)
    # loop over spontaneous vs evoked
    for t, type in enumerate(types):
        # loop over experiments and conditions
        for c, (experiment, condition) in enumerate(zip(self.experiments, self.conditions)):
            # extract experiments from nested list
            sub_experiments, sub_conditions = extract_experiments(experiment, condition)
            # initialize the arena
            arena, arena_color, scaling_factor = initialize_arena(self, sub_experiments, sub_conditions)
            # initialize edginess
            all_traversals_edginess = []
            # loop over back and front sides
            for s, start in enumerate(sides):
                m = 0
                # loop over each experiment and condition
                for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
                    # loop over each mouse in the experiment
                    for i, mouse in enumerate(self.analysis[experiment][condition][start + ' traversal']):
                        # find all the paths across the arena
                        traversal = self.analysis[experiment][condition][start + ' traversal'][mouse]
                        if traversal:
                            all_traversals_edginess.append(traversal[4])
                            traversal = np.array(traversal[t*2])
                        else: continue
                        m += 1
                print(m)
                        # loop over all paths
                        # for trial in range(traversal.shape[0]):
                        #     display_traversal(arena, arena_color, fast_color, scaling_factor, slow_color, traversal, trial, type)
            # plot the edginess of traversals
            # get the data
            plot_data = abs(np.array(list(flatten(all_traversals_edginess))))
            # plot each trial
            scatter_axis = scatter_the_axis(p, plot_data)
            ax.scatter(scatter_axis[plot_data > HV_cutoff], plot_data[plot_data > HV_cutoff], color=edge_vector_color[::-1], s=30, zorder=99)
            ax.scatter(scatter_axis[plot_data < HV_cutoff], plot_data[plot_data < HV_cutoff], color=homing_vector_color[::-1], s=30, zorder=99)
            # do kde
            kde = fit_kde(plot_data, bw=.04)  # .04
            plot_kde(ax, kde, plot_data, z=3 * p + .3, vertical=True, normto=1, color=[.5, .5, .5], violin=False, clip=True)  # True)
            p+=1

            print('Traversals - ' + type + ' - ' + start + ' - ' + self.labels[c])
            HV = 100 * np.sum(plot_data < HV_cutoff) / len(plot_data)
            OE = 100 * np.sum(plot_data > HV_cutoff) / len(plot_data)
            print('HV: ' + str(HV))
            print('OE: ' + str(OE))
            print(len(plot_data))
            fig2 = plot_strategies([HV, OE, 0], homing_vector_color_bar, non_escape_color_bar, edge_vector_color_bar)
            fig2.savefig(os.path.join(self.summary_plots_folder, 'Traversals - ' + type + ' - ' + self.labels[c] + '.tif'), format='tif', bbox_inches='tight', pad_inches=0)
            fig2.savefig(os.path.join(self.summary_plots_folder, 'Traversals - ' + type + ' - ' + self.labels[c] + '.eps'), format='eps', bbox_inches='tight', pad_inches=0)
            # save image
            # scipy.misc.imsave(os.path.join(self.summary_plots_folder, 'Traversals - ' + type + ' - ' + self.labels[c] + '.tif'), arena_color[:, :, ::-1])
        # save the plot
        fig.savefig(os.path.join(self.summary_plots_folder, 'Traversal edginess comparison.tif'), format='tif', bbox_inches='tight', pad_inches=0)
        fig.savefig(os.path.join(self.summary_plots_folder, 'Traversal edginess comparison.eps'), format='eps', bbox_inches='tight', pad_inches=0)
        plt.show()


def plot_speed_traces(self, speed = 'absolute'):
    '''     plot the speed traces       '''
    # loop over experiments and conditions
    for c, (experiment, condition) in enumerate(zip(self.experiments, self.conditions)):
        # extract experiments from nested list
        sub_experiments, sub_conditions = extract_experiments(experiment, condition)
        # get the number of trials
        number_of_trials = get_number_of_trials(sub_experiments, sub_conditions, self.analysis)
        number_of_mice = get_number_of_mice(sub_experiments, sub_conditions, self.analysis)
        RT, end_idx, scaling_factor, speed_traces, subgoal_speed_traces, time, time_axis, trial_num = \
            initialize_variables(number_of_trials, self,sub_experiments)
        # create custom colormap
        colormap = speed_colormap(scaling_factor, n_bins=256, v_min=0, v_max=65)
        # loop over each experiment and condition
        for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
            # loop over each mouse
            for i, mouse in enumerate(self.analysis[experiment][condition]['speed']):
                # control analysis
                if self.analysis_options['control'] and not mouse=='control': continue
                if not self.analysis_options['control'] and mouse=='control': continue
                # loop over each trial
                for trial in range(len(self.analysis[experiment][condition]['speed'][mouse])):
                    trial_num = fill_in_trial_data(RT, condition, end_idx, experiment, mouse, scaling_factor, self,
                                       speed_traces, subgoal_speed_traces, time, trial, trial_num)
        # print some useful metrics
        print_metrics(RT, end_idx, number_of_mice, number_of_trials)
        # put the speed traces on the plot
        fig = show_speed_traces(colormap, condition, end_idx, experiment, number_of_trials, speed, speed_traces, subgoal_speed_traces, time_axis)
        # save the plot
        fig.savefig(os.path.join(self.summary_plots_folder,'Speed traces - ' + self.labels[c] + '.tif'), format='tif', bbox_inches = 'tight', pad_inches = 0)
        fig.savefig(os.path.join(self.summary_plots_folder,'Speed traces - ' + self.labels[c] + '.eps'), format='eps', bbox_inches = 'tight', pad_inches = 0)
    plt.show()




def plot_escape_paths(self):
    '''     plot the escape paths       '''
    # initialize parameters
    edge_vector_color = np.array([.65, .85, 1])
    homing_vector_color = np.array([1, .85, .95])
    non_escape_color = np.array([.925, .925, .925])

    edge_vector_color = np.array([1, .85, .85])
    homing_vector_color = np.array([.725, .725, .725])
    non_escape_color = np.array([0,0,0])

    fps = 30
    escape_duration = 9 #6 #9 for food
    min_distance_to_shelter = 30
    HV_cutoff = .75 #.75
    # loop over experiments and conditions
    for c, (experiment, condition) in enumerate(zip(self.experiments, self.conditions)):
        # extract experiments from nested list
        sub_experiments, sub_conditions = extract_experiments(experiment, condition)
        # initialize the arena
        arena, arena_color, scaling_factor = initialize_arena(self, sub_experiments, sub_conditions)
        # more arena stuff for this analysis type
        arena_reference = arena_color.copy()
        arena_color[arena_reference == 245] = 255
        get_arena_details(self, experiment=sub_experiments[0])
        shelter_location = [s / scaling_factor / 10 for s in self.shelter_location]
        # initialize strategy array
        strategies = np.array([0,0,0])
        # loop over each experiment and condition
        for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
            # loop over each mouse
            for i, mouse in enumerate(self.analysis[experiment][condition]['speed']):
                print(mouse)
                # control analysis
                if self.analysis_options['control'] and not mouse=='control': continue
                if not self.analysis_options['control'] and mouse=='control': continue
                # show escape paths
                show_escape_paths(HV_cutoff, arena, arena_color, arena_reference, c, condition, edge_vector_color, escape_duration, experiment, fps,
                                  homing_vector_color, min_distance_to_shelter, mouse, non_escape_color, scaling_factor, self, shelter_location, strategies)
        # save image
        scipy.misc.imsave(os.path.join(self.summary_plots_folder, 'Escape paths - ' + self.labels[c] + '.tif'), arena_color[:,:,::-1])
        # plot a stacked bar of strategies
        fig = plot_strategies(strategies, homing_vector_color, non_escape_color, edge_vector_color)
        fig.savefig(os.path.join(self.summary_plots_folder, 'Escape categories - ' + self.labels[c] + '.tif'), format='tif', bbox_inches = 'tight', pad_inches = 0)
        fig.savefig(os.path.join(self.summary_plots_folder, 'Escape categories - ' + self.labels[c] + '.eps'), format='eps', bbox_inches = 'tight', pad_inches = 0)
    # plt.show()

# strategies = np.array([4,5,0])
# fig = plot_strategies(strategies, homing_vector_color, non_escape_color, edge_vector_color)
# plt.show()
# fig.savefig(os.path.join(self.summary_plots_folder, 'Trajectory by previous edge-vectors 2.tif'), format='tif', bbox_inches='tight', pad_inches=0)
# fig.savefig(os.path.join(self.summary_plots_folder, 'Trajectory by previous edge-vectors 2.eps'), format='eps', bbox_inches='tight', pad_inches=0)

def plot_edginess(self):
    # initialize parameters
    fps = 30
    escape_duration = 9 #6
    HV_cutoff = .75 #.7
    ETD = 8
    traj_loc = 40

    edge_vector_color = np.array([0, .7, .99])**5
    homing_vector_color = np.array([.95, 0, .85])**5

    edge_vector_color = np.array([.98, .6, .6])**5
    homing_vector_color = np.array([.9, .9, .9])**5

    # initialize figures
    fig, fig2, fig3, ax, ax2, ax3 = initialize_figures(self)
    # loop over experiments and conditions
    for c, (experiment, condition) in enumerate(zip(self.experiments, self.conditions)):
        # extract experiments from nested list
        sub_experiments, sub_conditions = extract_experiments(experiment, condition)
        # get the number of trials
        number_of_trials = get_number_of_trials(sub_experiments, sub_conditions, self.analysis)
        number_of_mice = get_number_of_mice(sub_experiments, sub_conditions, self.analysis)
        # initialize array to fill in with each trial's data
        edginess, end_idx, time_since_down, time_to_shelter, time_to_shelter_all, prev_edginess, scaling_factor, time_in_center, trial_num, _, _ = \
            initialize_variable_edginess(number_of_trials, self, sub_experiments)
        # loop over each experiment and condition
        for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
            # loop over each mouse
            for i, mouse in enumerate(self.analysis[experiment][condition]['start time']):
                if self.analysis_options['control'] and not mouse=='control': continue
                if not self.analysis_options['control'] and mouse=='control': continue
                # loop over each trial
                prev_homings = []
                t = 0
                for trial in range(len(self.analysis[experiment][condition]['end time'][mouse])):
                    trial_num += 1
                    # impose conditions
                    if 'food' in experiment:
                        if t > 8: continue
                        if self.analysis[experiment][condition]['start time'][mouse][trial] < 20: continue
                    else:
                        if trial > 2: continue
                        if condition == 'obstacle' and trial < 2: continue
                        if condition == 'obstacle': continue


                    end_idx[trial_num] = self.analysis[experiment][condition]['end time'][mouse][trial]
                    if np.isnan(end_idx[trial_num]): continue
                    if (end_idx[trial_num] > escape_duration * fps): continue
                    elif trial and 'quick' in experiment: continue
                    # skip certain trials
                    y_start = self.analysis[experiment][condition]['path'][mouse][trial][1][0] * scaling_factor
                    x_start = self.analysis[experiment][condition]['path'][mouse][trial][0][0] * scaling_factor
                    if y_start > 23: continue #23
                    if abs(x_start-50) > 25: continue #23
                    # if y_start > 25: continue
                    # if abs(x_start-50) > 25: continue #22
                    # add data
                    edginess[trial_num] = self.analysis[experiment][condition]['edginess'][mouse][trial]
                    # get previous edginess #TEMPORARY COMMENT
                    # time_to_shelter, SR = get_prev_edginess(ETD, condition, experiment, mouse, prev_edginess, scaling_factor, self, traj_loc, trial, trial_num)
                    # get time in center
                    # time_in_center[trial_num] = self.analysis[experiment][condition]['in center'][mouse][trial] / 30
                    # get time since obstacle removal?
                    # time_since_down[trial_num] = self.analysis[experiment][condition]['start time'][mouse][trial] - self.analysis[experiment]['probe']['start time'][mouse][0]
                    t += 1
                # get prev homings
                time_to_shelter_all.append(time_to_shelter)


        '''     plot edginess by condition     '''
        # get the data
        data = abs(edginess)
        plot_data = data[~np.isnan(data)]
        # plot each trial
        scatter_axis = scatter_the_axis(c, plot_data)
        ax.scatter(scatter_axis[plot_data>HV_cutoff], plot_data[plot_data>HV_cutoff], color=edge_vector_color[::-1], s=30, zorder = 99)
        ax.scatter(scatter_axis[plot_data<HV_cutoff], plot_data[plot_data<HV_cutoff], color=homing_vector_color[::-1], s=30, zorder = 99)
        #do kde
        try:
            kde = fit_kde(plot_data, bw=.04)  # .04
            plot_kde(ax, kde, plot_data, z=3*c + .3, vertical=True, normto=1.3, color=[.5,.5,.5], violin=False, clip=True)  # True)
        except: pass

        # save the plot
        fig.savefig(os.path.join(self.summary_plots_folder,'Edginess comparison.tif'), format='tif', bbox_inches = 'tight', pad_inches = 0)
        fig.savefig(os.path.join(self.summary_plots_folder,'Edginess comparison.eps'), format='eps', bbox_inches = 'tight', pad_inches = 0)

        '''     plot the correlation    '''
        # do both prev homings and time in center # np.array(time_since_down) # 'Time since removal'
        # for plot_data_corr, fig_corr, ax_corr, data_label in zip([prev_edginess, time_in_center], [fig2, fig3], [ax2, ax3], ['Prior homings','Exploration']): #
        #     plot_data_corr = plot_data_corr[~np.isnan(data)]
        #     # plot data
        #     ax_corr.scatter(plot_data_corr, plot_data, color=[.5, .5, .5], s=30, alpha=1, edgecolors=[.2, .2, .2], linewidth=1)
        #     # do correlation
        #     r, p = scipy.stats.pearsonr(plot_data_corr, plot_data)
        #     print(r, p)
        #     # do linear regression
        #     plot_data_corr, prediction = do_linear_regression(plot_data, plot_data_corr)
        #     # plot linear regresssion
        #     ax_corr.plot(plot_data_corr, prediction['Pred'].values, color=[.0, .0, .0], linewidth=1, linestyle='--', alpha=.7)
        #     ax_corr.fill_between(plot_data_corr, prediction['lower'].values, prediction['upper'].values, color=[.2, .2, .2], alpha=.05)
        #     fig_corr.savefig(os.path.join(self.summary_plots_folder, 'Edginess by ' + data_label + ' - ' + self.labels[c] + '.tif'), format='tif')
        #     fig_corr.savefig(os.path.join(self.summary_plots_folder, 'Edginess by ' + data_label + ' - ' + self.labels[c] + '.eps'), format='eps')
    plt.show()
    time_to_shelter_all = np.concatenate(list(flatten(time_to_shelter_all))).astype(float)
    np.percentile(time_to_shelter_all, 25)
    np.percentile(time_to_shelter_all, 75)



def plot_prediction(self):
    # initialize parameters
    fps = 30
    escape_duration = 6
    ETD = 4
    traj_loc = 40


    # initialize figures
    fig1, ax1, fig2, ax2 = initialize_figures_prediction(self)
    # loop over experiments and conditions
    for c, (experiment, condition) in enumerate(zip(self.experiments, self.conditions)):
        # extract experiments from nested list
        sub_experiments, sub_conditions = extract_experiments(experiment, condition)
        # get the number of trials
        number_of_trials = get_number_of_trials(sub_experiments, sub_conditions, self.analysis)
        number_of_mice = get_number_of_mice(sub_experiments, sub_conditions, self.analysis)
        # initialize array to fill in with each trial's data
        edginess, end_idx, angle_turned, _, _, prev_edginess, scaling_factor, _, trial_num, prev_movement_and_ICs, data_y_for_prev_movement = \
            initialize_variable_edginess(number_of_trials, self, sub_experiments)
        # loop over each experiment and condition
        for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
            # loop over each mouse
            for i, mouse in enumerate(self.analysis[experiment][condition]['start time']):
                if self.analysis_options['control'] and not mouse=='control': continue
                if not self.analysis_options['control'] and mouse=='control': continue
                # loop over each trial
                prev_homings = []
                for trial in range(len(self.analysis[experiment][condition]['end time'][mouse])):
                    trial_num += 1
                    # impose conditions
                    if trial > 2: continue
                    end_idx[trial_num] = self.analysis[experiment][condition]['end time'][mouse][trial]
                    if np.isnan(end_idx[trial_num]): continue
                    if (end_idx[trial_num] > escape_duration * fps): continue
                    elif trial and 'quick' in experiment: continue
                    # skip certain trials
                    y_start = self.analysis[experiment][condition]['path'][mouse][trial][1][0] * scaling_factor
                    x_start = self.analysis[experiment][condition]['path'][mouse][trial][0][0] * scaling_factor
                    if y_start > 23: continue
                    if abs(x_start-50) > 23: continue #22
                    # add data
                    edginess[trial_num] = abs(self.analysis[experiment][condition]['edginess'][mouse][trial])
                    # get previous edginess
                    time_to_shelter, SR = get_prev_edginess(ETD, condition, experiment, mouse, prev_edginess, scaling_factor, self, traj_loc, trial, trial_num)
                    # get the angle turned during the escape
                    angle_turned[trial_num] = self.analysis[experiment][condition]['movement'][mouse][trial][2]
                    # get the angle turned, delta x, delta y, and delta phi of previous homings
                    # x_start, y_start, angle_start, turn_angle
                    bout_start_angle = self.analysis[experiment][condition]['movement'][mouse][trial][1]
                    bout_start_position  = self.analysis[experiment][condition]['movement'][mouse][trial][0]

                    # pos control: insert informative points
                    #     self.analysis[experiment][condition]['prev movements'][mouse][trial][0][k] = bout_start_position[0]
                    #     self.analysis[experiment][condition]['prev movements'][mouse][trial][1][k] = bout_start_position[1]
                    #     self.analysis[experiment][condition]['prev movements'][mouse][trial][2][k] = bout_start_angle
                    #     self.analysis[experiment][condition]['prev movements'][mouse][trial][3][k] = angle_turned[trial_num]

                    prev_movement_and_ICs[0].append([x - bout_start_position[0] for x in self.analysis[experiment][condition]['prev movements'][mouse][trial][0]])
                    prev_movement_and_ICs[1].append([y - bout_start_position[1] for y in self.analysis[experiment][condition]['prev movements'][mouse][trial][1]])
                    prev_movement_and_ICs[2].append([abs(sa - bout_start_angle) for sa in self.analysis[experiment][condition]['prev movements'][mouse][trial][2]])
                    prev_movement_and_ICs[3].append(self.analysis[experiment][condition]['prev movements'][mouse][trial][3])

                    data_y_for_prev_movement[0].append(list(np.ones(len(self.analysis[experiment][condition]['prev movements'][mouse][trial][0])) * edginess[trial_num]))
                    data_y_for_prev_movement[1].append(list(np.ones(len(self.analysis[experiment][condition]['prev movements'][mouse][trial][0])) * angle_turned[trial_num]))

        # format the prev movement and ICs data
        delta_x = list(flatten(prev_movement_and_ICs[0]))
        delta_y = list(flatten(prev_movement_and_ICs[1]))
        delta_angle = list(flatten(prev_movement_and_ICs[2]))
        turn_angle = list(flatten(prev_movement_and_ICs[3]))

        edginess_for_prev_movement = list(flatten(data_y_for_prev_movement[0]))
        turn_angle_for_prev_movement = list(flatten(data_y_for_prev_movement[1]))

        prev_movements_and_ICs_array = np.ones((len(delta_x), 4)) * 0 #np.nan
        prev_movements_and_ICs_array[:, 0] = scipy.stats.zscore(delta_x) * turn_angle
        prev_movements_and_ICs_array[:, 1] = scipy.stats.zscore(delta_y) * turn_angle
        prev_movements_and_ICs_array[:, 2] = scipy.stats.zscore(delta_angle) * turn_angle
        prev_movements_and_ICs_array[:, 3] = turn_angle

        # get the data
        predict_data_y_all = [[edginess[~np.isnan(edginess)].reshape(-1, 1), # for the mean edginess input data
                              angle_turned[~np.isnan(edginess)].reshape(-1, 1)], # for the mean edginess input data
                              [edginess_for_prev_movement, turn_angle_for_prev_movement]] # for the movements input data
        data_y_labels = ['trajectory', 'angle']

        predict_data_x_all = [prev_edginess[~np.isnan(edginess)].reshape(-1, 1),  # mean prev edginess
                              prev_movements_and_ICs_array] # all prev homing movements
                             # np.zeros_like(prev_edginess[~np.isnan(edginess)]).reshape(-1, 1),  # zeros as negative control

        # edgy input colors
        input_colors = [[0, .6, .4], [.6, 0, .4]] #[.5, .5, .5],
        # split the data for cross val
        trials = 1000
        # loop acros angle prediction and traj prediction
        for h, (fig, ax) in enumerate(zip([fig1, fig2],[ax1, ax2])):
            # loop across traj inputs and movement inputs
            for i, predict_data_x in enumerate(predict_data_x_all):
                # get prediction data
                predict_data_y = predict_data_y_all[i][h]
                # get color
                color = input_colors[i]
                # initialize prediction arrays
                prediction_scores = np.zeros(trials)
                for j in range(trials):
                    X_train, X_test, y_train, y_test = train_test_split(predict_data_x, \
                                                predict_data_y, test_size=0.5, random_state=j)
                    # create the model
                    # LR = linear_model.LinearRegression()
                    LR = linear_model.Ridge(alpha = .5)
                    # train the model
                    LR.fit(X_train, y_train)
                    # get the score
                    prediction_scores[j] = LR.score(X_test, y_test)
                # exclude super negative ones
                prediction_scores = prediction_scores[prediction_scores > np.percentile(prediction_scores, 10)]
                # plot the scores
                # ax.scatter(prediction_scores, np.zeros_like(prediction_scores), color=color, s=20, alpha = .1)
                #do kde
                kde = fit_kde(prediction_scores, bw=.03)  # .04
                plot_kde(ax, kde, prediction_scores, z = 0, vertical=False, color=color, violin=False, clip=False)  # True)
            fig.savefig(os.path.join(self.summary_plots_folder,'Prediction of ' + data_y_labels[h] + ' - ' + self.labels[c] + '.tif'), format='tif')
            fig.savefig(os.path.join(self.summary_plots_folder,'Precition of ' + data_y_labels[h] + ' - ' + self.labels[c] + '.eps'), format='eps')
    plt.show()


def plot_efficiency(self):
    # initialize parameters
    fps = 30
    escape_duration = 6
    HV_cutoff = .75
    ETD = np.inf
    ax2, fig2, ax3, fig3 = initialize_figures_efficiency(self)
    efficiency_data = [[],[]]
    duration_data = [[],[]]
    # loop over experiments and conditions
    for c, (experiment, condition) in enumerate(zip(self.experiments, self.conditions)):
        # extract experiments from nested list
        sub_experiments, sub_conditions = extract_experiments(experiment, condition)
        # get the number of trials
        number_of_trials = get_number_of_trials(sub_experiments, sub_conditions, self.analysis)
        number_of_mice = get_number_of_mice(sub_experiments, sub_conditions, self.analysis)
        # initialize array to fill in with each trial's data
        efficiency, efficiency_RT, end_idx, num_prev_homings, scaling_factor, time, trial_num, trials, edginess = \
            initialize_variables_efficiency(number_of_trials, self, sub_experiments)
        # loop over each experiment and condition
        for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
            # loop over each mouse
            for i, mouse in enumerate(self.analysis[experiment][condition]['full path length']):
                # control analysis
                if self.analysis_options['control'] and not mouse=='control': continue
                if not self.analysis_options['control'] and mouse=='control': continue
                # loop over each trial
                for trial in range(len(self.analysis[experiment][condition]['end time'][mouse])):
                    trial_num += 1
                    if trial > 2: continue
                    # impose coniditions
                    end_idx[trial_num] = self.analysis[experiment][condition]['end time'][mouse][trial]
                    if (end_idx[trial_num] > escape_duration * fps): continue
                    # skip certain trials
                    y_start = self.analysis[experiment][condition]['path'][mouse][trial][1][0] * scaling_factor
                    x_start = self.analysis[experiment][condition]['path'][mouse][trial][0][0] * scaling_factor
                    # if y_start > 23: continue
                    if abs(x_start-50) > 23: continue #22
                    # add data
                    fill_in_trial_data_efficiency(ETD, condition, efficiency, efficiency_RT, experiment, mouse, num_prev_homings, self, time, trial, trial_num, trials, edginess)

                    # normalize end idx to
                    end_idx[trial_num] = end_idx[trial_num] / self.analysis[experiment][condition]['optimal path length'][mouse][trial] * 500
        # format end ind
        end_idx = np.array([e/30 for e in end_idx])
        end_idx[np.isnan(efficiency)] = np.nan
        # loop over data to plot
        # for i, (data, data_label) in enumerate(zip([efficiency, end_idx, edginess], ['Efficiency', 'Duration', 'Trajectory'])):
        #     for x_data, x_data_label in zip([num_prev_homings, trials+1, time], ['Prior homings', 'Trials','Time']):
        for i, (data, data_label) in enumerate(zip([edginess, efficiency, end_idx], ['Trajectory', 'Efficiency', 'Duration'])):
            for x_data, x_data_label in zip([num_prev_homings], ['Prior homings']):
                # initialize figure
                fig1, ax1 = plt.subplots(figsize=(9, 9))
                # set up the figure
                if data_label=='Efficiency': ax1.set_ylim([-.03, 1.03])
                elif data_label=='Duration': ax1.set_ylim([-.1, 6.5])
                # only plot escapes
                data_for_box_plot = data[~np.isnan(data)]
                x_data = x_data[~np.isnan(data)]
                # format for time and trial differently
                if np.max(x_data) < 5:
                    plot_data_x, plot_data = x_data, data_for_box_plot
                    ax1.set_xticks(np.unique(x_data).astype(int))
                else:
                    plot_data_x, plot_data = x_data[x_data < 60], data_for_box_plot[x_data < 60]
                    ax1.set_xticks(np.arange(5,25,5))
                    # ax1.set_xlim([5,20])
                # get the correlation
                r, p = scipy.stats.pearsonr(plot_data_x, plot_data)
                print(r, p)
                # jitter the axis
                scatter_axis = scatter_the_axis_efficiency(plot_data, plot_data_x)
                # plot each trial
                ax1.scatter(scatter_axis, plot_data, color=[.5, .5, .5], s=30, alpha=1, edgecolor=[0,0,0], linewidth=1)
                ax1.scatter(scatter_axis[plot_data > HV_cutoff], plot_data[plot_data > HV_cutoff], color=[.5, .5, .5], s=50, alpha=1, edgecolor=[0, 0, 0], linewidth=1)
                # do a linear regression
                plot_data_x, prediction = do_linear_regression(plot_data, plot_data_x)
                # plot the linear regression
                ax1.plot(plot_data_x, prediction['Pred'].values, color=[.0, .0, .0], linewidth=1, linestyle='--', alpha=.7)
                ax1.fill_between(plot_data_x, prediction['lower'].values, prediction['upper'].values, color=[.2,.2,.2], alpha=.05)  # 6
                # save the plot
                plt.savefig(os.path.join(self.summary_plots_folder, data_label + ' by ' + x_data_label + ' - ' + self.labels[c] + '.tif'), format='tif')
                plt.savefig(os.path.join(self.summary_plots_folder, data_label + ' by ' + x_data_label + ' - ' + self.labels[c] + '.eps'), format='eps')
                # plot the boxplot
                if data_label == 'Efficiency':
                    ax, fig = ax2, fig2
                    efficiency_data[c] = plot_data
                elif data_label == 'Duration':
                    ax, fig = ax3, fig3
                    duration_data[c] = plot_data
                else: continue
                scatter_axis = scatter_the_axis_efficiency(plot_data, np.ones_like(plot_data)*c)
                ax.scatter(scatter_axis, plot_data, color=[.25, .25, .25], s=40, alpha=1, edgecolor=[0, 0, 0], linewidth=1)
                # plot kde
                kde = fit_kde(plot_data, bw=.04) #.2)  # .04
                plot_kde(ax, kde, plot_data, z=c + .1, vertical=True, normto=.3, color=[.5, .5, .5], violin=False, clip=True)  # True)
                # plot errorbar
                median = np.percentile(plot_data, 50)
                third_quartile = np.percentile(plot_data, 75)
                first_quartile = np.percentile(plot_data, 25)
                ax.errorbar(c - .2, median, yerr=np.array([[median - first_quartile], [third_quartile - median]]), color=[0,0,0], capsize=10, capthick=3, alpha=1, linewidth=3)
                ax.scatter(c - .2, median, color=[0,0,0], s=175, alpha=1)
                # save the plot
                fig.savefig(os.path.join(self.summary_plots_folder, data_label + ' comparison - ' + self.labels[c] + '.tif'), format='tif')
                fig.savefig(os.path.join(self.summary_plots_folder, data_label + ' comparison - ' + self.labels[c] + '.eps'), format='eps')


    # do t test
    t, p = scipy.stats.ttest_ind(efficiency_data[0], efficiency_data[1], equal_var=False)
    print('Efficiency: ' + str(p))
    print(np.mean(efficiency_data[0]))
    print(np.mean(efficiency_data[1]))

    t, p = scipy.stats.ttest_ind(duration_data[0], duration_data[1], equal_var=False)
    print('Duration: ' + str(p))
    print(np.mean(duration_data[0]))
    print(np.mean(duration_data[1]))

    plt.show()




def plot_exploration(self):
    '''     plot the average exploration heat map       '''
    # loop over experiments and conditions
    for c, (experiment, condition) in enumerate(zip(self.experiments, self.conditions)):
        # extract experiments from nested list
        sub_experiments, sub_conditions = extract_experiments(experiment, condition)
        # get the number of trials
        number_of_mice = get_number_of_mice(sub_experiments, sub_conditions, self.analysis)
        mouse_num = 0
        # initialize array to fill in with each mouse's data
        shape = self.analysis[sub_experiments[0]]['obstacle']['shape']
        exploration = np.zeros((shape[0], shape[1], number_of_mice))
        # loop over each experiment and condition
        for e, (experiment, condition) in enumerate(zip(sub_experiments, sub_conditions)):
            # loop over each mouse
            for mouse in self.analysis[experiment][condition]['exploration']:
                # fill array with each mouse's data
                exploration[:, :, mouse_num] = self.analysis[experiment][condition]['exploration'][mouse]
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
        scipy.misc.imsave(os.path.join(self.summary_plots_folder, experiment + '_exploration_' + condition + '.tif'), exploration_image[:,:,::-1])

        shape = self.analysis[sub_experiments[0]]['obstacle']['shape']
        obstacle_type = self.analysis[sub_experiments[0]]['obstacle']['type']
        _, _, shelter_roi = model_arena(shape, False, False, obstacle_type, simulate=False, dark=self.dark_theme)
        percent_in_shelter = []
        for m in range( exploration.shape[2]):
            mouse_exploration = exploration[:,:,m]
            percent_in_shelter.append( np.sum(mouse_exploration*shelter_roi) )



