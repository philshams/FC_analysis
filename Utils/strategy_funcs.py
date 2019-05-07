import numpy as np
import cv2
import scipy
import copy
import os
import itertools
from tqdm import tqdm
import scipy.signal
import scipy.ndimage
from Utils.registration_funcs import model_arena
import time


'''
FIRST, SOME AUXILIARY FUNCTIONS TO HELP WITH STRATEGY PROCESSING
'''

def filter_seq(n, sign):

    filter_sequence = np.ones(n) * sign

    return filter_sequence


def convolve(data, n, sign, time = 'current', time_chase = 20):

    if time == 'past':
        convolved_data = np.concatenate((np.zeros(n - 1), np.convolve(data, filter_seq(n, sign), mode='valid'))) / n
    elif time == 'future':
        convolved_data = np.concatenate((np.convolve(data, filter_seq(n, sign), mode='valid'), np.zeros(n - 1))) / n
    elif time == 'far future':
        convolved_data = np.concatenate((np.convolve(data, filter_seq(n, sign), mode='valid'), np.zeros(n - 1 + time_chase))) / n
        convolved_data = convolved_data[time_chase:]
    else:
        convolved_data = np.concatenate((np.zeros(int(n/2 - 1)), np.convolve(data, filter_seq(n, sign), mode='valid'), np.zeros(int(n/2)))) / n

    return convolved_data


def initialize_arrays(exploration_arena_copy, stim_frame, previous_stim_frame):

    exploration_arena = copy.deepcopy(exploration_arena_copy)
    save_exploration_arena = exploration_arena.copy()
    model_mouse_mask_initial = (exploration_arena[:, :, 0] * 0).astype(bool)
    thresholds_passed = np.zeros(stim_frame - previous_stim_frame)
    stimulus_started = False

    return exploration_arena, save_exploration_arena, model_mouse_mask_initial, thresholds_passed, stimulus_started


def create_local_variables(coordinates, stim_frame, previous_stim_frame, skip_frames):

    goal_speeds = coordinates['speed_toward_subgoal'][previous_stim_frame + skip_frames:stim_frame + skip_frames]
    absolute_speeds = coordinates['speed'][previous_stim_frame + skip_frames:stim_frame + skip_frames]
    subgoal_angles = coordinates['subgoal_angle'][previous_stim_frame + skip_frames:stim_frame + skip_frames]
    body_angles = coordinates['body_angle'][previous_stim_frame + skip_frames:stim_frame + skip_frames]
    angular_speed = coordinates['angular_speed_shelter'][previous_stim_frame + skip_frames:stim_frame + skip_frames]
    distance_from_shelter = coordinates['distance_from_shelter'][previous_stim_frame + skip_frames:stim_frame + skip_frames]
    distance_from_subgoal = coordinates['distance_from_shelter'][previous_stim_frame + skip_frames:stim_frame + skip_frames]
    within_subgoal_bound = coordinates['in_subgoal_bound'][previous_stim_frame + skip_frames:stim_frame + skip_frames]

    x_location_butt = coordinates['butty_location'][0][previous_stim_frame + skip_frames:stim_frame + skip_frames].astype(np.uint16)
    y_location_butt = coordinates['butty_location'][1][previous_stim_frame + skip_frames:stim_frame + skip_frames].astype(np.uint16)

    x_location_face = coordinates['front_location'][0][previous_stim_frame + skip_frames:stim_frame + skip_frames].astype(np.uint16)
    y_location_face = coordinates['front_location'][1][previous_stim_frame + skip_frames:stim_frame + skip_frames].astype(np.uint16)

    return goal_speeds, absolute_speeds, body_angles, subgoal_angles, angular_speed, distance_from_shelter, distance_from_subgoal, within_subgoal_bound, x_location_butt, y_location_butt, x_location_face, y_location_face

def threshold(data, limit, type = '>'):

    if type == '>':
        passed_threshold = (data > limit).astype(np.uint16)
    elif type == '<':
        passed_threshold = (data < limit).astype(np.uint16)

    return passed_threshold

def get_homing_groups(thresholds_passed, minimum_distance, minimum_duration, distance_from_subgoal, distance_from_shelter, minimum_shelter_distance, speeds):

    groups = []
    idx = 0
    group_idx = np.zeros(len(thresholds_passed))

    for k, g in itertools.groupby(thresholds_passed):
        groups.append(list(g))
        group_length = len(groups[len(groups) - 1]);
        idx += group_length
        distance_traveled = (distance_from_subgoal[idx - 1] - distance_from_subgoal[idx - group_length]) / distance_from_subgoal[idx - group_length]
        if k and ((group_length < minimum_duration) or (distance_traveled > minimum_distance) or (distance_from_shelter[idx - group_length] < minimum_shelter_distance)):
        # if k and ((distance_from_shelter[idx - group_length] < minimum_shelter_distance)):
            thresholds_passed[idx - group_length: idx] = False
        elif k:
            group_idx[idx - group_length] = group_length
            group_idx[idx - 1] = 2000*(np.mean(speeds[idx - group_length:idx]) / group_length**2)
            # print(distance_traveled)

    return thresholds_passed, group_idx


def multi_phase_phinder(thresholds_passed, minimum_distance, max_shelter_proximity, body_angles, distance_from_shelter,
                            x_location_butt, y_location_butt, x_location_face, y_location_face, critical_turn, first_frame):

    groups = []; idx = 0
    vectors  = [];
    group_idx = np.zeros(len(thresholds_passed))
    end_idx = np.zeros(len(thresholds_passed))
    distance_from_start = np.zeros(len(thresholds_passed))

    for k, g in itertools.groupby(thresholds_passed):
        groups.append(list(g))
        group_length = len(groups[len(groups)-1]);
        idx += group_length
        # for each bout, get the relevant vectors
        if k:
            # get later phases of the escape
            start_index = idx - group_length

            # get the distance travelled so far during the bout
            distance_from_start[idx - group_length: idx] = np.sqrt((x_location_butt[idx - group_length: idx] - x_location_butt[idx - group_length]) ** 2 + \
                                                                   (y_location_butt[idx - group_length: idx] - y_location_butt[idx - group_length]) ** 2)

            while True:
                # get the cumulative distance traveled
                distance_traveled = np.sqrt((x_location_butt[start_index:idx] - x_location_butt[start_index]) ** 2
                                            + (y_location_butt[start_index:idx] - y_location_butt[start_index]) ** 2)

                # has the minimum distance been traveled
                traveled_far_enough = np.where(distance_traveled > minimum_distance)[0]
                if not traveled_far_enough.size: break

                # now check for more phases: find the cumulative turn angle since the start of the bout
                angle_turned =(body_angles[start_index:idx] - body_angles[start_index + traveled_far_enough[0]])
                angle_turned[angle_turned > 180] = 360 - angle_turned[angle_turned > 180]
                angle_turned[angle_turned < -180] = 360 + angle_turned[angle_turned < -180]

                # not including the beginning
                angle_turned[:traveled_far_enough[0]] = 0

                # get the indices of critically large turns
                critically_turned = np.where(abs(angle_turned) > critical_turn)[0]

                if critically_turned.size:
                    # break if getting too close to shelter
                    if (distance_from_shelter[start_index + critically_turned[0]] < max_shelter_proximity):
                        group_idx[start_index] = idx + first_frame
                        end_idx[idx] = start_index + first_frame
                        break
                    else:
                        group_idx[start_index] = start_index + critically_turned[0] - 1 + first_frame
                        end_idx[start_index + critically_turned[0] - 1] = start_index + first_frame
                        start_index += critically_turned[0]
                else:
                    group_idx[start_index] = idx - 1 + first_frame
                    end_idx[idx - 1] = start_index + first_frame
                    break

    return group_idx, distance_from_start, end_idx


def trial_start(thresholds_passed, minimum_distance, x_location_butt, y_location_butt):

    groups = []; idx = 0
    vectors  = []; group_idx = np.zeros(len(thresholds_passed))
    trial_start_array = np.zeros(len(thresholds_passed))

    for k, g in itertools.groupby(thresholds_passed):
        groups.append(list(g))
        group_length = len(groups[len(groups)-1]);
        idx += group_length
        if k:
            distance_traveled = np.sqrt((x_location_butt[idx - group_length:idx] - x_location_butt[idx - group_length]) ** 2
                                        + (y_location_butt[idx - group_length:idx] - y_location_butt[idx - group_length]) ** 2)
            far_enough_way = np.where( distance_traveled > minimum_distance)[0]
            if far_enough_way.size:
                trial_start_array[idx - group_length: idx - group_length + far_enough_way[0]] = True

    return trial_start_array


def get_color_settings(frame_num, stim_frame, speed_setting, speed_colors, multipliers, exploration_arena, model_mouse_mask, group_counter, blue_duration):

    # determine color
    if frame_num >= stim_frame - 60:
        speed_color = np.array([200, 200, 200])
    else:
        speed_color = speed_colors[speed_setting] + (group_counter < blue_duration )*np.array([230,-4,30]) #[20, 254, 140]
        # speed_color = speed_colors[speed_setting] + (group_counter < 70) * np.array([230, -4, 30])

    # get darkness
    multiplier = multipliers[speed_setting]

    # make this trial's stimulus response more prominent in the saved version
    if frame_num >= stim_frame:
        save_multiplier = multiplier / 5
    else:
        save_multiplier = multiplier

    # create color multiplier to modify image
    color_multiplier = 1 - (1 - speed_color / [255, 255, 255]) / (np.mean(1 - speed_color / [255, 255, 255]) * multiplier)
    save_color_multiplier = 1 - (1 - speed_color / [255, 255, 255]) / (np.mean(1 - speed_color / [255, 255, 255]) * save_multiplier)

    return color_multiplier, save_color_multiplier


def stimulus_onset(stimulus_started, model_mouse_mask, exploration_arena):

    _, contours, _ = cv2.findContours(model_mouse_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    save_exploration_arena = exploration_arena.copy()
    stimulus_started = True
    first_stim_frame = True

    return save_exploration_arena, stimulus_started, contours

def draw_mouse(model_mouse_mask, model_mouse_mask_total, coordinates, frame_num, back_butt_dist, group_length, stim_frame, stimulus_started, group_counter, time_chase):

    # reset mask at start of bout
    model_mouse_mask = model_mouse_mask * (1 - (group_length > 0)) * (1 - (frame_num >= stim_frame and not stimulus_started))

    # extract DLC coordinates from the saved coordinates dictionary
    body_angle = coordinates['body_angle'][frame_num]
    shoulder_angle = coordinates['shoulder_angle'][frame_num]
    shoulder_location = tuple(coordinates['shoulder_location'][:, frame_num].astype(np.uint16))
    body_location = tuple(coordinates['center_body_location'][:, frame_num].astype(np.uint16))

    # draw ellipses representing model mouse
    model_mouse_mask = cv2.ellipse(model_mouse_mask, body_location, (int(back_butt_dist), int(back_butt_dist * .4)), 180 - body_angle, 0, 360, 100, thickness=-1)
    model_mouse_mask = cv2.ellipse(model_mouse_mask, shoulder_location, (int(back_butt_dist * .8), int(back_butt_dist * .26)), 180 - shoulder_angle, 0, 360, 100, thickness=-1)

    # stop shading after n frames
    if group_counter >= time_chase:
        # extract DLC coordinates from the saved coordinates dictionary
        old_body_angle = coordinates['body_angle'][frame_num - time_chase]
        old_shoulder_angle = coordinates['shoulder_angle'][frame_num - time_chase]
        old_shoulder_location = tuple(coordinates['shoulder_location'][:, frame_num - time_chase].astype(np.uint16))
        old_body_location = tuple(coordinates['center_body_location'][:, frame_num - time_chase].astype(np.uint16))

        # erase ellipses representing model mouse
        model_mouse_mask = cv2.ellipse(model_mouse_mask, old_body_location, (int(back_butt_dist), int(back_butt_dist * .4)), 180 - old_body_angle, 0, 360, 0,thickness=-1)
        model_mouse_mask = cv2.ellipse(model_mouse_mask, old_shoulder_location, (int(back_butt_dist * .8), int(back_butt_dist * .26)), 180 - old_shoulder_angle, 0, 360, 0, thickness=-1)

        # draw ellipses representing model mouse
        model_mouse_mask_total = cv2.ellipse(model_mouse_mask_total, body_location, (int(back_butt_dist), int(back_butt_dist * .33)), 180 - body_angle, 0, 360,100, thickness=-1)
        model_mouse_mask_total = cv2.ellipse(model_mouse_mask_total, shoulder_location, (int(back_butt_dist * .8), int(back_butt_dist * .23)),180 - shoulder_angle, 0, 360, 100, thickness=-1)
    else:
        model_mouse_mask_total = model_mouse_mask.copy()

    # return model_mouse_mask.astype(bool), model_mouse_mask_initial, model_mouse_mask_total
    return model_mouse_mask.astype(bool), model_mouse_mask_total

def draw_silhouette(model_mouse_mask_initial, coordinates, frame_num, back_butt_dist):

    # extract DLC coordinates from the saved coordinates dictionary]
    body_angle = coordinates['body_angle'][frame_num - 1]
    shoulder_angle = coordinates['shoulder_angle'][frame_num - 1]
    head_angle = coordinates['head_angle'][frame_num - 1]
    neck_angle = coordinates['neck_angle'][frame_num - 1]
    nack_angle = coordinates['nack_angle'][frame_num - 1]
    head_location = tuple(coordinates['head_location'][:, frame_num - 1].astype(np.uint16))
    nack_location = tuple(coordinates['nack_location'][:, frame_num - 1].astype(np.uint16))
    front_location = tuple(coordinates['front_location'][:, frame_num - 1].astype(np.uint16))
    shoulder_location = tuple(coordinates['shoulder_location'][:, frame_num - 1].astype(np.uint16))
    body_location = tuple(coordinates['center_body_location'][:, frame_num - 1].astype(np.uint16))

    # when turning, adjust relative sizes
    if abs(body_angle - shoulder_angle) > 20:
        shoulder = False
    else:
        shoulder = True

    # draw ellipses representing model mouse
    model_mouse_mask = cv2.ellipse(model_mouse_mask_initial.copy(), head_location, (int(back_butt_dist * .6), int(back_butt_dist * .3)), 180 - head_angle, 0, 360, 100, thickness=-1)
    model_mouse_mask = cv2.ellipse(model_mouse_mask, front_location, (int(back_butt_dist * .5), int(back_butt_dist * .33)), 180 - neck_angle, 0, 360, 100, thickness=-1)
    model_mouse_mask = cv2.ellipse(model_mouse_mask, body_location, (int(back_butt_dist * .9), int(back_butt_dist * .5)), 180 - body_angle, 0, 360, 100, thickness=-1)
    model_mouse_mask = cv2.ellipse(model_mouse_mask, nack_location, (int(back_butt_dist * .7), int(back_butt_dist * .35)), 180 - nack_angle, 0, 360, 100,thickness=-1)
    if shoulder:
        model_mouse_mask = cv2.ellipse(model_mouse_mask, shoulder_location, (int(back_butt_dist), int(back_butt_dist * .44)), 180 - shoulder_angle, 0, 360, 100, thickness=-1)

    return model_mouse_mask.astype(bool)

def dilute_shading(exploration_arena, save_exploration_arena, prior_exploration_arena, model_mouse_mask_initial, stimulus_started, end_index, speed_setting):

    speed = end_index #+ .3 * speed_setting
    if speed > .6: speed = .6
    if speed < .3: speed = .3
    # print(speed)

    exploration_arena[model_mouse_mask_initial.astype(bool)] = \
        ((speed * exploration_arena[model_mouse_mask_initial.astype(bool)].astype(np.uint16) +
          (1 - speed) * prior_exploration_arena[model_mouse_mask_initial.astype(bool)].astype(np.uint16))).astype(np.uint8)

    if stimulus_started:
        save_exploration_arena[model_mouse_mask_initial.astype(bool)] = \
            ((speed * save_exploration_arena[model_mouse_mask_initial.astype(bool)].astype(np.uint16) +
              (1 - speed) * prior_exploration_arena[model_mouse_mask_initial.astype(bool)].astype(np.uint16))).astype(np.uint8)
    # else:
    #     save_exploration_arena = None

    return exploration_arena, save_exploration_arena


'''
NOW, THE ACTUAL STRATEGY PROCESSING FUNCTIONS
'''

def spontaneous_homings(exploration_arena_copy, session_trials_plot_background, border_size, coordinates, previous_stim_frame, stim_frame, videoname, savepath, subgoal_locations, obstacle_type):
    '''
    compute and display EXPLORATION
    go through each frame, adding the mouse silhouette
    '''

   # make video
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_clip = cv2.VideoWriter(os.path.join(savepath, videoname + ' homings.avi'), fourcc, 160,
                                  (exploration_arena_copy.shape[1], exploration_arena_copy.shape[1]), True)

    # visualize results
    cv2.namedWindow(savepath + 'homings');

    # how many frames after each stimulus to look at
    skip_frames = 300 + 150*(obstacle_type=='void')
    if previous_stim_frame == 0: previous_stim_frame -= skip_frames

    # initialize arrays
    exploration_arena, save_exploration_arena, model_mouse_mask, thresholds_passed, stimulus_started = \
        initialize_arrays(exploration_arena_copy, stim_frame, previous_stim_frame)

    # get the frame numbers to analyze
    frame_nums = np.arange(previous_stim_frame + skip_frames, stim_frame + skip_frames)

    # set the speed parameters
    speeds = [[.5, .75, 1], [.5, 1, 4]]
    speed_colors = [np.array([20, 254, 140]), np.array([20, 254, 140])]
    multipliers = [50, 50]
    smooth_duration = 30

    # set distance parameters
    close_to_shelter_distance = 40
    minimum_shelter_distance = 150
    close_to_shelter_angle = 30
    small_shelter_angle = 10
    strict_shelter_angle = 85 + (obstacle_type=='side wall')*10

    # create local variables for the current epoch
    goal_speeds, absolute_speeds, shelter_angles, subgoal_angles, angular_speed, distance_from_shelter, distance_from_subgoal, within_subgoal_bound, _, _, _, _ = \
        create_local_variables(coordinates, stim_frame, previous_stim_frame, skip_frames)

    # do convolutions to get current, future, far future, and past speeds w.r.t the shelter
    current_speed = convolve(goal_speeds, 24, -1, time='current')
    past_speed = convolve(goal_speeds, 45, -1, time='past')
    future_speed = convolve(goal_speeds, 60, -1, time='future')
    far_future_speed = convolve(goal_speeds, 60, -1, time='far future', time_chase=20)

    # is the mouse close enough to the shelter
    minimum_distance = 400 + 100 * (obstacle_type=='void')
    close_enough_to_shelter = convolve(distance_from_shelter < minimum_distance, 60, +1, time='future')
    angle_thresholded = threshold(abs(subgoal_angles), close_to_shelter_angle, '<')  # np.array(within_subgoal_bound) +
    strict_angle_thresholded = threshold(abs(subgoal_angles), strict_shelter_angle, '<') + threshold(frame_nums, stim_frame)
    distance_thresholded = threshold(distance_from_shelter, close_to_shelter_distance)

    # loop across fast and slow homings
    for speed_setting in [True, False]:

        # use higher speed setting to make faster homings darker
        low_speed, medium_speed, high_speed = speeds[speed_setting][0], speeds[speed_setting][1], speeds[speed_setting][2]

        # threshold the convolved speeds to determine which frames to draw
        current_speed_thrsh = threshold(current_speed, high_speed)
        future_speed_thrsh = threshold(future_speed, high_speed) * threshold(current_speed, medium_speed)
        far_future_speed_thrsh = threshold(far_future_speed, high_speed) * threshold(current_speed, medium_speed)
        past_speed_thrsh = threshold(past_speed, high_speed) * threshold(current_speed, medium_speed)
        stimulus_thresholded = threshold(frame_nums, stim_frame - 1 + smooth_duration) * threshold(current_speed, low_speed)

        # combine speed thresholds into one
        combined_speed_thresholds = (current_speed_thrsh + future_speed_thrsh + far_future_speed_thrsh + past_speed_thrsh)

        # combine all thresholds into one
        thresholds_passed_first_pass = distance_thresholded * (angle_thresholded * close_enough_to_shelter * combined_speed_thresholds + stimulus_thresholded)

        # and smooth it *out*
        thresholds_passed_first_pass = (strict_angle_thresholded * convolve(thresholds_passed_first_pass, smooth_duration, +1, time='current') ).astype(bool)

        # finally, add a minimum duration threshold
        minimum_distance = -.5 + .3 * speed_setting + (obstacle_type=='void')*.3 # this will be problematic for other arenas
        minimum_duration = smooth_duration + 5

        minimum_distance = (-.4 + .2 * speed_setting) + (obstacle_type=='void')*.3
        minimum_duration = smooth_duration + 5
        # print(speed_setting)
        thresholds_passed_first_pass, group_idx = get_homing_groups(thresholds_passed_first_pass * (1 - thresholds_passed),
                                                                         minimum_distance, minimum_duration, distance_from_subgoal, distance_from_shelter,
                                                                        minimum_shelter_distance, absolute_speeds)

        # set function output
        trial_groups = thresholds_passed_first_pass + thresholds_passed

        # get the index when adequately long locomotor bouts pass the threshold
        thresholds_passed = thresholds_passed_first_pass  # + (frame_nums==stim_frame)

        thresholds_passed_idx = np.where(thresholds_passed)[0]
        group_counter = 0
        bout_starting = True
        model_mouse_mask_total = model_mouse_mask.copy()

        # loop over each frame that passed the threshold
        for idx in thresholds_passed_idx:

            # get frame number
            frame_num = frame_nums[idx]
            # if (frame_num >= stim_frame - 10):
            #     continue

            # set up new bout
            if group_idx[idx] and not group_counter:
                prior_exploration_arena = exploration_arena.copy(); group_length = group_idx[idx]
                model_mouse_mask_total = model_mouse_mask.copy(); bout_starting = False
            group_counter += 1

            # draw model mouse
            back_butt_dist = 11 + (frame_num >= stim_frame) * 2
            time_chase = 80 - (frame_num >= stim_frame) * 60
            model_mouse_mask, model_mouse_mask_total = draw_mouse(model_mouse_mask, model_mouse_mask_total, coordinates, frame_num, back_butt_dist,
                                                                  group_idx[idx], stim_frame, stimulus_started, group_counter, time_chase)

            # determine color and darkness
            blue_duration = 80 - max(0, 90 - group_length)
            color_multiplier, save_color_multiplier = get_color_settings(frame_num, stim_frame, speed_setting, speed_colors, multipliers,
                                                                         exploration_arena, model_mouse_mask, group_counter, blue_duration)

            # on the stimulus onset, get the contours of the mouse body
            if (frame_num >= stim_frame or not speed_setting) and not stimulus_started:
                save_exploration_arena, stimulus_started, _ = stimulus_onset(stimulus_started, model_mouse_mask, exploration_arena)
                continue

            # apply color to arena image
            exploration_arena[model_mouse_mask.astype(bool)] = exploration_arena[model_mouse_mask.astype(bool)] * color_multiplier

            # apply color to this trial's image
            if stimulus_started:
                save_exploration_arena[model_mouse_mask.astype(bool)] = save_exploration_arena[model_mouse_mask.astype(bool)] * save_color_multiplier

            # dilute the color at the end of the bout
            if group_idx[idx] and group_counter > 1:
                exploration_arena, save_exploration_arena = dilute_shading(exploration_arena, save_exploration_arena, prior_exploration_arena,
                                                                           model_mouse_mask_total, stimulus_started, group_idx[idx], speed_setting)
                group_counter = 0

            # display image
            cv2.imshow(savepath + 'homings', exploration_arena)

            video_clip.write(exploration_arena)

            # press q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



    # apply the contours and border to the image and save the image
    cv2.imshow(savepath + 'homings', save_exploration_arena); cv2.waitKey(1)
    session_trials_plot_background[border_size:, 0:-border_size] = save_exploration_arena
    scipy.misc.imsave(os.path.join(savepath, videoname + '_spont_homings.tif'), cv2.cvtColor(session_trials_plot_background, cv2.COLOR_BGR2RGB))

    video_clip.release()
    return exploration_arena, trial_groups



def procedural_learning(exploration_arena_copy, session_trials_plot_background, border_size, coordinates, previous_stim_frame, stim_frame, videoname, savepath, subgoal_locations, trial_groups):
    '''
    compute and display EXPLORATION
    go through each frame, adding the mouse silhouette
    '''

    # visualize results
    cv2.namedWindow(savepath + ' paths')

    # initialize the arena and mouse mask
    exploration_arena = copy.deepcopy(exploration_arena_copy)
    save_exploration_arena = exploration_arena.copy()
    model_mouse_mask_initial = exploration_arena[:, :, 0] * 0
    stimulus_started = True

    # instead of speed to shelter, take max of speed to shelter and speed to subgoals
    skip_frames = 300
    if previous_stim_frame == 0: previous_stim_frame -= skip_frames

    # set what this mouse considers a high speed
    speeds = [.5, 1, 4]
    high_speed, medium_speed, low_speed = speeds[2], speeds[1], speeds[0]

    # create local variables for the current epoch
    goal_speeds, _, body_angles, subgoal_angles, angular_speed, distance_from_shelter, _, _, x_location_butt, y_location_butt, x_location_face, y_location_face = \
        create_local_variables(coordinates, stim_frame, previous_stim_frame, skip_frames)

    # do convolutions to get current, future, far future, and past speeds w.r.t the shelter
    now_speed = convolve(goal_speeds, 12, -1, time='current')  # 12?
    past_speed = convolve(goal_speeds, 30, -1, time='past')  # 12?
    present_angular_speed = convolve(abs(angular_speed), 8, +1, time='current')  # 12?

    # get the frame numbers to analyze
    frame_nums = np.arange(previous_stim_frame + skip_frames, stim_frame + skip_frames)

    # take homings from spontaneous homings function output
    thresholds_passed = trial_groups.copy()

    # get rid of segments just prior to stimulus
    thresholds_passed[-(skip_frames + 10):-skip_frames] = False

    # get vectors from the first phase of all homings
    minimum_distance = 40 # 50
    max_shelter_proximity = 150 # 200 #150
    critical_turn = 25 #was 45, and replaced subgoal_angles with body_angles
    group_idx, distance_from_start, end_idx = multi_phase_phinder(thresholds_passed, minimum_distance, max_shelter_proximity, body_angles, distance_from_shelter,
                                             x_location_butt, y_location_butt, x_location_face, y_location_face, critical_turn, frame_nums[0])

    thresholds_passed_idx = np.where(group_idx)[0]
    # thresholds_passed_idx = np.where(thresholds_passed)[0]
    group_counter = 0

    # loop over each frame that passed the threshold
    for idx in thresholds_passed_idx:

        # get frame number
        frame_num = frame_nums[idx]
        # print(present_angular_speed[idx])

        # set scale for size of model mouse
        back_butt_dist = 18

        # draw ellipses representing model mouse
        model_mouse_mask = draw_silhouette(model_mouse_mask_initial.copy(), coordinates, frame_num, back_butt_dist)

        # determine color and darkness
        speed_color = np.array([210, 230, 200])  # blue
        multiplier = 10
        save_multiplier = multiplier

        # modify color and darkness for stimulus driven escapes
        if frame_num >= stim_frame - 30:
            speed_color = np.array([100, 100, 100])
            save_multiplier = 2

        # determine arrow color
        if group_idx[idx]:
            if frame_num < stim_frame - 60:
                line_color = [.6 * x for x in [210, 230, 200]]
                save_line_color = line_color
            else:
                line_color = [100, 100, 100]
                save_line_color = [10, 10, 10]

            # compute procedural vector
            origin = np.array([int(coordinates['center_body_location'][0][frame_num - 1]), int(coordinates['center_body_location'][1][frame_num - 1])])
            endpoint_index = int(group_idx[idx]) #int(frame_num - 1 + group_idx[idx] - end_idx[int(group_idx[idx])])
            endpoint = np.array([int(coordinates['center_body_location'][0][endpoint_index]), int(coordinates['center_body_location'][1][endpoint_index]) ])

            vector_tip = (origin + 20 * (endpoint - origin) / np.sqrt(np.sum( (endpoint - origin)**2)) ).astype(int)

            cv2.arrowedLine(exploration_arena, tuple(origin), tuple(vector_tip), line_color, thickness=1, tipLength=.2)
            cv2.arrowedLine(save_exploration_arena, tuple(origin), tuple(vector_tip), save_line_color, thickness=2, tipLength=.2)

            group_counter += 1

        # create color multiplier to modify image
        color_multiplier = 1 - (1 - speed_color / [255, 255, 255]) / (np.mean(1 - speed_color / [255, 255, 255]) * multiplier)
        save_color_multiplier = 1 - (1 - speed_color / [255, 255, 255]) / (np.mean(1 - speed_color / [255, 255, 255]) * save_multiplier)

        # apply color to arena image
        exploration_arena[model_mouse_mask.astype(bool)] = exploration_arena[model_mouse_mask.astype(bool)] * color_multiplier
        save_exploration_arena[model_mouse_mask.astype(bool)] = save_exploration_arena[model_mouse_mask.astype(bool)] * save_color_multiplier

        # display image
        cv2.imshow(savepath + ' paths', save_exploration_arena)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # apply the contours and border to the image and save the image
    cv2.imshow(savepath + ' paths', save_exploration_arena);
    cv2.waitKey(1)
    session_trials_plot_background[border_size:, 0:-border_size] = save_exploration_arena
    scipy.misc.imsave(os.path.join(savepath, videoname + '_procedural_learning.tif'), cv2.cvtColor(session_trials_plot_background, cv2.COLOR_BGR2RGB))





    return exploration_arena, group_idx, distance_from_start, end_idx




def exploration(exploration_arena_copy, session_plot_background, border_size, coordinates, previous_stim_frame, stim_frame, videoname, savepath, arena):
    '''
    compute and display EXPLORATION
    go through each frame, adding the mouse silhouette
    '''

    # for debugging, make a copy
    exploration_arena = copy.deepcopy(exploration_arena_copy)
    dim_exploration_arena = ((2 * cv2.cvtColor(exploration_arena_copy, cv2.COLOR_BGR2GRAY).astype(float) +
                                arena.astype(float) * 3) / 5).astype(np.uint8)
    exploration_arena_trial = cv2.cvtColor(dim_exploration_arena, cv2.COLOR_GRAY2RGB)

    # initialize the mouse mask
    model_mouse_mask_initial = exploration_arena[:,:,0] * 0

    # get the coordinates up to and including this trial
    current_speeds = coordinates['speed_toward_shelter'][previous_stim_frame:stim_frame]
    distance_from_shelter = coordinates['distance_from_shelter'][previous_stim_frame:stim_frame]

    # get the frame numbers to analyze
    frame_nums = np.arange(previous_stim_frame,stim_frame)

    # do convolutions to get current, future, far future, and past speeds w.r.t the shelter
    filter_sequence_24 = -1 * np.ones(24)
    current_speed = np.concatenate((np.zeros(12 - 1), np.convolve(current_speeds, filter_sequence_24, mode='valid'), np.zeros(12))) / len(filter_sequence_24)

    # loop over each frame that passed the threshold
    for frame_num in frame_nums:

        # on the stimulus onset, get the contours of the mouse body
        if frame_num == stim_frame:
            _, contours, _ = cv2.findContours(model_mouse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # angular speed
        current_speed_toward_shelter = current_speed[frame_num - (previous_stim_frame)]
        current_distance_from_shelter = distance_from_shelter[frame_num - (previous_stim_frame)]

        # stop when at shelter
        # if current_distance_from_shelter < 50:
        #     continue

        # extract DLC coordinates from the saved coordinates dictionary
        body_angle = coordinates['body_angle'][frame_num]
        shoulder_angle = coordinates['shoulder_angle'][frame_num]
        shoulder_location = tuple(coordinates['shoulder_location'][:, frame_num].astype(np.uint16))
        body_location = tuple(coordinates['center_body_location'][:, frame_num].astype(np.uint16))

        # set scale for size of model mouse
        back_butt_dist = 20

        # draw ellipses representing model mouse
        model_mouse_mask = cv2.ellipse(model_mouse_mask_initial.copy(), body_location, (int(back_butt_dist * .7), int(back_butt_dist * .35)), 180 - body_angle, 0, 360, 100, thickness=-1)
        model_mouse_mask = cv2.ellipse(model_mouse_mask , shoulder_location, (int(back_butt_dist), int(back_butt_dist*.2)), 180 - shoulder_angle ,0, 360, 100, thickness=-1)

        # determine color by angular speed
        if current_speed_toward_shelter <= 0:
            speed_color = np.array([190, 189, 240])  # red
            speed_color = np.array([250, 220, 223])  # red
            multiplier = 100
        else:
            speed_color = np.array([190, 240, 190])  # green
            speed_color = np.array([190, 240, 190])  # green
            multiplier = 50

        # create color multiplier to modify image
        color_multiplier = 1 - (1 - speed_color / [255, 255, 255]) / (np.mean(1 - speed_color / [255, 255, 255]) * multiplier)

        # prevent any region from getting too dark (trial)
        if np.mean(exploration_arena_trial[model_mouse_mask.astype(bool)]) < 100:
            continue

        # apply color to arena image (trial)
        exploration_arena_trial[model_mouse_mask.astype(bool)] = exploration_arena_trial[model_mouse_mask.astype(bool)] * color_multiplier

        # prevent any region from getting too dark (session)
        if np.mean(exploration_arena_trial[model_mouse_mask.astype(bool)]) < 100:
            continue

        # apply color to arena image (session)
        exploration_arena[model_mouse_mask.astype(bool)] = exploration_arena[model_mouse_mask.astype(bool)] * color_multiplier

        # display image
        cv2.imshow(savepath +' explore', exploration_arena)
        cv2.imshow(savepath + 'trial explore', exploration_arena_trial)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # apply the contours and border to the image and save the image
    try:
        session_plot_background[border_size:, 0:-border_size] = exploration_arena
        scipy.misc.imsave(os.path.join(savepath, videoname + '_exploration_all.tif'), cv2.cvtColor(session_plot_background, cv2.COLOR_BGR2RGB))

        session_plot_background[border_size:, 0:-border_size] = exploration_arena_trial
        scipy.misc.imsave(os.path.join(savepath, videoname + '_exploration_recent.tif'), cv2.cvtColor(session_plot_background, cv2.COLOR_BGR2RGB))
    except:
        print('repeat stimulus trial')

    # Make a position heat map as well
    scale = 1
    H, x_bins, y_bins = np.histogram2d(coordinates['center_location'][0, 0:stim_frame], coordinates['center_location'][1, 0:stim_frame],
                                       [np.arange(0, exploration_arena.shape[1] + 1, scale), np.arange(0, exploration_arena.shape[0] + 1, scale)], normed=True)
    H = H.T

    H = cv2.GaussianBlur(H, ksize=(5, 5), sigmaX=1, sigmaY=1)
    H[H > np.percentile(H, 98)] = np.percentile(H, 98)

    H_image = (H * 255 / np.max(H)).astype(np.uint8)
    H_image[(H_image < 25) * (H_image > 0)] = 25
    H_image[(arena > 0) * (H_image == 0)] = 9
    H_image = cv2.copyMakeBorder(H_image, border_size, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
    textsize = cv2.getTextSize(videoname, 0, .55, 1)[0]
    textX = int((arena.shape[1] - textsize[0]) / 2)
    cv2.putText(H_image, videoname, (textX, int(border_size * 3 / 4)), 0, .55, (255, 255, 255), thickness=1)

    cv2.imshow('heat map', H_image)
    cv2.waitKey(1)
    scipy.misc.imsave(os.path.join(savepath, videoname + '_exploration2.tif'), H_image)

    return exploration_arena

