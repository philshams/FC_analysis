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
from Utils.obstacle_funcs import set_up_speed_colors
import time
from scipy.ndimage import gaussian_filter1d


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


def create_local_variables(coordinates, stim_frame, previous_stim_frame, skip_frames, obstacle_type = 'wall', shelter_location = False, frame_shape = False):

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

    if shelter_location:
        shelter_location = [s * frame_shape[1-i]/1000 for i, s in enumerate(shelter_location)]
        distance_from_shelter_front = np.sqrt( (x_location_face - shelter_location[0] )**2 + (y_location_face - shelter_location[1])**2 )
    else: distance_from_shelter_front = None

    distance_arena = np.load('C:\\Drive\\DLC\\transforms\\distance_arena_' + obstacle_type + '.npy')
    angle_arena = np.load('C:\\Drive\\DLC\\transforms\\angle_arena_' + obstacle_type + '.npy')

    distance_from_obstacle = distance_arena[y_location_butt, x_location_butt]
    angles_from_obstacle = angle_arena[y_location_butt, x_location_butt]

    return goal_speeds, absolute_speeds, body_angles, subgoal_angles, angular_speed, distance_from_shelter, distance_from_subgoal, within_subgoal_bound, \
           x_location_butt, y_location_butt, x_location_face, y_location_face, distance_from_obstacle, angles_from_obstacle, distance_from_shelter_front

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


def multi_phase_phinder(thresholds_passed, minimum_distance_spont, max_shelter_proximity, body_angles, distance_from_shelter,
                            x_location_butt, y_location_butt, x_location_face, y_location_face, critical_turn, first_frame,
                                distance_from_obstacle, angles_from_obstacle, stim_on, absolute_speeds, shelter_location_front):

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

            # trim the end off it moves away from shelter
            if stim_on[start_index]:
                closest_to_shelter = np.argmin(shelter_location_front[idx - group_length: idx])
                if shelter_location_front[idx - group_length + closest_to_shelter] < 50:
                    idx -= (group_length - (closest_to_shelter + 1) )
                    group_length -= (group_length - (closest_to_shelter + 1) )

                #TEMPORARY: IF START TOO CLOSE, THEN SKIP
                if y_location_face[np.max(np.where(~stim_on[:start_index]))+1] < 23:
                    continue

            # get the distance travelled so far during the bout
            distance_from_start[idx - group_length: idx] = np.sqrt((x_location_butt[idx - group_length: idx] - x_location_butt[idx - group_length]) ** 2 + \
                                                                   (y_location_butt[idx - group_length: idx] - y_location_butt[idx - group_length]) ** 2)

            angle_for_comparison = 0; end_of_turn = 0; start_of_turn = 0
            if stim_on[start_index]: minimum_distance = minimum_distance_spont - 10
            else: minimum_distance = minimum_distance_spont# + 10

            while True:
                # get the cumulative distance traveled
                distance_traveled = np.sqrt((x_location_butt[start_index:idx] - x_location_butt[start_index]) ** 2
                                            + (y_location_butt[start_index:idx] - y_location_butt[start_index]) ** 2)

                # has the minimum distance been traveled
                traveled_far_enough = np.where(distance_traveled > minimum_distance)[0]
                if not traveled_far_enough.size: break

                # now check for more phases: find the cumulative turn angle since the start of the bout
                angle_for_comparison_index = np.max((start_index + traveled_far_enough[0], start_index + end_of_turn - start_of_turn))
                angle_for_comparison = body_angles[angle_for_comparison_index]

                angle_turned = np.zeros((idx-start_index, 2))
                if stim_on[start_index]:
                    angle_turned[:,0] = (body_angles[start_index:idx] - angle_for_comparison) #body_angles[start_index + traveled_far_enough[0]])
                    angle_turned[:, 1] = (body_angles[start_index:idx] - body_angles[start_index - 15:idx - 15])
                else:
                    angle_turned[:,0] = (gaussian_filter1d(body_angles[start_index:idx],3) - angle_for_comparison)
                    try: angle_turned[:, 1] = (gaussian_filter1d(body_angles[start_index:idx],3) - gaussian_filter1d(body_angles[start_index - 15:idx - 15],3))
                    except: break

                angle_turned[angle_turned > 180] = 360 - angle_turned[angle_turned > 180]
                angle_turned[angle_turned < -180] = 360 + angle_turned[angle_turned < -180]

                # not including the beginning
                # angle_turned[:traveled_far_enough[0], 0] = 0
                zero_index = np.max((end_of_turn-start_of_turn, traveled_far_enough[0]))
                angle_turned[:zero_index, 0] = 0
                angle_turned[:traveled_far_enough[0]+15, 1] = 0
                max_turn_idx = np.argmax(abs(angle_turned), 1)
                max_angle_turned = angle_turned[np.arange(0, idx - start_index), max_turn_idx]

                # give a bonus for being near obstacle?.. #RECENTLY REMOVED
                # near_obstacle = distance_from_obstacle[start_index:idx] < 25
                # above_obstacle = angles_from_obstacle[start_index:idx] == -90
                # max_angle_turned[(near_obstacle * above_obstacle).astype(bool)] = max_angle_turned[(near_obstacle * above_obstacle).astype(bool)] * 4 #1.75


                # get the indices of critically large turns
                critically_turned = np.where(abs(max_angle_turned) > critical_turn)[0]

                if critically_turned.size:

                    if stim_on[start_index]:
                        # check if there was a pause in the meanwhile!
                        bc = scipy.signal.boxcar(10) / 10
                        pausing = scipy.signal.convolve(absolute_speeds[start_index:idx], bc, mode='same') < 2 # was 3!
                        pausing[:15] = False; pausing[critically_turned[0]:] = False
                        # print(pausing)
                        take_a_break = np.where(pausing)[0]
                        # print(take_a_break)
                    else: take_a_break = np.array([])
                    if take_a_break.size:
                        start_of_turn = take_a_break[0]
                        end_of_turn = take_a_break[-1]

                    else:
                        # get the angular speed
                        angular_speed = np.diff(angle_turned[:critically_turned[0],0])
                        angular_speed_after = np.diff(angle_turned[critically_turned[0]:,0])

                        # find the beginning of the turn
                        try: start_of_turn = np.max(np.where( ((angular_speed*np.sign(angle_turned[critically_turned[0],0]))<-.05) )) + 1
                        except: start_of_turn = np.max(np.where( ((angular_speed*np.sign(angle_turned[critically_turned[0],0]))<=0) )) + 1

                        # find the end of the turn
                        stops_turning = np.where(np.sign(angular_speed_after) != np.sign(angle_turned[critically_turned[0],0]))[0]
                        if stops_turning.size: end_of_turn = stops_turning[0] + critically_turned[0]
                        else: end_of_turn = critically_turned[0]

                    # break if getting too close to shelter
                    if (distance_from_shelter[start_index + start_of_turn] < max_shelter_proximity):
                        group_idx[start_index] = idx + first_frame
                        end_idx[idx] = start_index + first_frame
                        break
                    else:
                        group_idx[start_index] = start_index + start_of_turn - 1 + first_frame
                        end_idx[start_index + start_of_turn - 1] = start_index + first_frame
                        start_index += start_of_turn
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
#
# def spontaneous_homings(exploration_arena_copy, session_trials_plot_background, border_size, coordinates, previous_stim_frame, stim_frame, videoname, savepath,
#         subgoal_locations, obstacle_type, width, height, make_vid = False):
#     '''
#     compute and display SPONTANEOUS HOMINGS
#     '''
#
#    # make video
#     if make_vid:
#         fourcc = cv2.VideoWriter_fourcc(*"XVID")
#         video_clip = cv2.VideoWriter(os.path.join(savepath, videoname + ' homings.avi'), fourcc, 160,
#                                   (exploration_arena_copy.shape[1], exploration_arena_copy.shape[1]), True)
#
#     # visualize results
#     cv2.namedWindow(savepath + 'homings');
#
#     # how many frames after each stimulus to look at
#     skip_frames = 300 + 150*(obstacle_type=='void')
#     if previous_stim_frame == 0: previous_stim_frame -= skip_frames
#
#     # initialize arrays
#     exploration_arena, save_exploration_arena, model_mouse_mask, thresholds_passed, stimulus_started = \
#         initialize_arrays(exploration_arena_copy, stim_frame, previous_stim_frame)
#
#     # get the frame numbers to analyze
#     frame_nums = np.arange(previous_stim_frame + skip_frames, stim_frame + skip_frames)
#
#     # set the speed parameters
#     speeds = [[.5, .75, 1], [.5, 1, 4]]
#     speed_colors = [np.array([20, 254, 140]), np.array([20, 254, 140])]
#     multipliers = [50, 50]
#     smooth_duration = 30
#
#     # set distance parameters
#     close_to_shelter_distance = 40
#     minimum_shelter_distance = 150
#     close_to_shelter_angle = 30
#     small_shelter_angle = 10
#     strict_shelter_angle = 85 + (obstacle_type=='side wall')*10
#
#     # create local variables for the current epoch
#     model_arena((height, width), 1, False, obstacle_type=obstacle_type, simulate=True)
#
#     goal_speeds, absolute_speeds, shelter_angles, subgoal_angles, angular_speed, distance_from_shelter, distance_from_subgoal, within_subgoal_bound, _, _, _, _, _, _, _ = \
#         create_local_variables(coordinates, stim_frame, previous_stim_frame, skip_frames, obstacle_type=obstacle_type)
#
#     # do convolutions to get current, future, far future, and past speeds w.r.t the shelter
#     current_speed = convolve(goal_speeds, 24, -1, time='current')
#     past_speed = convolve(goal_speeds, 45, -1, time='past')
#     future_speed = convolve(goal_speeds, 60, -1, time='future')
#     far_future_speed = convolve(goal_speeds, 60, -1, time='far future', time_chase=20)
#
#     # is the mouse close enough to the shelter
#     minimum_distance = 400 + 100 * (obstacle_type=='void')
#     close_enough_to_shelter = convolve(distance_from_shelter < minimum_distance, 60, +1, time='future')
#     angle_thresholded = threshold(abs(subgoal_angles), close_to_shelter_angle, '<')
#     strict_angle_thresholded = threshold(abs(subgoal_angles), strict_shelter_angle, '<') + threshold(frame_nums, stim_frame) #TEMPRARY COMMENDED STIM FRAME BIT
#     distance_thresholded = threshold(distance_from_shelter, close_to_shelter_distance)
#
#     # loop across fast and slow homings
#     for speed_setting in [True, False]:
#
#         # use higher speed setting to make faster homings darker
#         low_speed, medium_speed, high_speed = speeds[speed_setting][0], speeds[speed_setting][1], speeds[speed_setting][2]
#
#         # threshold the convolved speeds to determine which frames to draw
#         current_speed_thrsh = threshold(current_speed, high_speed)
#         future_speed_thrsh = threshold(future_speed, high_speed) * threshold(current_speed, medium_speed)
#         far_future_speed_thrsh = threshold(far_future_speed, high_speed) * threshold(current_speed, medium_speed)
#         past_speed_thrsh = threshold(past_speed, high_speed) * threshold(current_speed, medium_speed)
#         stimulus_thresholded = threshold(frame_nums, stim_frame - 1 + smooth_duration) * threshold(current_speed, low_speed)
#
#         # combine speed thresholds into one
#         combined_speed_thresholds = (current_speed_thrsh + future_speed_thrsh + far_future_speed_thrsh + past_speed_thrsh)
#
#         # combine all thresholds into one
#         thresholds_passed_first_pass = distance_thresholded * (angle_thresholded * close_enough_to_shelter * combined_speed_thresholds+ stimulus_thresholded)
#         #TEMPORARY -- REMOVED STIMULUS THRESHOLD
#
#         # and smooth it *out*
#         thresholds_passed_first_pass = (strict_angle_thresholded * convolve(thresholds_passed_first_pass, smooth_duration, +1, time='current') ).astype(bool)
#
#         # finally, add a minimum duration threshold
#         minimum_distance = -.5 + .3 * speed_setting + (obstacle_type=='void')*.3 # this will be problematic for other arenas
#         minimum_duration = smooth_duration + 5
#
#         minimum_distance = (-.4 + .2 * speed_setting) + (obstacle_type=='void')*.3
#         minimum_duration = smooth_duration + 5
#         # print(speed_setting)
#         thresholds_passed_first_pass, group_idx = get_homing_groups(thresholds_passed_first_pass * (1 - thresholds_passed),
#                                                                          minimum_distance, minimum_duration, distance_from_subgoal, distance_from_shelter,
#                                                                         minimum_shelter_distance, absolute_speeds)
#
#         # set function output
#         trial_groups = thresholds_passed_first_pass + thresholds_passed
#
#         # get the index when adequately long locomotor bouts pass the threshold
#         thresholds_passed = thresholds_passed_first_pass  # + (frame_nums==stim_frame)
#
#         thresholds_passed_idx = np.where(thresholds_passed)[0]
#         group_counter = 0
#         bout_starting = True
#         model_mouse_mask_total = model_mouse_mask.copy()
#
#         # loop over each frame that passed the threshold
#         for idx in thresholds_passed_idx:
#
#             # get frame number
#             frame_num = frame_nums[idx]
#             if frame_num >= stim_frame-10:
#                 break
#
#             # set up new bout
#             if group_idx[idx] and not group_counter:
#                 prior_exploration_arena = exploration_arena.copy(); group_length = group_idx[idx]
#                 model_mouse_mask_total = model_mouse_mask.copy(); bout_starting = False
#             group_counter += 1
#
#             # x_loc = coordinates['center_body_location'][0, frame_num - group_counter:frame_num - group_counter + int(group_length)]
#             # y_loc = coordinates['center_body_location'][1, frame_num - group_counter:frame_num - group_counter + int(group_length)]
#             # if np.any( (x_loc > 450) * (y_loc < 250)):
#             #     continue
#
#             # draw model mouse
#             back_butt_dist = 11 + (frame_num >= stim_frame) * 2
#             time_chase = 80 - (frame_num >= stim_frame) * 60
#             model_mouse_mask, model_mouse_mask_total = draw_mouse(model_mouse_mask, model_mouse_mask_total, coordinates, frame_num, back_butt_dist,
#                                                                   group_idx[idx], stim_frame, stimulus_started, group_counter, time_chase)
#
#             # determine color and darkness
#             blue_duration = 80 - max(0, 90 - group_length)
#             color_multiplier, save_color_multiplier = get_color_settings(frame_num, stim_frame, speed_setting, speed_colors, multipliers,
#                                                                          exploration_arena, model_mouse_mask, group_counter, blue_duration)
#
#             # on the stimulus onset, get the contours of the mouse body
#             if (frame_num >= stim_frame or not speed_setting) and not stimulus_started:
#                 save_exploration_arena, stimulus_started, _ = stimulus_onset(stimulus_started, model_mouse_mask, exploration_arena)
#                 continue
#
#             # apply color to arena image
#             exploration_arena[model_mouse_mask.astype(bool)] = exploration_arena[model_mouse_mask.astype(bool)] * color_multiplier
#
#             # apply color to this trial's image
#             if stimulus_started:
#                 save_exploration_arena[model_mouse_mask.astype(bool)] = save_exploration_arena[model_mouse_mask.astype(bool)] * save_color_multiplier
#
#             # dilute the color at the end of the bout
#             if group_idx[idx] and group_counter > 1:
#                 exploration_arena, save_exploration_arena = dilute_shading(exploration_arena, save_exploration_arena, prior_exploration_arena,
#                                                                            model_mouse_mask_total, stimulus_started, group_idx[idx], speed_setting)
#                 group_counter = 0
#
#             # display image
#             cv2.imshow(savepath + 'homings', exploration_arena)
#
#             if make_vid: video_clip.write(exploration_arena)
#
#             # press q to quit
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#
#
#     # apply the contours and border to the image and save the image
#     cv2.imshow(savepath + 'homings', save_exploration_arena); cv2.waitKey(1)
#     session_trials_plot_background[border_size:, 0:-border_size] = save_exploration_arena
#     scipy.misc.imsave(os.path.join(savepath, videoname + '_spont_homings.tif'), cv2.cvtColor(session_trials_plot_background, cv2.COLOR_BGR2RGB))
#
#     if make_vid: video_clip.release()
#
#     return exploration_arena, trial_groups


def spontaneous_homings(exploration_arena_copy, session_trials_plot_background, border_size, coordinates, previous_stim_frame, stim_frame, videoname, savepath,
        subgoal_locations, obstacle_type, width, height, make_vid = False):
    '''
    compute and display SPONTANEOUS HOMINGS
    '''

   # make video
    if make_vid:
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
    model_arena((height, width), 1, False, obstacle_type=obstacle_type, simulate=True)

    goal_speeds, absolute_speeds, shelter_angles, subgoal_angles, angular_speed, distance_from_shelter, distance_from_subgoal, within_subgoal_bound, _, _, _, _, _, _, _ = \
        create_local_variables(coordinates, stim_frame, previous_stim_frame, skip_frames, obstacle_type=obstacle_type)

    # do convolutions to get current, future, far future, and past speeds w.r.t the shelter
    current_speed = convolve(goal_speeds, 24, -1, time='current')
    past_speed = convolve(goal_speeds, 45, -1, time='past')
    future_speed = convolve(goal_speeds, 60, -1, time='future')
    far_future_speed = convolve(goal_speeds, 60, -1, time='far future', time_chase=20)

    # is the mouse close enough to the shelter
    minimum_distance = 400 + 100 * (obstacle_type=='void')
    close_enough_to_shelter = convolve(distance_from_shelter < minimum_distance, 60, +1, time='future')
    angle_thresholded = threshold(abs(subgoal_angles), close_to_shelter_angle, '<')
    strict_angle_thresholded = threshold(abs(subgoal_angles), strict_shelter_angle, '<') + threshold(frame_nums, stim_frame) #TEMPRARY COMMENDED STIM FRAME BIT
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
        thresholds_passed_first_pass = distance_thresholded * (angle_thresholded * close_enough_to_shelter * combined_speed_thresholds+ stimulus_thresholded)
        #TEMPORARY -- REMOVED STIMULUS THRESHOLD

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
        # model_mouse_mask_total = model_mouse_mask.copy()

        # smooth speed trace for coloration
        smoothed_speed = np.concatenate((np.zeros(6 - 1), np.convolve(coordinates['speed'], np.ones(12), mode='valid'), np.zeros(6))) / 12
        counter = 0

        # loop over each frame that passed the threshold
        for idx in thresholds_passed_idx:

            # get frame number
            frame_num = frame_nums[idx]
            if frame_num >= stim_frame-10:
                break

            # set up new bout
            if group_idx[idx] and not group_counter:
                prior_exploration_arena = exploration_arena.copy(); group_length = group_idx[idx]
                bout_starting = False
                start_position = coordinates['center_location'][1][frame_num-1]

                end_position = coordinates['center_location'][1][frame_num-1 + int(group_length)]
            if end_position < 324:
                continue

            group_counter += 1


            # draw model mouse
            back_butt_dist = 13

            # draw ellipses representing model mouse
            model_mouse_mask = draw_silhouette(np.zeros_like(exploration_arena)[:,:,0], coordinates, frame_num, back_butt_dist)

            speed = smoothed_speed[frame_num - 1] * 1.7
            speed_color_light, speed_color_dark = set_up_speed_colors(speed, spontaneous = True)

            # emphasize the back
            if start_position > 280:
                speed_color_light, speed_color_dark = speed_color_light**.3, speed_color_dark**.3

            if frame_num >= stim_frame - 10:
                speed_color_light, speed_color_dark = np.ones(3)*np.mean(speed_color_light), np.ones(3)*np.mean(speed_color_dark)


                # apply color to arena image
            if counter*speed > 80:
                exploration_arena[model_mouse_mask.astype(bool)] = exploration_arena[model_mouse_mask.astype(bool)] * speed_color_dark
                counter = 0
            else:
                exploration_arena[model_mouse_mask.astype(bool)] = exploration_arena[model_mouse_mask.astype(bool)] * speed_color_light
                counter += 1

            # dilute the color at the end of the bout
            if group_idx[idx] and group_counter > 1:
                group_counter = 0

            # display image
            cv2.imshow(savepath + 'homings', exploration_arena)

            if make_vid: video_clip.write(exploration_arena)

            # press q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # apply the contours and border to the image and save the image
    save_exploration_arena = exploration_arena
    cv2.imshow(savepath + 'homings', save_exploration_arena); cv2.waitKey(1)
    session_trials_plot_background[border_size:, 0:-border_size] = save_exploration_arena
    scipy.misc.imsave(os.path.join(savepath, videoname + '_spont_homings.tif'), cv2.cvtColor(session_trials_plot_background, cv2.COLOR_BGR2RGB))

    if make_vid: video_clip.release()

    return exploration_arena, trial_groups


def procedural_learning(exploration_arena_copy, session_trials_plot_background, border_size, coordinates,
        previous_stim_frame, stim_frame, videoname, savepath, subgoal_locations, trial_groups, obstacle_type, shelter_location):
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
    skip_frames = 300 + 150*(obstacle_type=='void')
    if previous_stim_frame == 0: previous_stim_frame -= skip_frames

    # set what this mouse considers a high speed
    speeds = [.5, 1, 4]
    high_speed, medium_speed, low_speed = speeds[2], speeds[1], speeds[0]

    # create local variables for the current epoch
    goal_speeds, absolute_speeds, body_angles, subgoal_angles, angular_speed, distance_from_shelter, _, _, \
    x_location_butt, y_location_butt, x_location_face, y_location_face, distance_from_obstacle, angles_from_obstacle, shelter_location_front = \
        create_local_variables(coordinates, stim_frame, previous_stim_frame, skip_frames, obstacle_type = obstacle_type, shelter_location = shelter_location, frame_shape = exploration_arena.shape)

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
    minimum_distance = 50 # 50
    max_shelter_proximity = 50 # 200 #150
    critical_turn = 45 #was 45, and replaced subgoal_angles with body_angles
    critical_turn = 25  # was 45, then 25 and replaced subgoal_angles with body_angles
    group_idx, distance_from_start, end_idx = multi_phase_phinder(thresholds_passed, minimum_distance, max_shelter_proximity, body_angles, distance_from_shelter,
                                             x_location_butt, y_location_butt, x_location_face, y_location_face, critical_turn, frame_nums[0],
                                          distance_from_obstacle, angles_from_obstacle, frame_nums >= stim_frame, absolute_speeds, shelter_location_front)

    thresholds_passed_idx = np.where(group_idx)[0]
    # thresholds_passed_idx = np.where(thresholds_passed)[0]
    group_counter = 0

    # loop over each frame that passed the threshold
    for idx in thresholds_passed_idx:

        # get frame number
        frame_num = frame_nums[idx]
        # print(present_angular_speed[idx])
        # if frame_num >= stim_frame:
        #     continue

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




def exploration(session_plot_background, border_size, coordinates, previous_stim_frame, stim_frame, videoname, savepath, arena):
    '''
    compute and display EXPLORATION
    go through each frame, adding the mouse silhouette
    '''

    # for debugging, make a copy
    exploration_arena_trial = cv2.cvtColor(arena, cv2.COLOR_GRAY2RGB)

    # initialize the mouse mask
    model_mouse_mask_initial = exploration_arena_trial[:,:,0] * 0

    # get the coordinates up to and including this trial
    current_speeds = coordinates['speed_toward_shelter'][previous_stim_frame:stim_frame]
    distance_from_shelter = coordinates['distance_from_shelter'][previous_stim_frame:stim_frame]

    # get the frame numbers to analyze
    frame_nums = np.arange(previous_stim_frame,stim_frame)

    # find when last in shelter
    last_in_shelter = np.where(distance_from_shelter < 30)[0]
    if last_in_shelter.size: start_idx = last_in_shelter[-1]
    else: start_idx = 0

    # set up coloring
    total_time = len(frame_nums[start_idx:])
    start_frame = frame_nums[start_idx]

    # loop over each frame that passed the threshold
    for frame_num in frame_nums[start_idx:]:

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

        # determine color by time since shelter
        f1 = (frame_num - start_frame)/total_time
        f2 = 1 - f1

        time_color = f1 * np.array([190, 240, 190]) + f2 * np.array([250, 220, 223])
        multiplier = f1 * 40 + f2 * 80

        # create color multiplier to modify image
        color_multiplier = 1 - (1 - time_color / [255, 255, 255]) / (np.mean(1 - time_color / [255, 255, 255]) * multiplier)


        # prevent any region from getting too dark (trial)
        # if np.mean(exploration_arena_trial[model_mouse_mask.astype(bool)]) < 100:
        #     continue

        # apply color to arena image (trial)
        exploration_arena_trial[model_mouse_mask.astype(bool)] = exploration_arena_trial[model_mouse_mask.astype(bool)] * color_multiplier

        # display image
        cv2.imshow(savepath + 'trial explore', exploration_arena_trial)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # color in final position in red
    _, contours, _ = cv2.findContours(model_mouse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(exploration_arena_trial, contours, 0, (0,0,160), thickness=2)

    # apply the contours and border to the image and save the image
    try:
        session_plot_background[border_size:, 0:-border_size] = exploration_arena_trial
        scipy.misc.imsave(os.path.join(savepath, videoname + '_exploration_recent.tif'), cv2.cvtColor(session_plot_background, cv2.COLOR_BGR2RGB))
    except:
        print('repeat stimulus trial')

    '''
    make a position heat map
    '''

    # Make a position heat map as well
    scale = 1
    # average all mice data
    position = coordinates['butty_location']
    H, x_bins, y_bins = np.histogram2d(position[0, 0:stim_frame], position[1, 0:stim_frame],
                                       [np.arange(0, exploration_arena_trial.shape[1] + 1, scale), np.arange(0, exploration_arena_trial.shape[0] + 1, scale)], normed=True)
    exploration_all = H.T

    # gaussian blur
    exploration_blur = cv2.GaussianBlur(exploration_all, ksize=(201, 201), sigmaX=13, sigmaY=13)

    # normalize
    exploration_blur = (exploration_blur / np.percentile(exploration_blur, 98.7) * 255)
    exploration_blur[exploration_blur > 255] = 255
    exploration_all[exploration_all>0] = 255

    # change color map
    exploration_blur = cv2.applyColorMap(exploration_blur.astype(np.uint8), cv2.COLORMAP_OCEAN)
    exploration_all = cv2.cvtColor(exploration_all.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    # exploration_blur[arena == 0] = 0

    # make composite image
    # exploration_image = ((exploration_all.astype(float) + exploration_blur.astype(float)) / 2)
    exploration_image = exploration_all.astype(float)*.8 + exploration_blur.astype(float) / 2
    exploration_image[exploration_image > 255] = 255
    exploration_image = exploration_image.astype(np.uint8)

    exploration_image[(arena > 0) * (exploration_image[:,:,0] < 10)] = 15
    # exploration_all[(arena > 0) * (exploration_all[:, :, 0] == 0)] = 20
    # exploration_image_save = cv2.copyMakeBorder(exploration_image, border_size, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
    # textsize = cv2.getTextSize(videoname, 0, .55, 1)[0]
    # textX = int((arena.shape[1] - textsize[0]) / 2)
    # cv2.putText(exploration_image_save, videoname, (textX, int(border_size * 3 / 4)), 0, .55, (255, 255, 255), thickness=1)

    cv2.imshow('heat map', exploration_image)
    # cv2.imshow('traces', exploration_all)
    cv2.waitKey(1)
    exploration_image = cv2.cvtColor(exploration_image, cv2.COLOR_RGB2BGR)
    scipy.misc.imsave(os.path.join(savepath, videoname + '_exploration.tif'), exploration_image)

    '''
    make the successor representation
    '''

    # # Histogram of positions
    # bins_per_side = 9
    # bin_size = int(arena.shape[0] / bins_per_side)
    #
    # # just use homings!
    # start_idx = np.where(coordinates['start_index'][:stim_frame])[0]
    # end_idx = coordinates['start_index'][start_idx]
    #
    # # homing_idx = np.arange(stim_frame)
    # bin_counts = np.zeros((bins_per_side ** 2, bins_per_side ** 2), dtype=int)
    # homing_idx = np.array([], int)
    # for s, e in zip(start_idx, end_idx):
    #     homing_idx = np.arange(s, e).astype(int)
    #
    #     # bin the position data
    #     bin_array = np.ones(len(coordinates['center_location'][0][homing_idx])) * np.nan
    #     bin_ID = 0
    #     bin_sum = 0
    #     if coordinates['center_location'][coordinates['center_location'] >= arena.shape[0]].size:
    #         coordinates['center_location'][coordinates['center_location'] >= arena.shape[0]] = arena.shape[0] - 1
    #
    #     for x_bin in range(bins_per_side):
    #         in_x_bin = (coordinates['center_location'][0][homing_idx] >= (bin_size * x_bin)) * (
    #                     coordinates['center_location'][0][homing_idx] < (bin_size * (x_bin + 1)))
    #
    #         for y_bin in range(bins_per_side):
    #             in_y_bin = (coordinates['center_location'][1][homing_idx] >= (bin_size * y_bin)) * (
    #                         coordinates['center_location'][1][homing_idx] < (bin_size * (y_bin + 1)))
    #
    #             bin_array[in_x_bin * in_y_bin] = bin_ID
    #             bin_sum += np.sum(bin_array)
    #             bin_ID += 1
    #
    #     bin_array = bin_array.astype(int)
    #
    #
    #     # Get counts of (bin_preceding --> bin_following) to (i,j) of bin_counts
    #     bin_counts_homing = np.zeros((bins_per_side ** 2, bins_per_side ** 2), dtype=int)
    #     np.add.at(bin_counts_homing, (bin_array[:-1], bin_array[1:]), 1)
    #
    #     bin_counts += bin_counts_homing
    #
    # # Get transition probs (% in column 1 going to row 1,2,3,etc.)
    # transition = (bin_counts.T / np.sum(bin_counts, axis=1)).T
    # np.nan_to_num(transition, copy=False)
    #
    # # get the SR
    # time_discount = .90  # .99 ~ 1 sec is .74 // 5 sec is .2 // 10 sec is .05
    # successor_representation = np.linalg.inv(np.identity(bins_per_side ** 2) - time_discount * transition)
    #
    #
    # # define shelter location etc
    # spatial_blurring = 2
    #
    # # shelter_indices = [(x, y) for x in range(8, 12) for y in range(16, 20)]
    # shelter_indices = [(x, y) for x in range(4, 6) for y in range(8, 10)]
    # shelter_bin_idx = [(SI[0] * bins_per_side) + SI[1] for SI in shelter_indices]
    #
    # x_bin_body = int(position[0][stim_frame] / bin_size)
    # y_bin_body = int(position[1][stim_frame] / bin_size)
    #
    # # initialize SR arenaq
    # successor_arena = np.zeros(arena.shape)
    #
    # # also use nearby points
    # for x_shift in range(-spatial_blurring, spatial_blurring + 1):
    #     for y_shift in range(-spatial_blurring, spatial_blurring + 1):
    #
    #         # initialize SR arenaq
    #         curr_successor_arena = np.zeros(arena.shape)
    #
    #         # get the SR for the clicked bin
    #         start_bin_idx = ((x_bin_body + x_shift) * bins_per_side) + (y_bin_body + y_shift)
    #
    #         # fill in each bin with the successor-ness
    #         for x_bin in range(bins_per_side):
    #             for y_bin in range(bins_per_side):
    #                 # get the bin index
    #                 end_bin_idx = (x_bin * bins_per_side) + y_bin
    #                 if end_bin_idx in shelter_bin_idx: continue
    #
    #                 if end_bin_idx != start_bin_idx:
    #                     # if np.mean(successor_representation[end_bin_idx, shelter_bin_idx]) > np.mean(successor_representation[start_bin_idx, shelter_bin_idx]):
    #                     # if ( abs(x_bin - x_bin_body)>2 or abs(y_bin - y_bin_body)>2 ) and ( abs(x_bin - 4.5)>2 or abs(y_bin - 8.5)>2 ):
    #                     curr_successor_arena[int(bin_size * (y_bin + .5)), int(bin_size * (x_bin + .5))] = \
    #                     successor_representation[start_bin_idx, end_bin_idx] #\
    #                         # np.mean(successor_representation[end_bin_idx, shelter_bin_idx]) *
    #                         # / np.max((1, np.sqrt((x_bin - x_bin_body) ** 2 + (y_bin - y_bin_body) ** 2) ** 4))
    #
    #         successor_arena = successor_arena + (curr_successor_arena)   #/ np.max((1, 1 * np.sqrt(x_shift ** 2 + y_shift ** 2)))
    #
    # # save results
    # # successor_from_stimulus = np.zeros((bins_per_side, bins_per_side))
    # # for x_bin in range(bins_per_side):
    # #     for y_bin in range(bins_per_side):
    # #         successor_from_stimulus[y_bin, x_bin] = successor_arena[int(bin_size * (y_bin + .5)), int(bin_size * (x_bin + .5))]
    #
    # # gaussian blur
    # # successor_arena = cv2.GaussianBlur(successor_arena, ksize=(181, 181), sigmaX=20, sigmaY=20)
    # successor_arena = cv2.GaussianBlur(successor_arena, ksize=(251, 251), sigmaX=bin_size/2, sigmaY=bin_size/2)
    #
    # print(np.max(successor_arena))
    # successor_arena = (successor_arena /  np.max(successor_arena)* 255)   # / np.max(successor_arena) #.003
    # successor_arena[successor_arena > 255] = 255
    # successor_arena[arena == 0] = 0
    # successor_arena = successor_arena.astype(np.uint8)
    #
    # color_successor = cv2.applyColorMap(cv2.resize(successor_arena, arena.shape), cv2.COLORMAP_OCEAN)
    #
    # # show arena and mouse position
    # # color_successor[(np.mean(color_successor,2)==0) * (arena == 255)] = 20
    # color_successor[(arena > 0) * (arena < 255)] = 20
    # cv2.circle(color_successor, (int(position[0, stim_frame]), int(position[1, stim_frame])), 5, (0, 0, 255), -1)
    #
    # cv2.imshow('SR', color_successor)
    #
    # scipy.misc.imsave(os.path.join(savepath, videoname + '_SR.tif'), cv2.cvtColor(color_successor, cv2.COLOR_RGB2BGR))

    # get mean position at center


