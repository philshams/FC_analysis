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

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def point_in_polygon(pt, corners, inf = np.inf):
    result = False
    for i in range(len(corners)-1):
        if intersect((corners[i][0], corners[i][1]), ( corners[i+1][0], corners[i+1][1]), (pt[0], pt[1]), (inf, pt[1])):
            result = not result
    if intersect((corners[-1][0], corners[-1][1]), (corners[0][0], corners[0][1]), (pt[0], pt[1]), (inf, pt[1])):
        result = not result
    return result


def spontaneous_homings(exploration_arena_copy, session_trials_plot_background, border_size, coordinates, previous_stim_frame, stim_frame, videoname, savepath, subgoal_locations):
    '''
    compute and display EXPLORATION
    go through each frame, adding the mouse silhouette
    '''

    # for debugging, make a copy
    exploration_arena = copy.deepcopy(exploration_arena_copy)
    save_exploration_arena = exploration_arena.copy()

    # instead of speed to shelter, take max of speed to shelter and speed to subgoals
    skip_frames = 300
    coordinates['speed_toward_subgoal'] = np.zeros((len(subgoal_locations['sub-goals'])+1, stim_frame - previous_stim_frame ))
    coordinates['speed_toward_subgoal'][0,:] = coordinates['speed_toward_shelter'][previous_stim_frame + skip_frames:stim_frame + skip_frames]
    coordinates['center_location'][0][coordinates['center_location'][0] >= exploration_arena.shape[1]] = exploration_arena.shape[1] - 1
    coordinates['center_location'][1][coordinates['center_location'][1] >= exploration_arena.shape[0]] = exploration_arena.shape[0] - 1
    x_location = coordinates['center_location'][0][previous_stim_frame + skip_frames:stim_frame + skip_frames].astype(np.uint16)
    y_location = coordinates['center_location'][1][previous_stim_frame + skip_frames:stim_frame + skip_frames].astype(np.uint16)
    subgoal_bound = [ (int(x * exploration_arena.shape[1] / 1000), int(y* exploration_arena.shape[0] / 1000)) for x, y in subgoal_locations['region'] ]

    # compute distance from subgoal
    polygon_mask = np.zeros(exploration_arena.shape[0:2])
    cv2.drawContours(polygon_mask, [np.array(subgoal_bound)], 0, 100, -1)
    polygon_mask = polygon_mask.astype(bool)

    for i, sg in enumerate(subgoal_locations['sub-goals']):
        # calculate distance to subgoal
        distance_from_subgoal = np.sqrt((x_location - sg[0] * exploration_arena.shape[1] / 1000) ** 2 +
                                                   (y_location - sg[1] * exploration_arena.shape[0] / 1000) ** 2)
        # compute valid locations to go to subgoal
        within_subgoal_bound = []
        for x, y in zip(x_location, y_location):
            # within_subgoal_bound.append(point_in_polygon([x, y],subgoal_bound))
            within_subgoal_bound.append(polygon_mask[y, x] )

        # compute speed w.r.t. subgoal
        coordinates['speed_toward_subgoal'][i+1, :] = np.concatenate( ([0], np.diff(distance_from_subgoal))) * within_subgoal_bound

    coordinates['speed_toward_subgoal'] = np.min(coordinates['speed_toward_subgoal'], 0)


    # initialize the mouse mask
    model_mouse_mask_initial = exploration_arena[:, :, 0] * 0

    # set what this mouse considers a high speed
    speeds = [.5, 1, 2, 4]

    # initialize thresholds passed array
    thresholds_passed = np.zeros(stim_frame - previous_stim_frame)
    stimulus_contour_retrieved = False

    # get the coordinates up to and including this trial
    # current_speeds = coordinates['speed_toward_shelter'][previous_stim_frame + skip_frames:stim_frame + skip_frames]
    current_speeds = coordinates['speed_toward_subgoal']
    absolute_speeds = coordinates['speed'][previous_stim_frame + skip_frames:stim_frame + skip_frames]
    shelter_angles = coordinates['shelter_angle'][previous_stim_frame + skip_frames:stim_frame + skip_frames]
    angular_speed = abs(np.concatenate(([0], np.diff(shelter_angles))))
    angular_speed[angular_speed > 180] = 360 - angular_speed[angular_speed > 180]
    distance_from_shelter = coordinates['distance_from_shelter'][previous_stim_frame + skip_frames:stim_frame + skip_frames]

    # do convolutions to get current, future, far future, and past speeds w.r.t the shelter
    filter_sequence_24 = -1 * np.ones(24)
    filter_sequence_30 = -1 * np.ones(30)
    filter_sequence_45 = -1 * np.ones(45)
    filter_sequence_60 = -1 * np.ones(60)

    past_speed = np.concatenate((np.zeros(len(filter_sequence_45) - 1), np.convolve(current_speeds, filter_sequence_45, mode='valid'))) / len(
        filter_sequence_45)
    future_speed = np.concatenate((np.convolve(current_speeds, filter_sequence_60, mode='valid'), np.zeros(len(filter_sequence_60) - 1))) / len(
        filter_sequence_60)
    far_future_speed = np.concatenate((np.convolve(current_speeds, filter_sequence_60, mode='valid'), np.zeros(len(filter_sequence_60) - 1 + 20))) / len(
        filter_sequence_60)
    far_future_speed = far_future_speed[20:]
    current_speed = np.concatenate((np.zeros(12 - 1), np.convolve(current_speeds, filter_sequence_24, mode='valid'), np.zeros(12))) / len(filter_sequence_24)


    for speed_setting in [True, False]:

        # use higher speed setting to make faster homings darker
        high_speed = speeds[2 + int(speed_setting)]
        medium_speed = speeds[1 + int(speed_setting)]
        low_speed = speeds[0]

        # get the frame numbers to analyze
        frame_nums = np.arange(previous_stim_frame + skip_frames,stim_frame + skip_frames)

        # threshold the convolved speeds to determine which frames to draw
        current_thresholded = (current_speed > high_speed).astype(np.uint16)
        future_thresholded = (future_speed > high_speed) * (current_speed > medium_speed).astype(np.uint16)
        far_future_thresholded = (far_future_speed > high_speed) * (current_speed > medium_speed).astype(np.uint16)
        past_thresholded = (past_speed > high_speed) * (current_speed > medium_speed).astype(np.uint16)

        # additional thresholds
        stimulus_thresholded = ( (absolute_speeds > .25) * (distance_from_shelter > 60) * (frame_nums >= stim_frame) ).astype(np.uint16) #(current_speed < 0) +
        angle_thresholded = ( within_subgoal_bound + abs(shelter_angles) < 60).astype(np.uint16)
        distance_thresholded = (distance_from_shelter > 80).astype(np.uint16)

        # make sure within a certain distance of shelter at the end of the bout
        minimum_distance = 400
        close_enough_to_shelter = np.concatenate((np.convolve(distance_from_shelter < minimum_distance, -1*filter_sequence_60, mode='valid'), np.zeros(len(filter_sequence_60) - 1)))

        # combine thresholds into one
        thresholds_passed_first_pass = ( distance_thresholded * angle_thresholded * close_enough_to_shelter * (current_thresholded + future_thresholded + far_future_thresholded + past_thresholded + stimulus_thresholded)) > 0

        # and smooth it *out*
        smooth_length = 30
        thresholds_passed_first_pass = np.concatenate((np.zeros(int(smooth_length/2) - 1), np.convolve(thresholds_passed_first_pass, np.ones(smooth_length), mode='valid'), np.zeros(int(smooth_length/2)))) > 0

        # finally, add a minimum duration threshold
        minimum_distance_traveled = -.5
        minimum_distance_traveled = minimum_distance_traveled - .25 * minimum_distance_traveled * speed_setting
        minimum_duration = 60 + smooth_length
        minimum_duration = minimum_duration - .5 * minimum_duration * speed_setting
        groups = []; idx = 0
        for k, g in itertools.groupby(thresholds_passed_first_pass*(1 - thresholds_passed)):
            groups.append(list(g))  # Store group iterator as a list
            group_length = len(groups[len(groups)-1]); idx += group_length
            distance_traveled = (distance_from_shelter[idx-1] - distance_from_shelter[idx - group_length]) / distance_from_shelter[idx - group_length]
            if k and (group_length < minimum_duration or (distance_traveled > minimum_distance_traveled)):
                thresholds_passed_first_pass[idx - group_length: idx] = False

        # get the index when adequately long locomotor bouts pass the threshold
        thresholds_passed = thresholds_passed_first_pass * (1 - thresholds_passed)
        thresholds_passed_idx = np.where(thresholds_passed)[0]

        # loop over each frame that passed the threshold
        for i, frame_num in enumerate(frame_nums[thresholds_passed_idx]):

            # extract DLC coordinates from the saved coordinates dictionary
            body_angle = coordinates['body_angle'][frame_num]
            shoulder_angle = coordinates['shoulder_angle'][frame_num]
            shoulder_location = tuple(coordinates['shoulder_location'][:, frame_num].astype(np.uint16))
            body_location = tuple(coordinates['center_body_location'][:, frame_num].astype(np.uint16))

            # set scale for size of model mouse
            back_butt_dist = 15 + speed_setting * 2

            # draw ellipses representing model mouse
            model_mouse_mask = cv2.ellipse(model_mouse_mask_initial.copy(), body_location, (int(back_butt_dist * .7), int(back_butt_dist * .35)), 180 - body_angle, 0, 360, 100, thickness=-1)
            model_mouse_mask = cv2.ellipse(model_mouse_mask , shoulder_location, (int(back_butt_dist), int(back_butt_dist*.2)), 180 - shoulder_angle ,0, 360, 100, thickness=-1)

            # angular speed
            current_angular_speed = angular_speed[frame_num - (previous_stim_frame + skip_frames)]
            current_speed_toward_shelter = current_speeds[frame_num - (previous_stim_frame + skip_frames)]

            # determine color by angular speed
            if current_angular_speed > 2.5 or current_speed_toward_shelter > -medium_speed:
                if not speed_setting: #frame_num < stim_frame and
                    speed_color = np.array([150, 220, 220])  # yellow
                    multiplier = 85
                else:
                    speed_color = np.array([220, 230, 205])  # blue
                    multiplier = 25
            else:
                if not speed_setting: #frame_num < stim_frame and
                    speed_color = np.array([120, 180, 230])  # orange
                    multiplier = 65
                else:
                    speed_color = np.array([152, 230, 152])  # green
                    multiplier = 25
            if frame_num >= stim_frame - 30:
                speed_color = np.array([100, 100, 100])

            # make this trial's stimulus response more prominent in the saved version
            if frame_num >= stim_frame and speed_setting:
                save_multiplier = multiplier / 4
            else:
                save_multiplier = multiplier

            # make lighter if there's overlap with previous homings
            if frame_num < stim_frame and np.mean(exploration_arena[model_mouse_mask.astype(bool)]) < 220:
                multiplier += 80
                save_multiplier += 80
            if frame_num > stim_frame and np.mean(exploration_arena[model_mouse_mask.astype(bool)]) < 200:
                continue

            # on the stimulus onset, get the contours of the mouse body
            if frame_num >= stim_frame and not stimulus_contour_retrieved:
                _, contours, _ = cv2.findContours(model_mouse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                save_exploration_arena = exploration_arena.copy()
                stimulus_contour_retrieved = True

            # create color multiplier to modify image
            color_multiplier = 1 - (1 - speed_color / [255, 255, 255]) / (np.mean(1 - speed_color / [255, 255, 255]) * multiplier)
            save_color_multiplier = 1 - (1 - speed_color / [255, 255, 255]) / (np.mean(1 - speed_color / [255, 255, 255]) * save_multiplier)

            # apply color to arena image
            exploration_arena[model_mouse_mask.astype(bool)] = exploration_arena[model_mouse_mask.astype(bool)] * color_multiplier

            # apply color to this trial's image
            if frame_num >= stim_frame or not speed_setting:
                save_exploration_arena[model_mouse_mask.astype(bool)] = save_exploration_arena[model_mouse_mask.astype(bool)] * save_color_multiplier

            # display image
            cv2.imshow(savepath +'homings', exploration_arena)
            # cv2.imshow(savepath + 'homings2', save_exploration_arena)

            # press q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # apply the contours and border to the image and save the image
    try:
        # save_exploration_arena[ cv2.cvtColor(save_exploration_arena, cv2.COLOR_BGR2GRAY) < 100 ] = save_exploration_arena[ cv2.cvtColor(save_exploration_arena, cv2.COLOR_BGR2GRAY) < 100 ] + 80
        save_exploration_arena = cv2.drawContours(save_exploration_arena, contours, 0, (150, 150, 150), -1) #(90, 0, 200), -1) # 164, 181, 124
        save_exploration_arena = cv2.drawContours(save_exploration_arena, contours, 0, (0,0,0), 2) #(90, 0, 200), 1)
        session_trials_plot_background[border_size:, 0:-border_size] = save_exploration_arena
        scipy.misc.imsave(os.path.join(savepath, videoname + '_spont_homings.tif'), cv2.cvtColor(session_trials_plot_background, cv2.COLOR_BGR2RGB))
    except:
        print('repeat stimulus trial')



    return exploration_arena


def planning(exploration_arena_copy, session_trials_plot_background, border_size, coordinates, previous_stim_frame, stim_frame, videoname, savepath, trial_type, obstacle_type, subgoal_locations):
    '''
    compute and display EXPLORATION
    go through each frame, adding the mouse silhouette
    '''

    # for debugging, make a copy
    exploration_arena = np.ones(exploration_arena_copy.shape, np.uint8)*255

    # set what this mouse considers a high speed
    speeds = [.5, 1, 2, 4]

    # initialize thresholds passed array
    thresholds_passed = np.zeros(stim_frame)
    stimulus_contour_retrieved = False

    # instead of speed to shelter, take max of speed to shelter and speed to subgoals
    coordinates['speed_toward_subgoal'] = np.zeros((len(subgoal_locations['sub-goals'])+1, stim_frame ))
    coordinates['speed_toward_subgoal'][0,:] = coordinates['speed_toward_shelter'][:stim_frame]

    # instead of speed to shelter, take max of speed to shelter and speed to subgoals
    skip_frames = 300
    coordinates['speed_toward_subgoal'] = np.zeros((len(subgoal_locations['sub-goals'])+1, stim_frame))
    coordinates['speed_toward_subgoal'][0,:] = coordinates['speed_toward_shelter'][:stim_frame]
    coordinates['center_location'][0][coordinates['center_location'][0] >= exploration_arena.shape[1]] = exploration_arena.shape[1] - 1
    coordinates['center_location'][1][coordinates['center_location'][1] >= exploration_arena.shape[0]] = exploration_arena.shape[0] - 1
    x_location = coordinates['center_location'][0][:stim_frame].astype(np.uint16)
    y_location = coordinates['center_location'][1][:stim_frame].astype(np.uint16)
    subgoal_bound = [ (int(x * exploration_arena.shape[1] / 1000), int(y* exploration_arena.shape[0] / 1000)) for x, y in subgoal_locations['region'] ]

    # compute distance from subgoal
    polygon_mask = np.zeros(exploration_arena.shape[0:2])
    cv2.drawContours(polygon_mask, [np.array(subgoal_bound)], 0, 100, -1)
    polygon_mask = polygon_mask.astype(bool)

    for i, sg in enumerate(subgoal_locations['sub-goals']):
        # calculate distance to subgoal
        distance_from_subgoal = np.sqrt((x_location - sg[0] * exploration_arena.shape[1] / 1000) ** 2 +
                                                   (y_location - sg[1] * exploration_arena.shape[0] / 1000) ** 2)
        # compute valid locations to go to subgoal
        within_subgoal_bound = []
        for x, y in zip(x_location, y_location):
            # within_subgoal_bound.append(point_in_polygon([x, y],subgoal_bound))
            within_subgoal_bound.append(polygon_mask[y, x] )

        # compute speed w.r.t. subgoal
        coordinates['speed_toward_subgoal'][i+1, :] = np.concatenate( ([0], np.diff(distance_from_subgoal))) * within_subgoal_bound

    coordinates['speed_toward_subgoal'] = np.min(coordinates['speed_toward_subgoal'], 0)



    for speed_setting in [True, False]:

        # use higher speed setting to make faster homings darker
        high_speed = speeds[2 + int(speed_setting)]
        medium_speed = speeds[1 + int(speed_setting)]
        low_speed = speeds[0]

        # initialize the mouse mask
        model_mouse_mask_initial = exploration_arena[:,:,0] * 0

        # get the coordinates up to and including this trial
        skip_frames = 300
        # current_speeds = coordinates['speed_toward_shelter'][:stim_frame]
        current_speeds = coordinates['speed_toward_subgoal']
        absolute_speeds = coordinates['speed'][:stim_frame]
        shelter_angles = coordinates['shelter_angle'][:stim_frame]
        angular_speed = abs(np.concatenate( ([0], np.diff(shelter_angles) ) ) )
        angular_speed[angular_speed > 180] = 360 - angular_speed[angular_speed > 180]
        distance_from_shelter = coordinates['distance_from_shelter'][:stim_frame]

        # get the frame numbers to analyze
        frame_nums = np.arange(stim_frame)

        # do convolutions to get current, future, far future, and past speeds w.r.t the shelter
        filter_sequence_24 = -1 * np.ones(24)
        filter_sequence_30 = -1 * np.ones(30)
        filter_sequence_45 = -1 * np.ones(45)
        filter_sequence_60 = -1 * np.ones(60)

        past_speed = np.concatenate((np.zeros(len(filter_sequence_45) - 1),np.convolve(current_speeds, filter_sequence_45, mode='valid'))) / len(filter_sequence_45)
        future_speed = np.concatenate((np.convolve(current_speeds, filter_sequence_60, mode='valid'), np.zeros(len(filter_sequence_60) - 1))) / len(filter_sequence_60)
        far_future_speed = np.concatenate((np.convolve(current_speeds, filter_sequence_60, mode='valid'), np.zeros(len(filter_sequence_60) - 1 + 20))) / len(filter_sequence_60)
        far_future_speed = far_future_speed[20:]
        current_speed = np.concatenate((np.zeros(12 - 1), np.convolve(current_speeds, filter_sequence_24, mode='valid'), np.zeros(12))) / len(filter_sequence_24)

        # threshold the convolved speeds to determine which frames to draw
        current_thresholded = (current_speed > high_speed).astype(np.uint16)
        future_thresholded = (future_speed > high_speed) * (current_speed > medium_speed).astype(np.uint16)
        far_future_thresholded = (far_future_speed > high_speed) * (current_speed > medium_speed).astype(np.uint16)
        past_thresholded = (past_speed > high_speed) * (current_speed > medium_speed).astype(np.uint16)

        # additional thresholds
        # stimulus_thresholded = ( (absolute_speeds > .25) * (distance_from_shelter > 60) * (frame_nums >= stim_frame) ).astype(np.uint16) #(current_speed < 0) +
        angle_thresholded = ( abs(shelter_angles) < 90).astype(np.uint16)
        distance_thresholded = (distance_from_shelter > 80).astype(np.uint16)

        # make sure within a certain distance of shelter at the end of the bout
        minimum_distance = 200
        close_enough_to_shelter = np.concatenate((np.convolve(distance_from_shelter < minimum_distance, -1*filter_sequence_60, mode='valid'), np.zeros(len(filter_sequence_60) - 1)))

        # combine thresholds into one
        thresholds_passed_first_pass = ( distance_thresholded * angle_thresholded * close_enough_to_shelter *
                                         (current_thresholded + future_thresholded + far_future_thresholded + past_thresholded)) > 0

        # and smooth it *out*
        smooth_length = 30
        thresholds_passed_first_pass = np.concatenate((np.zeros(int(smooth_length/2) - 1), np.convolve(thresholds_passed_first_pass, np.ones(smooth_length), mode='valid'), np.zeros(int(smooth_length/2)))) > 0

        # make sure not moving away from shelter during bout
        present_speed = np.concatenate((np.zeros(6 - 1), np.convolve(current_speeds, -1 * np.ones(12), mode='valid'), np.zeros(6)))
        thresholds_passed_first_pass = thresholds_passed_first_pass * (present_speed > 0)

        # make sure not making a turn during bout
        present_angular_speed = abs( np.concatenate((np.zeros(5 - 1), np.convolve(angular_speed, -1 * np.ones(10), mode='valid'), np.zeros(5))) )
        thresholds_passed_first_pass = thresholds_passed_first_pass * (present_angular_speed < 45)

        # finally, add a minimum duration threshold
        minimum_distance_traveled = -.25
        minimum_duration = smooth_length + 10
        minimum_duration = minimum_duration - .5 * minimum_duration * speed_setting
        groups = []; idx = 0; start_index = []; speed = []

        # loop across each group of frames
        for k, g in itertools.groupby(thresholds_passed_first_pass*(1 - thresholds_passed)):
            # Store group iterator as a list
            groups.append(list(g))
            # get the number of frames in the group
            group_length = len(groups[len(groups)-1]); idx += group_length
            # get the distance traveled
            distance_traveled = (distance_from_shelter[idx-1] - distance_from_shelter[idx - group_length]) / distance_from_shelter[idx - group_length]
            # if not enough distance traveled or not end at shelter, get rid of it
            if k and (distance_from_shelter[idx-1] > 150 or (group_length < minimum_duration or (not speed_setting and distance_traveled > minimum_distance_traveled))):
                thresholds_passed_first_pass[idx - group_length: idx] = False
            elif k:
                start_index = start_index + [idx - group_length]
                speed.append( abs((distance_from_shelter[idx-1] - distance_from_shelter[idx - group_length]) / (group_length/30) ) )

        # get the index when adequately long locomotor bouts pass the threshold
        thresholds_passed = thresholds_passed_first_pass*(1 - thresholds_passed)
        thresholds_passed_idx = np.where(thresholds_passed)[0]

        # loop over each frame that passed the threshold
        for i, frame_num in enumerate(frame_nums[start_index]): #enumerate(frame_nums[thresholds_passed_idx]):

            start_frame_num = frame_num
            start_idx = i
            previous_mouse_mask = model_mouse_mask_initial.copy()

            for b in range(-20, 10): #[0]: #
                frame_num = start_frame_num + b;
                i = start_idx + b

                # extract DLC coordinates from the saved coordinates dictionary
                body_angle = coordinates['body_angle'][frame_num]
                body_location = tuple(coordinates['center_body_location'][:, frame_num].astype(np.uint16))

                # set scale for size of model mouse
                back_butt_dist = 30 #+ speed_setting * 5

                # draw ellipses representing model mouse
                model_mouse_mask = cv2.ellipse(model_mouse_mask_initial.copy(), body_location, (int(back_butt_dist), int(back_butt_dist*.6)), 180 - body_angle, 0, 360, 1, thickness=-1)
                if np.sum(model_mouse_mask * previous_mouse_mask) > (np.pi * back_butt_dist**2 /2):
                    continue
                else:
                    previous_mouse_mask = model_mouse_mask

                # determine color
                speed_color = np.array([150, 255, 100])*.8  # green

                # determine darkness by speed
                multiplier = 50000 / speed[start_idx]**2
                # multiplier = 6
                if multiplier < 2.5:
                    multiplier = 2.5
                elif multiplier > 20:
                    multiplier = 20
                # print(multiplier)

                # create color multiplier to modify image
                color_multiplier = 1 - (1 - speed_color / [255, 255, 255]) / (np.mean(1 - speed_color / [255, 255, 255]) * multiplier)

                # apply color to arena image
                exploration_arena[model_mouse_mask.astype(bool)] = exploration_arena[model_mouse_mask.astype(bool)] * color_multiplier

                # display image
                cv2.imshow(savepath +' planning', exploration_arena)

                # press q to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    # blur the image
    exploration_arena_blur = cv2.GaussianBlur(exploration_arena, ksize=(25, 25), sigmaX=6, sigmaY=6)

    # draw arena
    arena, _, _ = model_arena(exploration_arena_copy.shape[0:2], trial_type, False, obstacle_type)
    arena_color = cv2.cvtColor(arena, cv2.COLOR_GRAY2RGB)
    exploration_arena_blur[arena < 255] = arena_color[arena < 255]
    exploration_arena_blur = (exploration_arena_blur).astype(np.uint8)

    # show and save image
    cv2.imshow(savepath + ' planning', exploration_arena_blur); cv2.waitKey(1)
    session_trials_plot_background[border_size:, 0:-border_size] = exploration_arena_blur
    scipy.misc.imsave(os.path.join(savepath, videoname + '_planning_utility.tif'), cv2.cvtColor(session_trials_plot_background, cv2.COLOR_BGR2RGB))




    return exploration_arena


def goalness():
    '''
    compute and display GOALNESS DURING EXPLORATION
    go through each frame, adding the mouse silhouette
    '''

    scale = int(frame.shape[0] /10)
    goal_arena, _, _ = model_arena(frame.shape[0:2], trial_type, False, False, obstacle_type)
    speed_map = np.zeros((scale, scale))
    occ_map = np.zeros((scale, scale))

    goal_map = np.zeros((scale, scale))

    # stim_erase_idx = np.arange(len(coordinates['center_location'][0][:stim_frame]))
    # stim_erase_idx = [np.min(abs(x - stims)) for x in stim_erase_idx]
    # stim_erase_idx = [x > 300 for x in stim_erase_idx]

    # filter_sequence = np.concatenate( (np.ones(15)*-np.percentile(coordinates['speed'],99.5), np.zeros(10)) )
    filter_sequence = np.ones(20) * -np.percentile(coordinates['speed'], 99.5)
    print(colored(' Calculating goalness...', 'green'))
    for x_loc in tqdm(range(occ_map.shape[0])):
        for y_loc in range(occ_map.shape[1]):
            curr_dist = np.sqrt((coordinates['center_location'][0][:stim_frame] - ((720 / scale) * (x_loc + 1 / 2))) ** 2 +
                                (coordinates['center_location'][1][:stim_frame] - ((720 / scale) * (y_loc + 1 / 2))) ** 2)
            occ_map[x_loc, y_loc] = np.mean(curr_dist < (2 * 720 / scale))
            curr_speed = np.concatenate(([0], np.diff(curr_dist))) # * np.array(stim_erase_idx)  # * (coordinates['center_location'][1] < 360) #
            speed_map[x_loc, y_loc] = abs(np.mean(curr_speed < -np.percentile(coordinates['speed'], 99.5)))

            goal_map[x_loc, y_loc] = np.percentile(np.concatenate((np.zeros(len(filter_sequence) - 1),
                                                                   np.convolve(curr_speed, filter_sequence, mode='valid'))) * (curr_dist < 60), 99.8)  # 98

    goal_map_plot = goal_map.T * (occ_map.T > 0)

    goal_image = goal_map_plot.copy()

    goal_image = goal_image * 255 / np.percentile(goal_map_plot, 99)
    goal_threshold = int(np.percentile(goal_map_plot, 90) * 255 / np.percentile(goal_map_plot, 99))

    goal_image[goal_image > 255] = 255
    goal_image = cv2.resize(goal_image.astype(np.uint8), frame.shape[0:2])

    goal_image[goal_image <= int(goal_threshold * 1 / 5) * (goal_image > 1)] = int(goal_threshold * 1 / 10)
    goal_image[(goal_image <= goal_threshold * 2 / 5) * (goal_image > int(goal_threshold * 1 / 5))] = int(goal_threshold * 2 / 10)
    goal_image[(goal_image <= goal_threshold * 3 / 5) * (goal_image > int(goal_threshold * 2 / 5))] = int(goal_threshold * 3 / 10)
    goal_image[(goal_image <= goal_threshold * 4 / 5) * (goal_image > int(goal_threshold * 3 / 5))] = int(goal_threshold * 4 / 10)
    goal_image[(goal_image <= goal_threshold) * (goal_image > int(goal_threshold * 4 / 5))] = int(goal_threshold * 6 / 10)
    goal_image[(goal_image < 255) * (goal_image > goal_threshold)] = int(goal_threshold)

    # goal_image[(arena_fresh[:,:,0] > 0) * (goal_image == 0)] = int(goal_threshold * 1 / 5)
    goal_image[(arena_fresh[:, :, 0] < 100)] = 0

    goal_image = cv2.copyMakeBorder(goal_image, border_size, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
    textsize = cv2.getTextSize(videoname, 0, .55, 1)[0]
    textX = int((width - textsize[0]) / 2)
    cv2.putText(goal_image, videoname, (textX, int(border_size * 3 / 4)), 0, .55, (255, 255, 255), thickness=1)
    scipy.misc.imsave(os.path.join(savepath, videoname + '_goalness.tif'), goal_image)

    cv2.imshow('goal', goal_image)
    cv2.waitKey(1)




#
#
# def planning(savepath, videoname, coordinates, height, width, trial_type, obstacle_type, previous_stim_frame, stim_frame):
#     '''
#     compute and display PLANNESS DURING EXPLORATION
#     go through each frame, adding the mouse silhouette
#     '''
#
#     # get model arena and downsample x10
#     downsample = 20
#     scale = int(height / downsample)
#     radius = np.sqrt(2*downsample**2)
#
#     # determine when the mouse is in the shelter
#     in_shelter = coordinates['distance_from_shelter'][:stim_frame] < 100
#     x_location = coordinates['center_location'][0][:stim_frame]
#     y_location = coordinates['center_location'][1][:stim_frame]
#
#     # determine when the mouse isn't moving toward the shelter
#     speed = coordinates['speed_toward_shelter'][:stim_frame]
#     # high_speed = np.percentile(speed, 99)*2
#     slow_speed = abs(speed) < .5
#
#     # determine when the mouse is moving toward the shelter
#     filter = np.ones(15)
#     future_speed = np.concatenate((np.convolve(speed, filter, mode='valid'), np.zeros(len(filter) - 1))) / len(filter)
#     will_move_toward_shelter = future_speed < -.5
#
#
#     # determine how long from each frame to the next frame in the shelter
#     time_to_shelter = []; groups = []
#     for k, g in itertools.groupby(in_shelter):
#         groups.append(list(g))
#         group_length = len(groups[len(groups) - 1]);
#         # if in the shelter, set value to zero
#         if k:
#             time_to_shelter = time_to_shelter + [255 for x in range(group_length)]
#         # if not in shelter, set value to timesteps to shelter
#         else:
#             current_time_to_shelter = [x for x in range(1, group_length+1)]
#             time_to_shelter = time_to_shelter + current_time_to_shelter[::-1]
#     time_to_shelter = np.array(time_to_shelter)
#
#     # initialize planning maps
#     plan_map = np.zeros((scale, scale))
#
#     # loop over each location on a grid
#     for x_loc in tqdm(range(plan_map.shape[0])):
#         for y_loc in range(plan_map.shape[1]):
#             # get indices when mouse is in current square
#             within_square = ( abs(x_location - ((width  / scale) * (x_loc + 1 / 2)) ) <= downsample/2 ) * \
#                             ( abs(y_location - ((height / scale) * (y_loc + 1 / 2))) <= downsample/2 )
#
#             # if there is any occupancy (during slow movement and future movement toward shelter)
#             if np.sum(within_square * slow_speed * will_move_toward_shelter):
#                 plan_map[y_loc, x_loc] = np.percentile(time_to_shelter[within_square * slow_speed * will_move_toward_shelter], 1) #np.min(time_to_shelter[within_square * slow_speed])
#             else:
#                 plan_map[y_loc, x_loc] = 255
#
#     # copy the plan map for plotting and reformate to be uint8
#     plan_map_plot = plan_map.copy()
#     plan_map_plot[plan_map_plot>=255] = 255
#     plan_map_plot = 255 - plan_map_plot.astype(np.uint8)
#
#     # upsample plot and then apply median filter
#     plan_map_plot = cv2.resize(plan_map_plot, (width, height), interpolation=cv2.INTER_CUBIC)
#     plan_map_plot = scipy.signal.medfilt2d(plan_map_plot, kernel_size=71)
#
#     # show plot of plans (white on black)
#     cv2.imshow('plans', plan_map_plot)
#
#     # draw arena
#     arena, _, _ = model_arena((height, width), trial_type, False, obstacle_type)
#     arena_color = cv2.cvtColor(arena, cv2.COLOR_GRAY2RGB)
#     arena_color_copy = arena_color.copy()
#
#     # convert plan map to color
#     plan_map_color = cv2.cvtColor(plan_map_plot, cv2.COLOR_GRAY2RGB)
#
#     # get the total intensity of pixels on this plot
#     total_plans = np.sum(plan_map_plot)
#
#     # get each blob of pixels
#     _, contours, _ = cv2.findContours(plan_map_plot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # initialize mask
#     blank_image = np.zeros(arena.shape)
#
#     # color in each blob from the plan map onto the arena plot
#     for c in range(len(contours)):
#         # draw contours on the blank mask
#         contour_mask = cv2.drawContours(blank_image.copy(), contours, c, color=(1, 1, 1), thickness=cv2.FILLED)
#
#         # calculate the importance of this particular blob
#         plans_in_curr_contour = np.sum(plan_map_plot[contour_mask.astype(bool)]) / total_plans
#
#         # for planning points accounting for at least 5% of planning utility, draw them in on the arena plot
#         if plans_in_curr_contour > .05:
#             arena_color[contour_mask.astype(bool)] = (255 - plan_map_color[contour_mask.astype(bool)]* [1, plans_in_curr_contour, 1])
#
#     # or color in all blobs at once
#     arena_color[plan_map_plot.astype(bool)] = plan_map_color[plan_map_plot.astype(bool)]
#
#     # draw the shelter and obstacle, in case they were drawn over
#     arena_color[arena<255] = arena_color_copy[arena<255]
#
#     # show the arena
#     cv2.imshow('arena', arena_color)
#     cv2.waitKey(1)
#
#     # save the image
#     scipy.misc.imsave(os.path.join(savepath, videoname + '_planning.tif'), cv2.cvtColor(arena_color, cv2.COLOR_BGR2RGB))






def exploration(exploration_arena_copy, exploration_arena_trial, session_plot_background, border_size, coordinates, previous_stim_frame, stim_frame, videoname, savepath, arena):
    '''
    compute and display EXPLORATION
    go through each frame, adding the mouse silhouette
    '''

    # for debugging, make a copy
    exploration_arena = copy.deepcopy(exploration_arena_copy)
    save_exploration_arena = exploration_arena.copy()
    trial_plot_background = session_plot_background

    # initialize the mouse mask
    model_mouse_mask_initial = exploration_arena[:,:,0] * 0

    # get the coordinates up to and including this trial
    current_speeds = coordinates['speed_toward_shelter'][previous_stim_frame :stim_frame]
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

        # stop once at shelter
        # if frame_num >= stim_frame and current_distance_from_shelter < 60:
        #     break
        if current_distance_from_shelter < 50:
            continue

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
        if current_speed_toward_shelter < 0:
            speed_color = np.array([190, 189, 225])  # red
            multiplier = 35
        else:
            speed_color = np.array([190, 220, 190])  # green
            multiplier = 30

        # create color multiplier to modify image
        color_multiplier = 1 - (1 - speed_color / [255, 255, 255]) / (np.mean(1 - speed_color / [255, 255, 255]) * multiplier)

        # prevent any region from getting too dark (trial)
        if np.mean(exploration_arena_trial[model_mouse_mask.astype(bool)]) < 100: #( 100 - (frame_num >= stim_frame) * 50 ):
            continue

        # apply color to arena image (trial)
        exploration_arena_trial[model_mouse_mask.astype(bool)] = exploration_arena_trial[model_mouse_mask.astype(bool)] * color_multiplier

        # prevent any region from getting too dark (session)
        if np.mean(exploration_arena_trial[model_mouse_mask.astype(bool)]) < 100: #( 100 - (frame_num >= stim_frame) * 50 ):
            continue

        # apply color to arena image (session)
        exploration_arena[model_mouse_mask.astype(bool)] = exploration_arena[model_mouse_mask.astype(bool)] * color_multiplier

        # display image
        cv2.imshow(savepath +'homings', exploration_arena)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # apply the contours and border to the image and save the image
    try:
        session_plot_background[border_size:, 0:-border_size] = exploration_arena
        trial_plot_background[border_size:, 0:-border_size] = exploration_arena_trial

        scipy.misc.imsave(os.path.join(savepath, videoname + '_exploration_all.tif'), cv2.cvtColor(session_plot_background, cv2.COLOR_BGR2RGB))
        scipy.misc.imsave(os.path.join(savepath, videoname + '_exploration_recent.tif'), cv2.cvtColor(trial_plot_background, cv2.COLOR_BGR2RGB))
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