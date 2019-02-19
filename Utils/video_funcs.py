from Utils.imports import *
from Config import track_options, dlc_config_settings

# from deeplabcut.utils import create_labeled_video

import cv2
import numpy as np
import scipy.misc
from termcolor import colored
from tqdm import tqdm
import glob
from Utils.registration_funcs import register_frame, model_arena, invert_fisheye_map
import matplotlib.pyplot as plt



# # =================================================================================
# #              GENERATE PERI-STIMULUS VIDEO CLIPS and FLIGHT IMAGE
# # =================================================================================
# def peri_stimulus_video_clip(vidpath = '', videoname = '', savepath = '', start_frame=0., end_frame=100., stim_frame = 0,
#                    registration = 0, fps=False, save_clip = False, display_clip = False, counter = True, make_flight_image = True):
#     # GET BEAHVIOUR VIDEO - ######################################
#     vid = cv2.VideoCapture(vidpath)
#     if not fps: fps = vid.get(cv2.CAP_PROP_FPS)
#     width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     # SETUP VIDEO CLIP SAVING - ######################################
#     # file_already_exists = os.path.isfile(os.path.join(savepath,videoname+'.avi'))
#     fourcc = cv2.VideoWriter_fourcc(*"XVID")  # LJPG for lossless, XVID for compressed
#     border_size = 20
#     if save_clip:
#         if not os.path.isdir(savepath):
#             os.makedirs(savepath)
#         video_clip = cv2.VideoWriter(os.path.join(savepath,videoname+'.avi'), fourcc, fps, (width+2*border_size*counter, height+2*border_size*counter), counter)
#     vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
#
#     pre_stim_color = [255, 120, 120]
#     post_stim_color = [120, 120, 255]
#
#     if registration[3]: # setup fisheye correction
#         maps = np.load(registration[3])
#         map1 = maps[:, :, 0:2]
#         map2 = maps[:, :, 2] * 0
#     else:
#         print(colored('Fisheye correction unavailable', 'green'))
#     # RUN SAVING AND ANALYSIS OVER EACH FRAME - ######################################
#     while True: #and not file_already_exists:
#         ret, frame = vid.read()  # get the frame
#         if ret:
#             frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES)
#             if [registration]:
#                 # load the fisheye correction
#                 frame_register = frame[:, :, 0]
#                 if registration[3]:
#                     frame_register = cv2.copyMakeBorder(frame_register, y_offset, int((map1.shape[0] - frame.shape[0]) - y_offset),
#                                                         x_offset, int((map1.shape[1] - frame.shape[1]) - x_offset), cv2.BORDER_CONSTANT, value=0)
#                     frame_register = cv2.remap(frame_register, map1, map2, interpolation=cv2.INTER_LINEAR,
#                                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
#                     frame_register = frame_register[y_offset:-int((map1.shape[0] - frame.shape[0]) - y_offset),
#                                      x_offset:-int((map1.shape[1] - frame.shape[1]) - x_offset)]
#                 frame = cv2.cvtColor(cv2.warpAffine(frame_register, registration[0], frame.shape[0:2]),cv2.COLOR_GRAY2RGB)
#
#             # MAKE ESCAPE TRAJECTORY IMAGE - #######################################
#             if make_flight_image:
#                 # at stimulus onset, take this frame to lay all the superimposed mice on top of
#                 if frame_num == stim_frame:
#                     flight_image_by_distance = frame[:,:,0].copy()
#
#                 # in subsequent frames, see if frame is different enough from previous image to merit joining the image
#                 elif frame_num > stim_frame and (frame_num - stim_frame) < 30*10:
#                     # get the number of pixels that are darker than the flight image
#                     difference_from_previous_image = ((frame[:,:,0]+.001) / (flight_image_by_distance+.001))<.55 #.5 original parameter
#                     number_of_darker_pixels = np.sum(difference_from_previous_image)
#
#                     # if that number is high enough, add mouse to image
#                     if number_of_darker_pixels > 1050: # 850 original parameter
#                         # add mouse where pixels are darker
#                         flight_image_by_distance[difference_from_previous_image] = frame[difference_from_previous_image,0]
#
#             # SHOW BOUNDARY AND TIME COUNTER - #######################################
#             if counter and (display_clip or save_clip):
#                 # cv2.rectangle(frame, (0, height), (150, height - 60), (150,150,150), -1)
#                 if frame_num < stim_frame:
#                     cur_color = tuple([x * ((frame_num - start_frame) / (stim_frame - start_frame)) for x in pre_stim_color])
#                     sign = ''
#                 else:
#                     cur_color = tuple([x * (1 - (frame_num - stim_frame) / (end_frame-stim_frame))  for x in post_stim_color])
#                     sign = '+'
#
#                 # border and colored rectangle around frame
#                 frame = cv2.copyMakeBorder(frame, border_size, border_size, border_size, border_size,cv2.BORDER_CONSTANT, value=cur_color)
#
#                 # report video details
#                 cv2.putText(frame, videoname, (20, 40), 0, .55, (180, 180, 180), thickness=1)
#
#                 # report time relative to stimulus onset
#                 frame_time = (frame_num - stim_frame) / fps
#                 frame_time = str(round(.1*round(frame_time/.1), 1))+ '0'*(abs(frame_time)<10)
#                 cv2.putText(frame, sign + str(frame_time) + 's', (width-110, height+10), 0, 1,(180,180,180), thickness=2)
#
#             else:
#                 frame = frame[:,:,0] # or use 2D grayscale image instead
#
#             # SHOW AND SAVE FRAME - #######################################
#             if display_clip:
#                 cv2.imshow('Trial Clip', frame)
#             if save_clip:
#                 video_clip.write(frame)
#             if display_clip:
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#             if frame_num >= end_frame:
#                 break
#         else:
#             print('Problem with movie playback')
#             cv2.waitKey(1000)
#             break
#
#     # wrap up
#     vid.release()
#     if make_flight_image:
#         flight_image_by_distance = cv2.copyMakeBorder(flight_image_by_distance, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0)
#         cv2.putText(flight_image_by_distance, videoname, (border_size, border_size-5), 0, .55, (180, 180, 180), thickness=1)
#         cv2.imshow('Flight image', flight_image_by_distance)
#         cv2.waitKey(10)
#         scipy.misc.imsave(os.path.join(savepath, videoname + '.tif'), flight_image_by_distance)
#     if save_clip:
#         video_clip.release()
#     # cv2.destroyAllWindows()
#


# =================================================================================
#        GENERATE PERI-STIMULUS VIDEO CLIPS and FLIGHT IMAGE ***with wall***
# =================================================================================
def peri_stimulus_video_clip(vidpath = '', videoname = '', savepath = '', start_frame=0., end_frame=100., stim_frame = 0,
                   registration = 0, x_offset = 300, y_offset = 100, fps=False, analyze_wall = True, save_clip = False, display_clip = False, counter = True):
    # GET BEAHVIOUR VIDEO - ######################################
    vid = cv2.VideoCapture(vidpath)
    if not fps: fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # SETUP VIDEO CLIP SAVING - ######################################
    # file_already_exists = os.path.isfile(os.path.join(savepath,videoname+'.avi'))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # LJPG for lossless, XVID for compressed
    border_size = 20
    if save_clip:
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        video_clip = cv2.VideoWriter(os.path.join(savepath,videoname+'.avi'), fourcc, fps, (width+2*border_size*counter, height+2*border_size*counter), counter)

    # set of border colors
    pre_stim_color = [0, 0, 0, ]
    post_stim_color = [200, 200, 200]

    # setup fisheye correction
    if registration:
        maps = np.load(registration[3])
        map1 = maps[:, :, 0:2]
        map2 = maps[:, :, 2] * 0


    # RUN SAVING AND ANALYSIS OVER EACH FRAME - ######################################
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if display_clip:
        cv2.namedWindow('Trial Clip')
        cv2.moveWindow('Trial Clip',100,100)
    while True: # and analyze: #and not file_already_exists:
        ret, frame = vid.read()  # get the frame
        if ret:
            frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES)
            if [registration]:
                # load the fisheye correction
                frame_register = frame[:, :, 0]
                if registration:
                    frame_register = cv2.copyMakeBorder(frame_register, y_offset, int((map1.shape[0] - frame.shape[0]) - y_offset),
                                                        x_offset, int((map1.shape[1] - frame.shape[1]) - x_offset), cv2.BORDER_CONSTANT, value=0)
                    frame_register = cv2.remap(frame_register, map1, map2, interpolation=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    frame_register = frame_register[y_offset:-int((map1.shape[0] - frame.shape[0]) - y_offset),
                                     x_offset:-int((map1.shape[1] - frame.shape[1]) - x_offset)]
                    frame = cv2.cvtColor(cv2.warpAffine(frame_register, registration[0], frame.shape[0:2]),cv2.COLOR_GRAY2RGB)


            # SHOW BOUNDARY AND TIME COUNTER - #######################################
            if counter and (display_clip or save_clip or wall_mouse_show):
                # cv2.rectangle(frame, (0, height), (150, height - 60), (150,150,150), -1)
                if frame_num < stim_frame:
                    cur_color = tuple([x * ((frame_num - start_frame) / (stim_frame - start_frame)) for x in pre_stim_color])
                    sign = ''
                else:
                    cur_color = tuple([x * (1 - (frame_num - stim_frame) / (end_frame-stim_frame))  for x in post_stim_color])
                    sign = '+'

                # border and colored rectangle around frame
                frame = cv2.copyMakeBorder(frame, border_size, border_size, border_size, border_size,cv2.BORDER_CONSTANT, value=cur_color)

                # report video details
                cv2.putText(frame, videoname, (20, 40), 0, .55, (180, 180, 180), thickness=1)

                # report time relative to stimulus onset
                frame_time = (frame_num - stim_frame) / fps
                frame_time = str(round(.1*round(frame_time/.1), 1))+ '0'*(abs(float(frame_time))<10)
                cv2.putText(frame, sign + str(frame_time) + 's', (width-110, height+10), 0, 1,(180,180,180), thickness=2)

            else:
                frame = frame[:,:,0] # or use 2D grayscale image instead

            # SHOW AND SAVE FRAME - #######################################
            if display_clip:
                cv2.imshow('Trial Clip', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if save_clip:
                video_clip.write(frame)
            if frame_num >= end_frame:
                finished_clip = True
                break
        else:
            print('Problem with movie playback')
            cv2.waitKey(1000)
            break

    # wrap up
    vid.release()
    if save_clip:
        video_clip.release()
    # cv2.destroyAllWindows()




# =========================================================================================
#        GENERATE PERI-STIMULUS VIDEO CLIPS and FLIGHT IMAGE ***with wall, using DLC***
# =========================================================================================
def peri_stimulus_analysis(coordinates, vidpath = '', videoname = '', savepath = '', session_video = None, previous_stim_frame = 0, x_offset = 300, y_offset = 100, obstacle_type = 'wall',
                           session_trials_video = None, session_trials_plot_workspace = None, session_trials_plot_background = None,
                           exploration_arena_in_cum = None, number_of_trials = 10, stims = [], trial_num = 0, start_frame=0., end_frame=100., stim_frame = 0,
                           registration = 0, fps=False, border_size = 40, save_clip = False, analyze_wall=True, display_clip = False, counter = True, make_flight_image = True):

    # GET BEHAVIOUR VIDEO - ######################################
    print('')
    vid = cv2.VideoCapture(vidpath)
    if not fps: fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # SET UP VIDEO CLIP SAVING - ####################################
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # LJPG for lossless, XVID for compressed

    if save_clip:
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        video_clip = cv2.VideoWriter(os.path.join(savepath, videoname + '.avi'), fourcc,
                                     fps,(width + 2 * border_size * counter, height + 2 * border_size * counter), counter)
        video_clip_dlc = cv2.VideoWriter(os.path.join(savepath, videoname + '_dlc.avi'), fourcc,
                                     fps, (width + border_size, height + border_size), True)

    # set up dlc
    body_parts = dlc_config_settings['body parts']

    # generate model arena and wall ROIs
    if analyze_wall:
        wall_height_timecourse, trial_type, x_wall_up_ROI_left, x_wall_up_ROI_right, y_wall_up_ROI, wall_darkness, wall_mouse_already_shown, map1, map2 = \
            initialize_wall_analysis(analyze_wall, stim_frame,  start_frame, end_frame, registration, x_offset, y_offset, vid, width, height)
    else:
        if registration[3]:
            maps = np.load(registration[3])
            map1 = maps[:, :, 0:2]
            map2 = maps[:, :, 2] * 0
        else:
            print(colored('Fisheye correction unavailable', 'green'))
            trial_type = 0
            wall_height_timecourse = None

    wall_mouse_show = False

    # set up border colors
    pre_stim_color = [0, 0, 0,]
    post_stim_color = [200, 200, 200]

    # set up flight path colors
    wall_color = np.array([222, 122, 122])
    probe_color = np.array([200, 200, 200])
    no_wall_color = np.array([122, 122, 222])

    middle_trial = np.ceil(number_of_trials / 2) - 1

    if abs(trial_type) == 1:
        current_flight_color = probe_color
    elif trial_type == 2:
        current_flight_color = wall_color
    else:
        current_flight_color = no_wall_color

    current_flight_color_light = 1 - ( 1 - current_flight_color / [255, 255, 255] ) / ( np.mean( 1 - current_flight_color / [255, 255, 255] ) / .08)
    current_flight_color_dark = 1 - (1 - current_flight_color / [255, 255, 255]) / ( np.mean(1 - current_flight_color / [255, 255, 255]) / .38)



    # set up model arena
    ret, frame = vid.read()
    arena, _, shelter_roi = model_arena(frame.shape[0:2], trial_type*(trial_type-1), False, False, obstacle_type)
    flight_image = arena.copy()
    model_flight_image = cv2.cvtColor(arena.copy(), cv2.COLOR_GRAY2RGB)

    model_mouse_mask_previous = np.zeros(arena.shape).astype(np.uint8)
    model_mouse_mask_initial = np.zeros(arena.shape).astype(np.uint8)

    arena_fresh = cv2.cvtColor(arena.copy(), cv2.COLOR_GRAY2RGB)

    model_flight_image_iters = 0
    if abs(trial_type) > 1:
        trial_type = 0

    # set up session trials plot
    if session_trials_plot_workspace is None:
        session_trials_plot = cv2.cvtColor(arena.copy(), cv2.COLOR_GRAY2RGB)
        session_trials_plot_workspace = session_trials_plot.copy()
        exploration_arena_in_cum = arena_fresh.copy()
    else:
        session_trials_plot = ((session_trials_plot_workspace.astype(float) * 1 + arena_fresh.copy().astype(float) * 4) / 5).astype(np.uint8)

    initial_session_trials_plot = session_trials_plot.copy()
    initial_session_trials_plot_workspace = session_trials_plot_workspace.copy()
    session_trials_plot = cv2.cvtColor(session_trials_plot, cv2.COLOR_BGR2RGB)
    session_trials_plot_background = cv2.cvtColor(session_trials_plot_background, cv2.COLOR_BGR2RGB)
    model_flight_background = session_trials_plot_background.copy()

    previous_center_location = np.nan
    previous_body_angle = np.nan
    previous_head_angle = np.nan

    # CALCULATE COORDINATES
    # for debugging - extract all coordinates
    # array of all body parts, axis x body part x frame
    all_body_parts = np.zeros((2, len(body_parts), coordinates['nose'].shape[1]))
    for i, body_part in enumerate(body_parts):
        # put all together
        all_body_parts[:, i, :] = coordinates[body_part][0:2]
    coordinates['head_location'] = np.nanmean(all_body_parts[:, 0:5, :], axis=1)
    coordinates['snout_location'] = np.nanmean(all_body_parts[:, 0:3, :], axis=1)
    coordinates['neck_location'] = np.nanmean(all_body_parts[:, 3:6, :], axis=1)
    coordinates['butt_location'] = np.nanmean(all_body_parts[:, 9:, :], axis=1)
    coordinates['back_location'] = np.nanmean(all_body_parts[:, 6:9, :], axis=1)
    coordinates['nack_location'] = np.nanmean(all_body_parts[:, 3:9, :], axis=1)
    coordinates['front_location'] = np.nanmean(all_body_parts[:, 0:9, :], axis=1)
    coordinates['center_body_location'] = np.nanmean(all_body_parts[:, 6:, :], axis=1)
    coordinates['center_location'] = np.nanmean(all_body_parts[:, :, :], axis=1)

    delta_position = np.concatenate( ( np.zeros((2,1)), np.diff(coordinates['center_location']) ) , axis = 1)
    coordinates['speed'] = np.sqrt(delta_position[0,:]**2 + delta_position[1,:]**2)

    coordinates['distance_from_shelter'] = np.sqrt((coordinates['center_location'][0] - 500 * 720 / 1000) ** 2 +
                                                   (coordinates['center_location'][1] - 885 * 720 / 1000) ** 2)

    coordinates['speed_toward_shelter'] = np.concatenate( ([0], np.diff(coordinates['distance_from_shelter'])))

    coordinates['speed'] = np.array(pd.Series(coordinates['speed']).interpolate())
    coordinates['distance_from_shelter'] = np.array(pd.Series(coordinates['distance_from_shelter']).interpolate())
    coordinates['speed_toward_shelter'] = np.array(pd.Series(coordinates['speed_toward_shelter']).interpolate())

    locations = ['speed', 'distance_from_shelter', 'speed_toward_shelter', 'head_location', 'butt_location', 'snout_location',
                'neck_location', 'back_location', 'nack_location', 'center_body_location', 'center_location' ]

    for loc_num, loc in enumerate(locations):
        if loc_num < 3:
            coordinates[loc] = np.array(pd.Series(coordinates[loc]).interpolate())
            coordinates[loc] = np.array(pd.Series(coordinates[loc]).fillna(method='bfill'))
            coordinates[loc] = np.array(pd.Series(coordinates[loc]).fillna(method='ffill'))
        else:
            for i in [0,1]:
                coordinates[loc][i] = np.array(pd.Series(coordinates[loc][i]).interpolate())
                coordinates[loc][i] = np.array(pd.Series(coordinates[loc][i]).fillna(method='bfill'))
                coordinates[loc][i] = np.array(pd.Series(coordinates[loc][i]).fillna(method='ffill'))

    coordinates['body_angle'] = np.angle((coordinates['back_location'][0] - coordinates['butt_location'][0]) + (-coordinates['back_location'][1] + coordinates['butt_location'][1]) * 1j, deg=True)
    coordinates['shoulder_angle'] = np.angle((coordinates['head_location'][0] - coordinates['center_body_location'][0]) + (-coordinates['head_location'][1] + coordinates['center_body_location'][1]) * 1j, deg=True)
    head_angle = np.zeros( (len(coordinates['body_angle']) , 2))
    head_angle[:,0] = np.angle((coordinates['snout_location'][0] - coordinates['neck_location'][0]) + (-coordinates['snout_location'][1] + coordinates['neck_location'][1]) * 1j, deg=True)
    head_angle[:,1] = np.angle((coordinates['head_location'][0] - coordinates['nack_location'][0]) + (-coordinates['head_location'][1] + coordinates['nack_location'][1]) * 1j, deg=True)
    coordinates['head_angle'] = np.nanmean(head_angle,1)



    # #####################################
    # compute and display GOALNESS DURING EXPLORATION
    # go through each frame, adding the mouse silhouette
    # ########################################

    scale = int(frame.shape[0]/10)
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

    # #####################################
    # compute and display PLANNESS DURING EXPLORATION
    # go through each frame, adding the mouse silhouette
    # ########################################
    scale = int(frame.shape[0] / 10)
    goal_arena, _, _ = model_arena(frame.shape[0:2], trial_type, False, False, obstacle_type)
    speed_map = np.zeros((scale, scale))
    occ_map = np.zeros((scale, scale))

    plan_map = np.zeros((scale, scale))

    # stim_erase_idx = np.arange(len(coordinates['center_location'][0][:stim_frame]))
    # stim_erase_idx = [np.min(abs(x - stims)) for x in stim_erase_idx]
    # stim_erase_idx = [x > 300 for x in stim_erase_idx]

    # filter_sequence = np.concatenate( (np.ones(15)*-np.percentile(coordinates['speed'],99.5), np.zeros(10)) )
    # stim_frame = 30*60*28

    filter_sequence = np.ones(20) * -np.percentile(coordinates['speed'], 99.5)
    print(colored(' Calculating planness...', 'green'))
    speed_toward_shelter = np.convolve(coordinates['speed_toward_shelter'][:stim_frame], filter_sequence, mode='valid')
    distance_from_shelter = coordinates['distance_from_shelter'][:stim_frame]
    # arrival_in_shelter = coordinates['distance_from_shelter'][:stim_frame] < 100
    # future_arrival_in_shelter = np.concatenate( (arrival_in_shelter[:-30], np.zeros(30) ) )
    for x_loc in tqdm(range(occ_map.shape[0])):
        for y_loc in range(occ_map.shape[1]):
            curr_dist = np.sqrt((coordinates['center_location'][0][:stim_frame] - ((720 / scale) * (x_loc + 1 / 2))) ** 2 +
                                (coordinates['center_location'][1][:stim_frame] - ((720 / scale) * (y_loc + 1 / 2))) ** 2)
            occ_map[x_loc, y_loc] = np.mean(curr_dist < (2 * 720 / scale))
            # curr_speed = np.concatenate(([0], np.diff(curr_dist)))  # * np.array(stim_erase_idx)  # * (coordinates['center_location'][1] < 360) #
            # speed_map[x_loc, y_loc] = abs(np.mean(curr_speed < -np.percentile(coordinates['speed'], 99.5)))

            plan_map[x_loc, y_loc] = np.percentile(np.concatenate((speed_toward_shelter, np.zeros(len(filter_sequence) - 1) )) * (curr_dist < 50) * (distance_from_shelter > 175), 99.2)  # 98

    plan_map_plot = plan_map.T * (occ_map.T > 0)

    plan_image = plan_map_plot.copy()

    plan_image = plan_image * 255 / np.percentile(plan_map_plot, 99.9)
    try:
        plan_threshold = int(np.percentile(plan_map_plot, 99) * 255 / np.percentile(plan_map_plot, 99.9))
    except:
        plan_threshold = 200

    plan_image[plan_image > 255] = 255
    plan_image = cv2.resize(plan_image.astype(np.uint8), frame.shape[0:2])

    plan_image[plan_image <= int(plan_threshold * 1 / 5) * (plan_image > 1)] = int(plan_threshold * 1 / 5)
    plan_image[(plan_image <= plan_threshold * 2 / 5) * (plan_image > int(plan_threshold * 1 / 5))] = int(plan_threshold * 3 / 5)
    plan_image[(plan_image <= plan_threshold * 3 / 5) * (plan_image > int(plan_threshold * 2 / 5))] = int(plan_threshold * 3 / 5)
    plan_image[(plan_image <= plan_threshold * 4 / 5) * (plan_image > int(plan_threshold * 3 / 5))] = int(plan_threshold * 4 / 5)
    plan_image[(plan_image <= plan_threshold) * (plan_image > int(plan_threshold * 4 / 5))] = plan_threshold
    plan_image[(plan_image < 255) * (plan_image > plan_threshold)] = int((plan_threshold + 255) / 2)

    # plan_image[(arena_fresh[:,:,0] > 0) * (plan_image == 0)] = int(plan_threshold * 3 / 10)
    plan_image[(arena_fresh[:, :, 0] < 100)] = 0

    plan_image = cv2.copyMakeBorder(plan_image, border_size, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
    textsize = cv2.getTextSize(videoname, 0, .55, 1)[0]
    textX = int((width - textsize[0]) / 2)
    cv2.putText(plan_image, videoname, (textX, int(border_size * 3 / 4)), 0, .55, (255, 255, 255), thickness=1)
    scipy.misc.imsave(os.path.join(savepath, videoname + '_planness.tif'), plan_image)

    cv2.imshow('plan', plan_image)
    cv2.waitKey(1)

    # #####################################
    # compute and display EXPLORATION
    # go through each frame, adding the mouse silhouette
    # ########################################
    # exploration_arena_in = arena_fresh.copy()
    # slow_color = np.array([155, 245, 245])  # brown-yellow
    # exploration_arena_in_cum = arena_fresh.copy()
    high_speed = np.percentile(coordinates['speed_toward_shelter'], 99)
    # print(high_speed)
    # vid_EXPLORE = cv2.VideoWriter(os.path.join(savepath, videoname + '_dlc_spont_homings.avi'), fourcc,
    #                                  fps, (width, height), True)

    for frame_num in range(previous_stim_frame + 300, stim_frame + 300):

        # speed_toward_shelter = coordinates['speed_toward_shelter'][frame_num - 1]
        speed_toward_shelter = np.median(coordinates['speed_toward_shelter'][frame_num - 1 - 8:frame_num+8])
        speed_toward_shelter_future = np.median(coordinates['speed_toward_shelter'][frame_num - 1:frame_num+30])
        speed_toward_shelter_far_future = np.median(coordinates['speed_toward_shelter'][frame_num + 15:frame_num + 60])
        speed_toward_shelter_past = np.median(coordinates['speed_toward_shelter'][frame_num - 30:frame_num - 1])
        speed = coordinates['speed'][frame_num - 1]

        if (speed_toward_shelter < -5) or (speed_toward_shelter_future < -4.5) or (speed_toward_shelter_far_future < -5 and speed_toward_shelter < 1) or \
                (speed_toward_shelter_past < -4.5) or (frame_num > stim_frame and speed > 3 and speed_toward_shelter < 3) or frame_num == stim_frame:

            multiplier = 1
            if (speed_toward_shelter_far_future < -5):
                # print('far future')
                multiplier = .8
            if (speed_toward_shelter_future < -4.5):
                # print('future')
                multiplier = .9
            if (speed_toward_shelter_past < -4.5):
                # print('past')
                multiplier = 1
            if (speed_toward_shelter < -5):
                # print('speed')
                multiplier = 1


            body_angle = coordinates['body_angle'][frame_num - 1]
            shoulder_angle = coordinates['shoulder_angle'][frame_num - 1]
            head_angle = coordinates['head_angle'][frame_num - 1]

            head_location = tuple(coordinates['front_location'][:, frame_num - 1].astype(np.uint16))
            butt_location = tuple(coordinates['butt_location'][:, frame_num - 1].astype(np.uint16))
            back_location = tuple(coordinates['back_location'][:, frame_num - 1].astype(np.uint16))
            center_body_location = tuple(coordinates['center_body_location'][:, frame_num - 1].astype(np.uint16))
            center_location = tuple(coordinates['center_location'][:, frame_num - 1].astype(np.uint16))

            model_mouse_mask = cv2.ellipse(model_mouse_mask_initial.copy(), head_location, (8, 4), 180 - head_angle, 0, 360, 100, thickness=-1)
            model_mouse_mask = cv2.ellipse(model_mouse_mask, center_location, (12, 6), 180 - shoulder_angle, 0, 360, 100, thickness=-1)
            model_mouse_mask = cv2.ellipse(model_mouse_mask, center_body_location, (13, 7), 180 - body_angle, 0, 360, 100, thickness=-1)

            speed_toward_shelter = abs(speed_toward_shelter)
            speed_toward_shelter_past = abs(speed_toward_shelter_past)
            speed_toward_shelter_future = abs(speed_toward_shelter_future)

            if not speed_toward_shelter:
                speed_color = np.array([245, 245, 245])
            elif speed_toward_shelter < high_speed: #5:
                speed_color = np.array([255, 254, 253.9])  # blue
                if frame_num > stim_frame:
                    speed_color = np.array([200, 105, 200]) # purple
                    multiplier = 1.7
            elif speed_toward_shelter < high_speed*2:
                speed_color = np.array([220, 220, 200])
                if frame_num > stim_frame:
                    speed_color = np.array([185, 175, 240]) #purple
                    multiplier = .7
            else:
                speed_color = np.array([152, 222, 152])  # green
                if frame_num > stim_frame:
                    speed_color = np.array([150, 150, 250]) # red


            speed_color  = 1 - (1 - speed_color / [255, 255, 255]) / (np.mean(1 - speed_color / [255, 255, 255]) /
                                                                      (.0035/2*(speed_toward_shelter+speed_toward_shelter_future+1)*multiplier ) )

            if not np.isnan(speed_color[0]):
                exploration_arena_in_cum[model_mouse_mask.astype(bool)] = exploration_arena_in_cum[model_mouse_mask.astype(bool)] * speed_color

            cv2.imshow('explore_in', exploration_arena_in_cum)
            # vid_EXPLORE.write(exploration_arena_in_cum)

            if frame_num == stim_frame:
                _, contours, _ = cv2.findContours(model_mouse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # vid_EXPLORE.release()
    try:
        exploration_arena_in_cum_save = cv2.drawContours(exploration_arena_in_cum.copy(), contours, 0, (0, 0, 0), 2)
        exploration_arena_in_cum_save = cv2.drawContours(exploration_arena_in_cum_save, contours, 0, (255, 255, 255), 1)
        scipy.misc.imsave(os.path.join(savepath, videoname + '_spont_homings.tif'), cv2.cvtColor(exploration_arena_in_cum_save, cv2.COLOR_BGR2RGB))
    except:
        print('repeat stimulus trial')
    # Make a position heat map as well
    scale = 1
    H, x_bins, y_bins = np.histogram2d(coordinates['back_location'][0,0:stim_frame], coordinates['back_location'][1,0:stim_frame],
                                       [np.arange(0,width+1,scale), np.arange(0,height+1,scale)], normed=True)
    H = H.T

    H = cv2.GaussianBlur(H, ksize=(5, 5), sigmaX=1, sigmaY=1)
    H[H > np.percentile(H, 98)] = np.percentile(H, 98)

    H_image = (H * 255 / np.max(H)).astype(np.uint8)
    H_image[(H_image < 25) * (H_image > 0)] = 25
    H_image[(arena > 0) * (H_image == 0)] = 9
    H_image = cv2.copyMakeBorder(H_image, border_size, 0,0,0, cv2.BORDER_CONSTANT, value=0)
    textsize = cv2.getTextSize(videoname, 0, .55, 1)[0]
    textX = int((width - textsize[0]) / 2)
    cv2.putText(H_image, videoname, (textX, int(border_size * 3 / 4)), 0, .55, (255, 255, 255), thickness=1)

    cv2.imshow('heat map',  H_image )
    cv2.waitKey(1)
    scipy.misc.imsave(os.path.join(savepath, videoname + '_exploration.tif'), H_image)

    slow_color = np.array([200, 200, 200])
    medium_color = np.array([150, 190, 222])
    fast_color = np.array([122, 222, 122])
    finished_clip = False

    # RUN SAVING AND ANALYSIS OVER EACH FRAME - ######################################
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if display_clip:
        cv2.namedWindow('Trial Clip')
        cv2.moveWindow('Trial Clip', 100, 100)
    while True:  # and analyze: #and not file_already_exists:
        ret, frame = vid.read()  # get the frame

        if ret:
            frame_num = int(vid.get(cv2.CAP_PROP_POS_FRAMES))
            if [registration]:
                # load the fisheye correction
                frame = register_frame(frame, x_offset, y_offset, registration, map1, map2)

            # MAKE ESCAPE TRAJECTORY IMAGE - #######################################
            if make_flight_image: # and frame_num >= stim_frame and (frame_num - stim_frame) < fps * 10:

                if frame_num < stim_frame:
                    model_flight_image = ((model_flight_image.astype(float)*3 + arena_fresh.astype(float)*1 ) / 4).astype(np.uint8)
                    session_trials_plot = ((session_trials_plot.astype(float) * 1 + initial_session_trials_plot.astype(float) * 1) / 2).astype(np.uint8)
                    model_mouse_mask_previous = 0
                elif frame_num == stim_frame:
                    if trial_type:
                        current_model_arena, _, _ = model_arena(frame.shape[0:2], trial_type > 0, False, False, obstacle_type)
                        current_model_arena = cv2.cvtColor(current_model_arena, cv2.COLOR_GRAY2RGB)
                        session_trials_plot = ((session_trials_plot_workspace.astype(float) * 1 + current_model_arena.astype(float) * 4) / 5).astype(np.uint8)
                        model_flight_image = current_model_arena.copy()
                    else:
                        session_trials_plot = initial_session_trials_plot.copy()
                    session_trials_plot_workspace = initial_session_trials_plot_workspace.copy()


                # ##################################
                # add mouse model to model arena
                # ##################################
                body_angle = coordinates['body_angle'][frame_num - 1]
                shoulder_angle = coordinates['shoulder_angle'][frame_num - 1]
                head_angle = coordinates['head_angle'][frame_num - 1]

                head_location = tuple(coordinates['front_location'][:,frame_num-1].astype(np.uint16))
                butt_location = tuple(coordinates['butt_location'][:,frame_num-1].astype(np.uint16))
                back_location = tuple(coordinates['back_location'][:,frame_num-1].astype(np.uint16))
                center_body_location = tuple(coordinates['center_body_location'][:,frame_num-1].astype(np.uint16))
                center_location = tuple(coordinates['center_location'][:, frame_num - 1].astype(np.uint16))

                speed = coordinates['speed'][frame_num-1]
                # print(speed)
                if speed > 20:
                    speed_color = fast_color
                elif speed > 10:
                    speed_color = ( (20 - speed)* medium_color + (speed - 10)* fast_color ) / (20 - 10)
                else:
                    speed_color = ( speed * medium_color + (10 - speed)* slow_color) / 10
                current_speed_color_light = 1 - (1 - speed_color / [255, 255, 255]) / (np.mean(1 - speed_color / [255, 255, 255]) / .08)
                current_speed_color_dark = 1 - (1 - speed_color / [255, 255, 255]) / (np.mean(1 - speed_color / [255, 255, 255]) / .38)
                # print(current_speed_color_light)

                heading_direction = 180 - np.mean([head_angle, body_angle])

                model_mouse_mask = cv2.ellipse(model_mouse_mask_initial.copy() , head_location,(8, 4), 180 - head_angle, 0, 360, 100, thickness=-1)
                model_mouse_mask = cv2.ellipse(model_mouse_mask , center_location, (12, 6), 180 - shoulder_angle ,0, 360, 100, thickness=-1)
                model_mouse_mask = cv2.ellipse(model_mouse_mask , center_body_location, (13, 7), 180 - body_angle, 0, 360, 100, thickness=-1)

                # stop after 10 secs
                if model_flight_image_iters > 10*fps:
                    pass

                # add gray mouse if distant from previous mouse
                elif np.sum(model_mouse_mask * model_mouse_mask_previous) < 40:
                    model_flight_image[model_mouse_mask.astype(bool)] = model_flight_image[model_mouse_mask.astype(bool)] * current_speed_color_dark  #.57
                    model_mouse_mask_previous = model_mouse_mask
                    session_trials_plot[model_mouse_mask.astype(bool)] = session_trials_plot[model_mouse_mask.astype(bool)] * current_flight_color_dark
                    if model_flight_image_iters > 0:
                        session_trials_plot_workspace[model_mouse_mask.astype(bool)] = session_trials_plot_workspace[model_mouse_mask.astype(bool)] * current_flight_color_dark

                # occupancy shading
                elif model_flight_image_iters > 0:
                    # once at shelter, give it only a few more seconds
                    if np.sum(shelter_roi * model_mouse_mask) > 500 and model_flight_image_iters < 8 * fps:
                        model_flight_image_iters = 10 * fps
                    else: # shade in mouse position
                        model_flight_image[model_mouse_mask.astype(bool)] = model_flight_image[model_mouse_mask.astype(bool)] * current_speed_color_light #.9
                        session_trials_plot[model_mouse_mask.astype(bool)] = session_trials_plot[model_mouse_mask.astype(bool)] * current_flight_color_light
                        session_trials_plot_workspace[model_mouse_mask.astype(bool)] = session_trials_plot_workspace[model_mouse_mask.astype(bool)] * current_flight_color_light


                if frame_num == stim_frame:
                # if model_flight_image_iters == 0:
                #     # get contour of initial ellipse, and apply to image at end
                    _, contours, _ = cv2.findContours(model_mouse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


                elif frame_num >= stim_frame:
                    cv2.drawContours(model_flight_image, contours, 0, (0,0,0), 2)
                    cv2.drawContours(model_flight_image, contours, 0, (255,255,255), 1)

                    cv2.drawContours(session_trials_plot_workspace, contours, 0, tuple([int(x) for x in current_flight_color*.7]), thickness = 3)
                    cv2.drawContours(session_trials_plot_workspace, contours, 0, (255,255,255), thickness = 1)

                    cv2.drawContours(session_trials_plot, contours, 0, tuple([int(x) for x in current_flight_color*.7]), thickness = 3)
                    cv2.drawContours(session_trials_plot, contours, 0, (255,255,255), thickness = 1)

                model_flight_image_iters += 1 * (frame_num >= stim_frame)

                session_trials_plot_background[border_size:, 0:-border_size] = cv2.cvtColor(session_trials_plot, cv2.COLOR_BGR2RGB) #session_trials_plot
                model_flight_background[border_size:, 0:-border_size,:] = model_flight_image
                # model_flight_background = cv2.cvtColor(model_flight_background, cv2.COLOR_BGR2RGB)
                # cv2.imshow('session flight image', session_trials_plot)

                # print(body_angle)
                # print(shoulder_angle)
                # print(head_angle)
                # cv2.circle(session_trials_plot_background, tuple(coordinates['nose'][:,frame_num-1].astype(np.uint16)), 2, (0,0,0), -1)
                # cv2.circle(session_trials_plot_background, tuple(coordinates['L eye'][:, frame_num - 1].astype(np.uint16)), 2, (0,0,0), -1)
                # cv2.circle(session_trials_plot_background, tuple(coordinates['R eye'][:, frame_num - 1].astype(np.uint16)), 2, (0,0,0), -1)
                # cv2.circle(session_trials_plot_background, tuple(coordinates['L ear'][:, frame_num - 1].astype(np.uint16)), 2, (0,0,0), -1)
                # cv2.circle(session_trials_plot_background, tuple(coordinates['neck'][:, frame_num - 1].astype(np.uint16)), 2, (0,0,0), -1)
                # cv2.circle(session_trials_plot_background, tuple(coordinates['R ear'][:, frame_num - 1].astype(np.uint16)), 2, (0,0,0), -1)


                cv2.imshow('session flight bg', session_trials_plot_background)
                # cv2.imshow('small_model flight image', model_flight_background)



                # check on wall progress for .5 sec
                if analyze_wall:
                    wall_mouse_show, wall_height_timecourse = do_wall_analysis(trial_type, frame[:,:,0], frame_num, stim_frame, fps, x_wall_up_ROI_left,
                                     x_wall_up_ROI_right, y_wall_up_ROI, wall_darkness, wall_height_timecourse, wall_mouse_already_shown)

            # SHOW BOUNDARY AND TIME COUNTER - #######################################
            if counter and (display_clip or save_clip or wall_mouse_show):
                # cv2.rectangle(frame, (0, height), (150, height - 60), (150,150,150), -1)
                if frame_num < stim_frame:
                    cur_color = tuple(
                        [x * ((frame_num - start_frame) / (stim_frame - start_frame)) for x in pre_stim_color])
                    sign = ''
                else:
                    cur_color = tuple(
                        [x * (1 - (frame_num - stim_frame) / (end_frame - stim_frame)) for x in post_stim_color])
                    sign = '+'

                # border and colored rectangle around frame
                frame = cv2.copyMakeBorder(frame, border_size, border_size, border_size, border_size,
                                           cv2.BORDER_CONSTANT, value=cur_color)

                # report video details
                textsize = cv2.getTextSize(videoname, 0, .55, 1)[0]
                textX = int((width+2*border_size - textsize[0]) / 2)
                cv2.rectangle(frame, (0, 0), (width+2*border_size, border_size), 0, -1)
                cv2.putText(frame, videoname, (textX, int(border_size*3/4)), 0, .55, (255, 255, 255), thickness=1)
                # cv2.putText(model_flight_image, videoname, (textX, border_size - 5), 0, .55, (255, 255, 255), thickness=1)

                # report time relative to stimulus onset
                frame_time = (frame_num - stim_frame) / fps
                frame_time = str(round(.1 * round(frame_time / .1), 1)) + '0' * (abs(float(frame_time)) < 10)
                cv2.putText(frame, sign + str(frame_time) + 's', (width - 110, height + 10), 0, 1, (180, 180, 180),
                            thickness=2)

            else:
                frame = frame[:, :, 0]  # or use 2D grayscale image instead

            # SHOW AND SAVE FRAME - #######################################
            if display_clip:
                cv2.imshow('Trial Clip', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if wall_mouse_show:
                scipy.misc.imsave(os.path.join(savepath, videoname + '-probe trial.tif'), frame[:, :, ::-1])
                wall_mouse_already_shown = True
            if save_clip:
                video_clip.write(frame)
                session_video.write(frame)
                video_clip_dlc.write(model_flight_background)
                session_trials_video.write(session_trials_plot_background)
            if frame_num >= end_frame:
                finished_clip = True
                break
        else:
            print('Problem with movie playback')
            cv2.waitKey(1000)
            break

    # wrap up
    print(wall_height_timecourse)
    vid.release()

    if make_flight_image and finished_clip:
        scipy.misc.imsave(os.path.join(savepath, videoname + '_dlc.tif'), cv2.cvtColor(model_flight_background, cv2.COLOR_BGR2RGB))

        scipy.misc.imsave(os.path.join(savepath, videoname + '_dlc_history.tif'), cv2.cvtColor(session_trials_plot_background, cv2.COLOR_BGR2RGB))

    if save_clip:
        video_clip.release()
        video_clip_dlc.release()

        if trial_num == (number_of_trials - 1):
            session_trials_plot_background[border_size:, 0:-border_size] = cv2.cvtColor( ( ( session_trials_plot_workspace.astype(float) * 2
                                                                    + cv2.cvtColor(arena.copy(), cv2.COLOR_GRAY2RGB).astype(float) * 3) / 5)
                                                                                         .astype(np.uint8),cv2.COLOR_BGR2RGB)
            session_trials_video.write(session_trials_plot_background)

            session_trials_plot_workspace = ((session_trials_plot_workspace.astype(float) * 2 + cv2.cvtColor(arena.copy(), cv2.COLOR_GRAY2RGB).astype(float) * 2) / 4).astype(np.uint8)

    return session_trials_plot_workspace, exploration_arena_in_cum






def initialize_wall_analysis(analyze_wall, stim_frame, start_frame, end_frame, registration, x_offset, y_offset, vid, width, height):
    '''determine whether this is a wall up, wall down, wall, or no wall trial'''
    # setup fisheye correction
    if registration[3]:
        maps = np.load(registration[3])
        map1 = maps[:, :, 0:2]
        map2 = maps[:, :, 2] * 0
    else:
        print(colored('Fisheye correction unavailable', 'green'))

    # arena = model_arena((width,height))
    x_wall_up_ROI_left = [int(x * width / 1000) for x in [223 - 10, 249 + 10]]
    x_wall_up_ROI_right = [int(x * width / 1000) for x in [752 - 10, 777 + 10]]
    y_wall_up_ROI = [int(x * height / 1000) for x in [494 - 10, 504 + 10]]

    # check state of wall on various frames
    frames_to_check = [start_frame, stim_frame - 1, stim_frame + 13, end_frame]
    wall_darkness = np.zeros((len(frames_to_check), 2))
    for i, frame_to_check in enumerate(frames_to_check):
        vid.set(cv2.CAP_PROP_POS_FRAMES, frame_to_check)
        ret, frame = vid.read()
        frame_register = frame[:, :, 0]
        if registration[3]:
            frame_register = cv2.copyMakeBorder(frame_register, y_offset,
                                                int((map1.shape[0] - frame.shape[0]) - y_offset),
                                                x_offset, int((map1.shape[1] - frame.shape[1]) - x_offset),
                                                cv2.BORDER_CONSTANT, value=0)
            frame_register = cv2.remap(frame_register, map1, map2, interpolation=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            frame_register = frame_register[y_offset:-int((map1.shape[0] - frame.shape[0]) - y_offset),
                             x_offset:-int((map1.shape[1] - frame.shape[1]) - x_offset)]
        frame = cv2.cvtColor(cv2.warpAffine(frame_register, registration[0], frame.shape[0:2]),
                             cv2.COLOR_GRAY2RGB)
        wall_darkness[i, 0] = sum(
            sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1], x_wall_up_ROI_left[0]:x_wall_up_ROI_left[1], 0] < 200))
        wall_darkness[i, 1] = sum(
            sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1], x_wall_up_ROI_right[0]:x_wall_up_ROI_right[1],
                0] < 200))

    wall_darkness_pre = np.min(wall_darkness[0:int(len(frames_to_check) / 2), 0:2])
    wall_darkness_post = np.min(wall_darkness[int(len(frames_to_check) / 2):len(frames_to_check), 0:2])

    # use these darkness levels to detect whether wall is up, down, rising, or falling:
    wall_mouse_already_shown = False
    wall_mouse_show = False
    finished_clip = False
    if (wall_darkness_pre - wall_darkness_post) < -30:
        print(colored('Wall rising trial!', 'green'))
        wall_height_timecourse = [0]
        trial_type = 1
    elif (wall_darkness_pre - wall_darkness_post) > 30:
        print(colored('Wall falling trial', 'green'))
        wall_height_timecourse = [1]
        trial_type = -1
    elif (wall_darkness_pre > 85) and (wall_darkness_post > 85):
        print(colored('Obstacle trial', 'green'))
        trial_type = 2
        wall_height_timecourse = 1  # [1 for x in list(range(int(fps * .5)))]
    elif (wall_darkness_pre < 85) and (wall_darkness_post < 85):
        print(colored('No obstacle trial', 'green'))
        trial_type = 0
        wall_height_timecourse = 0  # [0 for x in list(range(int(fps * .5)))]
    else:
        print('Uh-oh -- not sure what kind of trial!')

# print(wall_darkness)
# if not trial_type:
#     analyze = False
# else:
#     analyze = True

    return wall_height_timecourse, trial_type, x_wall_up_ROI_left, x_wall_up_ROI_right, y_wall_up_ROI, wall_darkness, wall_mouse_already_shown, map1, map2


def do_wall_analysis(trial_type, frame, frame_num, stim_frame, fps, x_wall_up_ROI_left, x_wall_up_ROI_right, y_wall_up_ROI, wall_darkness, wall_height_timecourse, wall_mouse_already_shown):

    wall_mouse_show = False

    if trial_type and (frame_num - stim_frame) < fps * .5:
        left_wall_darkness = sum(sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1],
                                     x_wall_up_ROI_left[0]:x_wall_up_ROI_left[1]] < 200))
        right_wall_darkness = sum(sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1],
                                      x_wall_up_ROI_right[0]:x_wall_up_ROI_right[1]] < 200))

        # calculate overall change
        left_wall_darkness_change_overall = left_wall_darkness - wall_darkness[1, 0]
        right_wall_darkness_change_overall = right_wall_darkness - wall_darkness[1, 1]

        # show wall up timecourse
        if trial_type == -1:
            wall_height = round(1 - np.mean(
                [left_wall_darkness_change_overall / (wall_darkness[2, 0] - wall_darkness[1, 0]),
                 right_wall_darkness_change_overall / (wall_darkness[2, 1] - wall_darkness[1, 1])]),
                                1)
            wall_height_timecourse.append(wall_height)

            # show mouse when wall is down
            if wall_height <= .1 and not wall_mouse_already_shown:
                wall_mouse_show = True

        if trial_type == 1:
            wall_height = round(np.mean(
                [left_wall_darkness_change_overall / (wall_darkness[2, 0] - wall_darkness[1, 0]),
                 right_wall_darkness_change_overall / (wall_darkness[2, 1] - wall_darkness[1, 1])]),1)
            wall_height_timecourse.append(wall_height)

            # show mouse when wall is up
            if wall_height >= .6 and not wall_mouse_already_shown:
                wall_mouse_show = True

    return wall_mouse_show, wall_height_timecourse


# def initialize_session_trials_plot(image_shape, obstacle_type):
#     arena, _, _ = model_arena(image_shape, 0, 0, 0, obstacle_type)
#     return arena


def extract_coordinates_with_dlc(dlc_config_settings, video, registration, stims, x_offset, y_offset, shelter_location, size, obstacle_type):
    '''extract coordinates for each frame, given a video and DLC network'''

    # analyze_videos(dlc_config_settings['config_file'], video)
    # create_labeled_video(dlc_config_settings['config_file'], video)

    # read the freshly saved coordinates file
    coordinates_file = glob.glob(os.path.dirname(video[0]) + '\\*.h5')[0]
    DLC_network = os.path.basename(coordinates_file)
    DLC_network = DLC_network[DLC_network.find('Deep'):-3]
    body_parts = dlc_config_settings['body parts']

    DLC_dataframe = pd.read_hdf(coordinates_file)

    # plot body part positions over time
    import matplotlib.pyplot as plt
    position_figure = plt.figure('position over time')
    ax = position_figure.add_subplot(111)
    # plt.title('DLC positions')
    # plt.xlabel('frame number')
    # plt.ylabel('position (pixels)')
    coordinates = {}

    # array of all body parts, axis x body part x frame
    all_body_parts = np.zeros((2, len(body_parts), DLC_dataframe[DLC_network]['nose'].values.shape[0]))

    # fisheye correct the coordinates
    registration = invert_fisheye_map(registration, dlc_config_settings['inverse_fisheye_map_location'])
    inverse_fisheye_maps = np.load(registration[4])


    for i, body_part in enumerate(body_parts):
        # initialize coordinates
        coordinates[body_part] = np.zeros((3, len(DLC_dataframe[DLC_network][body_part]['x'].values)))

        # extract coordinates
        for j, axis in enumerate(['x', 'y']):
            coordinates[body_part][j] = DLC_dataframe[DLC_network][body_part][axis].values
            coordinates[body_part][j] = DLC_dataframe[DLC_network][body_part][axis].values

    for bp, body_part in enumerate(body_parts):

        # get likelihood
        likelihood = DLC_dataframe[DLC_network][body_part]['likelihood'].values
        coordinates[body_part][2] = likelihood

        # remove coordinates with low confidence
        coordinates[body_part][0][likelihood < .999999] = np.nan
        coordinates[body_part][1][likelihood < .999999] = np.nan

        # put all together
        all_body_parts[:, bp, :] = coordinates[body_part][0:2]
    median_positions = np.nanmedian(all_body_parts, axis=1)
    num_of_nans = np.sum(np.isnan(all_body_parts[0,:,:]),0)
    no_median = num_of_nans > 8

    for bp, body_part in enumerate(body_parts):

        # remove coordinates far from rest of body parts
        distance_from_median_position = np.sqrt( (coordinates[body_part][0] - median_positions[0,:])**2 + (coordinates[body_part][1] - median_positions[1,:])**2 )
        coordinates[body_part][0][distance_from_median_position > 50] = np.nan
        coordinates[body_part][1][distance_from_median_position > 50] = np.nan

        coordinates[body_part][0][no_median] = np.nan
        coordinates[body_part][1][no_median] = np.nan

        # convert original coordinates to registered coordinates
        nan_index = np.isnan(coordinates[body_part][0])
        coordinates[body_part][0] = inverse_fisheye_maps[
                                        coordinates[body_part][1].astype(np.uint16) + y_offset, coordinates[body_part][
                                            0].astype(np.uint16) + x_offset, 0] - x_offset
        coordinates[body_part][1] = inverse_fisheye_maps[
                                        coordinates[body_part][1].astype(np.uint16) + y_offset, coordinates[body_part][
                                            0].astype(np.uint16) + x_offset, 1] - y_offset

        # affine transform to match model arena
        transformed_points = np.matmul(np.append(registration[0], np.zeros((1, 3)), 0),
                                       np.concatenate((coordinates[body_part][0:1], coordinates[body_part][1:2],
                                                       np.ones((1, len(coordinates[body_part][0])))), 0))
        coordinates[body_part][0] = transformed_points[0, :]
        coordinates[body_part][1] = transformed_points[1, :]

        coordinates[body_part][0][nan_index] = np.nan
        coordinates[body_part][1][nan_index] = np.nan

        all_body_parts[:, bp, :] = coordinates[body_part][0:2]

        # plot the coordinates
        # position_figure = plt.figure('position over time')
        # ax = position_figure.add_subplot(111)
        # ax.plot(distance_from_median_position)

        ax.plot(np.sqrt((coordinates[body_part][0] - shelter_location[0] * size[0] / 1000) ** 2 + (coordinates[body_part][1] - shelter_location[1] * size[1] / 1000) ** 2))
    # plt.show()
        # print(body_part)

    ax.legend(body_parts)

    # ax.plot(coordinates['head_angle'])
    # plt.show()

    # compute some metrics
    # for i, body_part in enumerate(body_parts):
    #     all_body_parts[:, i, :] = coordinates[body_part][0:2]

    # coordinates['head_location'] = np.nanmean(all_body_parts[:, 0:5, :], axis=1)
    # coordinates['snout_location'] = np.nanmean(all_body_parts[:, 0:3, :], axis=1)
    # coordinates['neck_location'] = np.nanmean(all_body_parts[:, 3:6, :], axis=1)
    # coordinates['butt_location'] = np.nanmean(all_body_parts[:, 9:, :], axis=1)
    # coordinates['back_location'] = np.nanmean(all_body_parts[:, 6:9, :], axis=1)
    # coordinates['nack_location'] = np.nanmean(all_body_parts[:, 3:9, :], axis=1)
    # coordinates['front_location'] = np.nanmean(all_body_parts[:, 0:9, :], axis=1)
    # coordinates['center_body_location'] = np.nanmean(all_body_parts[:, 6:, :], axis=1)
    # coordinates['center_location'] = np.nanmean(all_body_parts[:, :, :], axis=1)
    #
    # delta_position = np.concatenate( ( np.zeros((2,1)), np.diff(coordinates['center_location']) ) , axis = 1)
    # coordinates['speed'] = np.sqrt(delta_position[0,:]**2 + delta_position[1,:]**2)
    #
    # coordinates['distance_from_shelter'] = np.sqrt((coordinates['center_location'][0] - 500 * 720 / 1000) ** 2 +
    #                                                (coordinates['center_location'][1] - 885 * 720 / 1000) ** 2)
    #
    # coordinates['speed_toward_shelter'] = np.concatenate( ([0], np.diff(coordinates['distance_from_shelter'])))
    #
    # coordinates['speed'] = np.array(pd.Series(coordinates['speed']).interpolate())
    # coordinates['distance_from_shelter'] = np.array(pd.Series(coordinates['distance_from_shelter']).interpolate())
    # coordinates['speed_toward_shelter'] = np.array(pd.Series(coordinates['speed_toward_shelter']).interpolate())
    #
    # locations = ['speed', 'distance_from_shelter', 'speed_toward_shelter', 'head_location', 'butt_location', 'snout_location',
    #             'neck_location', 'back_location', 'nack_location', 'center_body_location', 'center_location' ]
    #
    # for loc_num, loc in enumerate(locations):
    #     if loc_num < 3:
    #         coordinates[loc] = np.array(pd.Series(coordinates[loc]).interpolate())
    #         coordinates[loc] = np.array(pd.Series(coordinates[loc]).fillna(method='bfill'))
    #         coordinates[loc] = np.array(pd.Series(coordinates[loc]).fillna(method='ffill'))
    #     else:
    #         for i in [0,1]:
    #             coordinates[loc][i] = np.array(pd.Series(coordinates[loc][i]).interpolate())
    #             coordinates[loc][i] = np.array(pd.Series(coordinates[loc][i]).fillna(method='bfill'))
    #             coordinates[loc][i] = np.array(pd.Series(coordinates[loc][i]).fillna(method='ffill'))
    #
    # coordinates['body_angle'] = np.angle((coordinates['back_location'][0] - coordinates['butt_location'][0]) + (-coordinates['back_location'][1] + coordinates['butt_location'][1]) * 1j, deg=True)
    # coordinates['shoulder_angle'] = np.angle((coordinates['head_location'][0] - coordinates['center_body_location'][0]) + (-coordinates['head_location'][1] + coordinates['center_body_location'][1]) * 1j, deg=True)
    # head_angle = np.zeros( (len(coordinates['body_angle']) , 2))
    # head_angle[:,0] = np.angle((coordinates['snout_location'][0] - coordinates['neck_location'][0]) + (-coordinates['snout_location'][1] + coordinates['neck_location'][1]) * 1j, deg=True)
    # head_angle[:,1] = np.angle((coordinates['head_location'][0] - coordinates['nack_location'][0]) + (-coordinates['head_location'][1] + coordinates['nack_location'][1]) * 1j, deg=True)
    # coordinates['head_angle'] = np.nanmean(head_angle,1)


    return coordinates, registration


# d = 22
# triangle_1 = (int( center_location[0] - d * np.cos(np.radians(heading_direction))), int(center_location[1] - d * np.sin(np.radians(heading_direction))))
# triangle_2 = (int( center_location[0] + d * np.cos(np.radians(heading_direction)) - d/5 * np.cos(np.radians(heading_direction - 90))),
#               int( center_location[1] + d * np.sin(np.radians(heading_direction)) - d/5 * np.sin(np.radians(heading_direction - 90))))
# triangle_3 = (int( center_location[0] + d * np.cos(np.radians(heading_direction)) + d/5 * np.cos(np.radians(heading_direction - 90))),
#               int( center_location[1] + d * np.sin(np.radians(heading_direction)) + d/5 * np.sin(np.radians(heading_direction - 90))))

# small_contours = [np.array([triangle_1, triangle_2, triangle_3])]
# cv2.drawContours(model_mouse_mask, small_contours, 0, 100, 2)
# cv2.circle(model_mouse_mask, center_location, 10, (200, 90, 90), -1)


    # video_clip_EXPLORE = cv2.VideoWriter(os.path.join(savepath, videoname + '_EXPLORE.avi'), fourcc,
    #                                  200, (width, height), True)
    # in_color = np.array([122, 222, 122])
    # out_color = np.array([122, 122, 222])

    # explore_in_color = 1 - ( 1 - in_color / [255, 255, 255] ) / ( np.mean( 1 - in_color / [255, 255, 255] ) / .08)
    # explore_out_color = 1 - ( 1 - out_color / [255, 255, 255] ) / ( np.mean( 1 - out_color / [255, 255, 255] ) / .08)

# AWAY FROM SHELTER
# if speed_toward_shelter > 0:
#     fast_color = np.array([152, 152, 222]) # red
#     slow_color = np.array([155, 225, 245]) # brown-yellow
#     slow_color = np.array([224, 245, 245])  # gray yellow
#     multiplier = .02
#     if speed > 10:
#         speed_color = fast_color
#     else:
#         speed_color = (speed * fast_color + (10 - speed) * slow_color) / 10
#
#     speed_color = 1 - (1 - speed_color / [255, 255, 255]) / (np.mean(1 - speed_color / [255, 255, 255]) / (.1 * speed * multiplier))
#
#     exploration_arena_both[model_mouse_mask.astype(bool)] = exploration_arena_both[model_mouse_mask.astype(bool)] * speed_color

########################################################################################################################
