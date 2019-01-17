from Utils.imports import *
from Config import track_options, dlc_config_settings

# if track_options['track whole session']:
#     from deeplabcut.pose_estimation_tensorflow import analyze_videos
    # from deeplabcut.utils import create_labeled_video

import cv2
import numpy as np
import scipy.misc
from termcolor import colored
from tqdm import tqdm
import glob
from Config import y_offset, x_offset
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
                   registration = 0, fps=False, analyze_wall = True, save_clip = False, display_clip = False, counter = True, make_flight_image = True):
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
    pre_stim_color = [255, 120, 120]
    post_stim_color = [120, 120, 255]

    # setup fisheye correction
    if registration[3]:
        maps = np.load(registration[3])
        map1 = maps[:, :, 0:2]
        map2 = maps[:, :, 2] * 0
    else:
        print(colored('Fisheye correction unavailable', 'green'))

    # generate model arena and wall ROIs
    if analyze_wall:
        # arena = model_arena((width,height))
        x_wall_up_ROI_left = [int(x*width/1000) for x in [223-10,249+10]]
        x_wall_up_ROI_right = [int(x*width/1000) for x in [752-10,777+10]]
        y_wall_up_ROI = [int(x*height/1000) for x in [494 - 10,504 + 10]]

        #check state of wall on various frames
        frames_to_check = [start_frame, stim_frame-1, stim_frame + 13, end_frame]
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
            frame = cv2.cvtColor(cv2.warpAffine(frame_register, registration[0], frame.shape[0:2]), cv2.COLOR_GRAY2RGB)
            wall_darkness[i, 0] = sum(
                sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1], x_wall_up_ROI_left[0]:x_wall_up_ROI_left[1], 0] < 200))
            wall_darkness[i, 1] = sum(
                sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1], x_wall_up_ROI_right[0]:x_wall_up_ROI_right[1], 0] < 200))

        wall_darkness_pre = np.min(wall_darkness[0:int(len(frames_to_check)/2),0:2])
        wall_darkness_post = np.min(wall_darkness[int(len(frames_to_check)/2):len(frames_to_check),0:2])

        # use these darkness levels to detect whether wall is up, down, rising, or falling:
        wall_mouse_already_shown = False
        wall_mouse_show = False
        finished_clip = False
        if (wall_darkness_pre - wall_darkness_post) < -30:
            print(colored('Wall rising trial detected!', 'green'))
            wall_height_timecourse = [0]
            trial_type = 1
        elif (wall_darkness_pre - wall_darkness_post) > 30:
            print(colored('Wall falling trial detected!', 'green'))
            wall_height_timecourse = [1]
            trial_type = -1
        elif (wall_darkness_pre > 85) and (wall_darkness_post > 85):
            print(colored('Wall trial detected', 'green'))
            trial_type = 0
            wall_height_timecourse = 1 #[1 for x in list(range(int(fps * .5)))]
        elif (wall_darkness_pre < 85) and (wall_darkness_post < 85):
            print(colored('No Wall trial detected', 'green'))
            trial_type = 0
            wall_height_timecourse = 0 # [0 for x in list(range(int(fps * .5)))]
        else:
            print('Uh-oh -- not sure what kind of trial!')
    else:
        trial_type = 0
    # print(wall_darkness)
    # if not trial_type:
    #     analyze = False
    # else:
    #     analyze = True

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
                if registration[3]:
                    frame_register = cv2.copyMakeBorder(frame_register, y_offset, int((map1.shape[0] - frame.shape[0]) - y_offset),
                                                        x_offset, int((map1.shape[1] - frame.shape[1]) - x_offset), cv2.BORDER_CONSTANT, value=0)
                    frame_register = cv2.remap(frame_register, map1, map2, interpolation=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    frame_register = frame_register[y_offset:-int((map1.shape[0] - frame.shape[0]) - y_offset),
                                     x_offset:-int((map1.shape[1] - frame.shape[1]) - x_offset)]
                frame = cv2.cvtColor(cv2.warpAffine(frame_register, registration[0], frame.shape[0:2]),cv2.COLOR_GRAY2RGB)

            # MAKE ESCAPE TRAJECTORY IMAGE - #######################################
            if make_flight_image:
                # at stimulus onset, take this frame to lay all the superimposed mice on top of
                if frame_num == stim_frame:
                    flight_image_by_distance = frame[:,:,0].copy()

                # in subsequent frames, see if frame is different enough from previous image to merit joining the image
                if frame_num >= stim_frame and (frame_num - stim_frame) < fps*10:

                    # get the number of pixels that are darker than the flight image
                    difference_from_previous_image = ((frame[:,:,0]+.001) / (flight_image_by_distance+.001))<.55 #.5 original parameter
                    number_of_darker_pixels = np.sum(difference_from_previous_image)

                    # check on wall progress for .5 sec
                    if trial_type and (frame_num - stim_frame) < fps * .5:
                        left_wall_darkness = sum(sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1],
                                                             x_wall_up_ROI_left[0]:x_wall_up_ROI_left[1], 0] < 200))
                        right_wall_darkness = sum(sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1],
                                                             x_wall_up_ROI_right[0]:x_wall_up_ROI_right[1], 0] < 200))

                        # calculate overall change
                        left_wall_darkness_change_overall = left_wall_darkness - wall_darkness[1,0]
                        right_wall_darkness_change_overall = right_wall_darkness - wall_darkness[1,1]

                        # show wall up timecourse
                        wall_mouse_show = False
                        if trial_type == -1:
                            wall_height = round(1 - np.mean(
                                [left_wall_darkness_change_overall / (wall_darkness[2,0] - wall_darkness[1,0]),
                                 right_wall_darkness_change_overall / (wall_darkness[2,1] - wall_darkness[1,1])]),1)
                            wall_height_timecourse.append(wall_height)

                            # show mouse when wall is down
                            if wall_height <= .1 and not wall_mouse_already_shown:
                                wall_mouse_show = True

                        if trial_type == 1:
                            wall_height = round(np.mean(
                                [left_wall_darkness_change_overall / (wall_darkness[2,0] - wall_darkness[1,0]),
                                 right_wall_darkness_change_overall / (wall_darkness[2,1] - wall_darkness[1,1])]),1)
                            wall_height_timecourse.append(wall_height)

                            # show mouse when wall is up
                            if wall_height >= .6 and not wall_mouse_already_shown:
                                wall_mouse_show = True

                    # if that number is high enough, add mouse to image
                    if number_of_darker_pixels > 950: # 850 original parameter
                        # add mouse where pixels are darker
                        flight_image_by_distance[difference_from_previous_image] = frame[difference_from_previous_image,0]



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
            if wall_mouse_show:
                cv2.imshow('Probe moment', frame)
                cv2.waitKey(10)
                scipy.misc.imsave(os.path.join(savepath, videoname + '-probe trial.tif'), frame[:,:,::-1])
                wall_mouse_already_shown = True
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
    print(wall_height_timecourse)
    vid.release()
    if make_flight_image and finished_clip:
        flight_image_by_distance = cv2.copyMakeBorder(flight_image_by_distance, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0)
        cv2.putText(flight_image_by_distance, videoname, (border_size, border_size-5), 0, .55, (180, 180, 180), thickness=1)
        cv2.imshow('Flight image', flight_image_by_distance)
        cv2.moveWindow('Flight image',1000,100)
        cv2.waitKey(10)
        scipy.misc.imsave(os.path.join(savepath, videoname + '.tif'), flight_image_by_distance)
    if save_clip:
        video_clip.release()
    # cv2.destroyAllWindows()




# =========================================================================================
#        GENERATE PERI-STIMULUS VIDEO CLIPS and FLIGHT IMAGE ***with wall, using DLC***
# =========================================================================================
def peri_stimulus_analysis(coordinates, vidpath = '', videoname = '', savepath = '', session_video = None, previous_stim_frame = 0, session_trials_video = None, session_trials_plot_workspace = None, session_trials_plot_background = None,
                           exploration_arena_in_cum = None, number_of_trials = 10, trial_num = 0, start_frame=0., end_frame=100., stim_frame = 0,
                           registration = 0, fps=False, border_size = 40, save_clip = False, analyze_wall=True, display_clip = False, counter = True, make_flight_image = True):

    # GET BEHAVIOUR VIDEO - ######################################
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

    #     current_flight_color = (early_trial_color * (middle_trial - trial_num) + middle_trial_color * trial_num) / middle_trial
    # else:
    #     current_flight_color = (middle_trial_color * (number_of_trials - trial_num - 1) + late_trial_color * (trial_num - middle_trial) ) / (number_of_trials - middle_trial - 1)

    current_flight_color_light = 1 - ( 1 - current_flight_color / [255, 255, 255] ) / ( np.mean( 1 - current_flight_color / [255, 255, 255] ) / .08)
    current_flight_color_dark = 1 - (1 - current_flight_color / [255, 255, 255]) / ( np.mean(1 - current_flight_color / [255, 255, 255]) / .38)



    # set up model arena
    ret, frame = vid.read()
    arena, _, shelter_roi = model_arena(frame.shape[0:2], trial_type*(trial_type-1), False, False)
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

    # for debugging - extract all coordinates
    # array of all body parts, axis x body part x frame
    all_body_parts = np.zeros((2, len(body_parts), coordinates['nose'].shape[1]))
    for i, body_part in enumerate(body_parts):
        # put all together
        all_body_parts[:, i, :] = coordinates[body_part]


    # #####################################
    # compute and display EXPLORATION
    # go through each frame, adding the mouse silhouette
    # ########################################
    # exploration_arena_in = arena_fresh.copy()
    # slow_color = np.array([155, 245, 245])  # brown-yellow
    # exploration_arena_in_cum = arena_fresh.copy()
    high_speed = np.percentile(coordinates['speed_toward_shelter'], 99)
    print(high_speed)
    vid_EXPLORE = cv2.VideoWriter(os.path.join(savepath, videoname + '_dlc_spont_homings.avi'), fourcc,
                                     fps, (width, height), True)

    for frame_num in range(previous_stim_frame + 300, stim_frame + 300):

        # speed_toward_shelter = coordinates['speed_toward_shelter'][frame_num - 1]
        speed_toward_shelter = np.median(coordinates['speed_toward_shelter'][frame_num - 1 - 8:frame_num+8])
        speed_toward_shelter_future = np.median(coordinates['speed_toward_shelter'][frame_num - 1:frame_num+30])
        speed_toward_shelter_far_future = np.median(coordinates['speed_toward_shelter'][frame_num + 15:frame_num + 60])
        speed_toward_shelter_past = np.median(coordinates['speed_toward_shelter'][frame_num - 30:frame_num - 1])
        speed = coordinates['speed'][frame_num - 1]

        if (speed_toward_shelter < -5) or (speed_toward_shelter_future < -4) or (speed_toward_shelter_far_future < -5) or \
                (speed_toward_shelter_past < -4) or (frame_num > stim_frame and speed > 3 and speed_toward_shelter < 3) or frame_num == stim_frame:

            multiplier = 1
            if (speed_toward_shelter_future < -4):
                print('future')
                multiplier = .9
            if (speed_toward_shelter_far_future < -5):
                print('far future')
                multiplier = .9
            if (speed_toward_shelter_past < -4):
                print('past')
                multiplier = 1
            if (speed_toward_shelter < -5):
                print('speed')
                multiplier = 1


            body_angle = coordinates['body_angle'][frame_num - 1]
            head_angle = coordinates['head_angle'][frame_num - 1]

            head_location = tuple(coordinates['head_location'][:, frame_num - 1].astype(np.uint16))
            butt_location = tuple(coordinates['butt_location'][:, frame_num - 1].astype(np.uint16))
            back_location = tuple(coordinates['back_location'][:, frame_num - 1].astype(np.uint16))
            center_body_location = tuple(coordinates['center_body_location'][:, frame_num - 1].astype(np.uint16))

            model_mouse_mask = cv2.ellipse(model_mouse_mask_initial.copy(), head_location, (8, 4), 180 - head_angle, 0, 360, 100, thickness=-1)
            model_mouse_mask = cv2.ellipse(model_mouse_mask, back_location, (12, 6), 180 - np.mean([head_angle, body_angle]), 0, 360, 100, thickness=-1)
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
                    multiplier = .8
            else:
                speed_color = np.array([152, 222, 152])  # green
                if frame_num > stim_frame:
                    speed_color = np.array([150, 150, 250]) # red

            speed_color  = 1 - (1 - speed_color / [255, 255, 255]) / (np.mean(1 - speed_color / [255, 255, 255]) /
                                                                      (.0035*(speed_toward_shelter+1)*multiplier ) )

            if not np.isnan(speed_color[0]):
                exploration_arena_in_cum[model_mouse_mask.astype(bool)] = exploration_arena_in_cum[model_mouse_mask.astype(bool)] * speed_color

            cv2.imshow('explore_in', exploration_arena_in_cum)
            vid_EXPLORE.write(exploration_arena_in_cum)

            if frame_num == stim_frame:
                _, contours, _ = cv2.findContours(model_mouse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid_EXPLORE.release()
    exploration_arena_in_cum_save = cv2.drawContours(exploration_arena_in_cum.copy(), contours, 0, (0, 0, 0), 2)
    exploration_arena_in_cum_save = cv2.drawContours(exploration_arena_in_cum_save, contours, 0, (255, 255, 255), 1)
    scipy.misc.imsave(os.path.join(savepath, videoname + '_spont_homings.tif'), cv2.cvtColor(exploration_arena_in_cum_save, cv2.COLOR_BGR2RGB))

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
                    session_trials_plot_workspace = initial_session_trials_plot_workspace.copy()
                    model_mouse_mask_previous = 0
                elif frame_num == stim_frame:
                    if trial_type:
                        current_model_arena, _, _ = model_arena(frame.shape[0:2], trial_type > 0, False, False)
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
                head_angle = coordinates['head_angle'][frame_num - 1]
                # if (abs(body_angle - coordinates['body_angle'][frame_num - 2]) > 100 or
                #     abs(head_angle - coordinates['head_angle'][frame_num - 2]) > 100) and not frozen_angle:
                #     frozen_angle = True
                # else:
                #     frozen_angle = False

                head_location = tuple(coordinates['head_location'][:,frame_num-1].astype(np.uint16))
                butt_location = tuple(coordinates['butt_location'][:,frame_num-1].astype(np.uint16))
                back_location = tuple(coordinates['back_location'][:,frame_num-1].astype(np.uint16))
                center_body_location = tuple(coordinates['center_body_location'][:,frame_num-1].astype(np.uint16))

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
                model_mouse_mask = cv2.ellipse(model_mouse_mask , back_location, (12, 6), heading_direction,0, 360, 100, thickness=-1)
                model_mouse_mask = cv2.ellipse(model_mouse_mask , center_body_location, (13, 7), 180 - body_angle, 0, 360, 100, thickness=-1)


                # stop after 10 secs
                if model_flight_image_iters > 10*fps:
                    pass

                # add gray mouse if distant from previous mouse
                elif np.sum(model_mouse_mask * model_mouse_mask_previous) < 40:
                    model_flight_image[model_mouse_mask.astype(bool)] = model_flight_image[model_mouse_mask.astype(bool)] * current_speed_color_dark  #.57
                    model_mouse_mask_previous = model_mouse_mask
                    session_trials_plot[model_mouse_mask.astype(bool)] = session_trials_plot[model_mouse_mask.astype(bool)] * current_flight_color_dark
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
                    # _, session_contours, _ = cv2.findContours(model_mouse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


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
                # cv2.imshow('session flight bg', session_trials_plot_background)
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
        print(colored('Wall trial', 'green'))
        trial_type = 2
        wall_height_timecourse = 1  # [1 for x in list(range(int(fps * .5)))]
    elif (wall_darkness_pre < 85) and (wall_darkness_post < 85):
        print(colored('No Wall trial', 'green'))
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


def initialize_session_trials_plot(image_shape):
    arena, _, _ = model_arena(image_shape, 0)
    return arena


def extract_coordinates_with_dlc(dlc_config_settings, video, registration):
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
    # import matplotlib.pyplot as plt
    # position_figure = plt.figure('position over time')
    # ax = position_figure.add_subplot(111)
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
        coordinates[body_part] = np.zeros((2, len(DLC_dataframe[DLC_network][body_part]['x'].values)))

        # extract coordinates
        for j, axis in enumerate(['x', 'y']):
            coordinates[body_part][j] = DLC_dataframe[DLC_network][body_part][axis].values
            coordinates[body_part][j] = DLC_dataframe[DLC_network][body_part][axis].values

        # put all together
        all_body_parts[:, i, :] = coordinates[body_part]
        median_positions = np.nanmedian(all_body_parts, axis=1)
        # median_distance = np.sqrt(median_positions[0,:]**2 + median_positions[1,:]**2)

    for body_part in body_parts:

        # get likelihood
        likelihood = DLC_dataframe[DLC_network][body_part]['likelihood'].values

        # remove coordinates with low confidence
        coordinates[body_part][0][likelihood < .9999999] = np.nan
        coordinates[body_part][1][likelihood < .9999999] = np.nan

        # remove coordinates far from rest of body parts
        distance_from_median_position = np.sqrt( (coordinates[body_part][0] - median_positions[0,:])**2 + (coordinates[body_part][1] - median_positions[1,:])**2 )
        coordinates[body_part][0][distance_from_median_position > 50] = np.nan
        coordinates[body_part][1][distance_from_median_position > 50] = np.nan

        # lineraly interporlate the low-confidence time points
        coordinates[body_part][0] = np.array(pd.Series(coordinates[body_part][0]).interpolate())
        coordinates[body_part][0][0:np.argmin(np.isnan(coordinates[body_part][0]))] = coordinates[body_part][0][
            np.argmin(np.isnan(coordinates[body_part][0]))]

        coordinates[body_part][1] = np.array(pd.Series(coordinates[body_part][1]).interpolate())
        coordinates[body_part][1][0:np.argmin(np.isnan(coordinates[body_part][1]))] = coordinates[body_part][1][
            np.argmin(np.isnan(coordinates[body_part][1]))]

        # convert original coordinates to registered coordinates
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

        # plot the coordinates
        # ax.plot(np.sqrt((coordinates[body_part][0] - 500 * 720 / 1000) ** 2 + (coordinates[body_part][1] - 885 * 720 / 1000) ** 2))
        # plt.pause(.01)

    # compute some metrics
    for i, body_part in enumerate(body_parts):
        all_body_parts[:, i, :] = coordinates[body_part]

    coordinates['head_location'] = np.nanmean(all_body_parts[:, 0:5, :], axis=1)
    coordinates['butt_location'] = np.nanmean(all_body_parts[:, 9:, :], axis=1)
    coordinates['back_location'] = np.nanmean(all_body_parts[:, 6:9, :], axis=1)
    coordinates['center_body_location'] = np.nanmean(all_body_parts[:, 6:, :], axis=1)
    coordinates['center_location'] = np.nanmean(all_body_parts[:, :, :], axis=1)

    delta_position = np.concatenate( ( np.zeros((2,1)), np.diff(coordinates['center_location']) ) , axis = 1)
    coordinates['speed'] = np.sqrt(delta_position[0,:]**2 + delta_position[1,:]**2)

    coordinates['distance_from_shelter'] = np.sqrt((coordinates['center_location'][0] - 500 * 720 / 1000) ** 2 +
                                                   (coordinates['center_location'][1] - 885 * 720 / 1000) ** 2)

    coordinates['speed_toward_shelter'] = np.concatenate( ([0], np.diff(coordinates['distance_from_shelter'])))

    # coordinates['head_butt_distance'] = np.sqrt(np.sum((np.array(head_location) - np.array(butt_location)) ** 2))
    coordinates['body_angle'] = np.angle((coordinates['back_location'][0] - coordinates['butt_location'][0]) + (-coordinates['back_location'][1] + coordinates['butt_location'][1]) * 1j, deg=True)
    coordinates['head_angle'] = np.angle((coordinates['head_location'][0] - coordinates['back_location'][0]) + (-coordinates['head_location'][1] + coordinates['back_location'][1]) * 1j, deg=True)

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
if __name__ == "__main__":
    peri_stimulus_video_clip(vidpath='', videoname='', savepath='', start_frame=0., end_frame=100., stim_frame=0,
                             registration=0, fps=False, save_clip=False, display_clip=False, counter=True,
                             make_flight_image=True)