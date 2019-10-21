# from Utils.imports import *
import cv2
import numpy as np
import scipy.misc
from termcolor import colored
from tqdm import tqdm
import glob
from Utils.registration_funcs import register_frame, model_arena, invert_fisheye_map
# from Utils.obstacle_funcs import set_up_colors, set_up_speed_colors #TEMPORARY ...?
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd


def set_up_video(vidpath, videoname, fps, savepath, border_size, counter):
    '''
    initialize video acquisition and saving for peri-stim analysis
    '''
    # open the behaviour video
    vid = cv2.VideoCapture(vidpath)

    # find the frame rate, width, and height
    if not fps: fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # set up the trial clip for saving
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    video_clip = cv2.VideoWriter(os.path.join(savepath,videoname+'.avi'), fourcc, fps, (width+2*border_size*counter, height+2*border_size*counter), counter)

    return vid, video_clip, width, height, fourcc




def register_frame(frame, x_offset, y_offset, registration, map1, map2):
    '''
    ..........................GO FROM A RAW TO A REGISTERED FRAME................................
    '''
    frame_register = frame[:, :, 0]

    frame_register = cv2.copyMakeBorder(frame_register, y_offset,
                                        int((map1.shape[0] - frame.shape[0]) - y_offset),
                                        x_offset, int((map1.shape[1] - frame.shape[1]) - x_offset),
                                        cv2.BORDER_CONSTANT, value=0)
    frame_register = cv2.remap(frame_register, map1, map2, interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    frame_register = frame_register[y_offset:-int((map1.shape[0] - frame.shape[0]) - y_offset),
                     x_offset:-int((map1.shape[1] - frame.shape[1]) - x_offset)]



    frame = cv2.cvtColor(cv2.warpAffine(frame_register, registration[0], frame.shape[0:2]),cv2.COLOR_GRAY2RGB)

    return frame



def apply_border_and_countdown(frame, frame_num, stim_frame, start_frame, end_frame, pre_stim_color, post_stim_color, border_size, videoname, fps, width):
    '''
    Apply a border, title, and countdown to an otherwise raw peri-stimulus behaviour video
    '''


    # get the color and time sign prior to stimulus onset
    if frame_num < stim_frame:
        cur_color = tuple([x * ((frame_num - start_frame) / (stim_frame - start_frame)) for x in pre_stim_color])
        sign = ''
    # get the color and time sign after stimulus onset
    else:
        cur_color = tuple([x * (1 - (frame_num - stim_frame) / (end_frame - stim_frame)) for x in post_stim_color])
        sign = '+'

    # colored border around frame
    frame = cv2.copyMakeBorder(frame, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=cur_color)

    # title the video
    textsize = cv2.getTextSize(videoname, 0, .55, 1)[0]
    textX = int((width + 2 * border_size - textsize[0]) / 2)
    cv2.rectangle(frame, (0, 0), (width + 2 * border_size, border_size), 0, -1)
    cv2.putText(frame, videoname, (textX, int(border_size * 3 / 4)), 0, .55, (255, 255, 255), thickness=1)

    # report time relative to stimulus onset
    frame_time = (frame_num - stim_frame) / fps
    frame_time = str(round(.1 * round(frame_time / .1), 1)) + '0' * (abs(float(frame_time)) < 10)
    cv2.putText(frame, sign + str(frame_time) + 's', (width + border_size - 50, int(border_size * 3 / 4)), 0, .65, (180, 180, 180), thickness=2)

    return frame



def peri_stimulus_video_clip(vidpath = '', videoname = '', savepath = '', start_frame=0., end_frame=100., stim_frame = 0,
                   registration = 0, x_offset = 300, y_offset = 100, fps=False, display_clip = False, counter = True):
    '''
    Generate and save peri-stimulus video clip
    '''
    # set up border colors
    pre_stim_color = [0, 0, 0, ]
    post_stim_color = [200, 200, 200]
    border_size = 40

    # Set up video acquisition and saving
    vid, video_clip, width, height, _ = set_up_video(vidpath, videoname, fps, savepath, border_size, counter)
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # set up fisheye correction
    if registration:
        maps = np.load(registration[3]); map1 = maps[:, :, 0:2]; map2 = maps[:, :, 2] * 0

    # run peri-stimulus video
    if display_clip:
        cv2.namedWindow(videoname); cv2.moveWindow(videoname,100,100)
    while True:
        # get the frame
        ret, frame = vid.read()
        if ret:
            # get the frame number
            frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES)

            # apply the fisheye correction
            if registration:
                frame = register_frame(frame, x_offset, y_offset, registration, map1, map2)

            # apply the border and count-down
            if counter:
                frame = apply_border_and_countdown(frame, frame_num, stim_frame, start_frame, end_frame, pre_stim_color, post_stim_color, border_size, videoname, fps, width)
            # just use a normal grayscale image instead
            else:
                frame = frame[:,:,0]

            # display the frame
            if display_clip:
                cv2.imshow(videoname, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # write the frame to video
            video_clip.write(frame)

            # end the video
            if frame_num >= end_frame:
                break
        else:
            print('Problem with movie playback'); cv2.waitKey(1000); break
    # wrap up
    vid.release()
    video_clip.release()





'''


....................the all-important video analysis function is below.................................................................................



'''




def peri_stimulus_analysis(coordinates, vidpath = '', videoname = '', savepath = '', dlc_config_settings = {}, session_video = None, x_offset = 300, y_offset = 100, obstacle_type = 'wall',
                           wall_change_frame = 0, session_trials_video = None, session_trials_plot_workspace = None, session_trials_plot_in_background = None,
                           number_of_trials = 10, trial_num = 0, start_frame=0., end_frame=100., stim_frame = 0, trial_type = 1,
                           registration = 0, fps=False, border_size = 40, display_clip = False, counter = True, finished_clip = False):
    '''
    Generate and save peri-stimulus video clip and dlc clip on the idealized arena
    '''
    # Set up video acquisition and saving
    vid, video_clip, width, height, fourcc = set_up_video(vidpath, videoname, fps, savepath, border_size, counter)
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    video_clip_dlc = cv2.VideoWriter(os.path.join(savepath, videoname + '_dlc.avi'), fourcc,fps, (width + border_size*counter, height + border_size*counter), True)

    # set up dlc
    body_parts = dlc_config_settings['body parts']

    # load fisheye mapping
    maps = np.load(registration[3]); map1 = maps[:, :, 0:2]; map2 = maps[:, :, 2] * 0

    # set up border colors
    pre_stim_color = [0, 0, 0,]
    post_stim_color = [200, 200, 200]

    # set up flight path colors
    wall_color, probe_color, no_wall_color, flight_color, _, _ = set_up_colors(trial_type)

    # set up model arena
    arena, _, shelter_roi = model_arena((height, width), trial_type > 0, False, obstacle_type, shelter = not 'no shelter' in videoname) #*(trial_type-1)
    arena = cv2.cvtColor(arena, cv2.COLOR_GRAY2RGB)
    arena_fresh = arena.copy()
    trial_plot = arena.copy()

    # initialize images for model mouse to go onto
    # model_flight_image = arena.copy()                                       # Single-trial image, with speed coded as color
    model_mouse_mask_previous = np.zeros(arena.shape[0:2]).astype(np.uint8)
    model_mouse_mask_initial = np.zeros(arena.shape[0:2]).astype(np.uint8)
    # model_flight_in_background = session_trials_plot_in_background.copy()
    frames_past_stimulus = 0
    arrived_at_shelter = False
    count_down = np.inf

    # set up session trials plot
    if session_trials_plot_workspace is None: #REMOVE THIS TO ADD DLC HISTORY TO BACKGORUND
        session_trials_plot = arena.copy()
        session_trials_plot_workspace = arena.copy()
#    else:
        # session_trials_plot = ((session_trials_plot_workspace.astype(float) * 1 + arena.copy().astype(float) * 4) / 5).astype(np.uint8)
        # session_trials_plot = ((session_trials_plot_workspace.astype(float) * 1 + arena.copy().astype(float) * 1) / 2).astype(np.uint8)
    if trial_type<=0 and 'down' in videoname and False: #False for Tiago
        old_arena, _, _ = model_arena((height, width), True, False, obstacle_type, shelter = not 'no shelter' in videoname)
        old_arena = cv2.cvtColor(old_arena, cv2.COLOR_GRAY2RGB)

        discrepancy = ~((arena - old_arena)==0)

        session_trials_plot_workspace[discrepancy] = ((old_arena[discrepancy] * 1 + arena[discrepancy].astype(float) * 4) / 5).astype(np.uint8)
        trial_plot[discrepancy] =                    ((old_arena[discrepancy] * 1 + arena[discrepancy].astype(float) * 4) / 5).astype(np.uint8)

    session_trials_plot = session_trials_plot_workspace.copy()  # TEMPORARY

    # set up pre-stimulus session trials plot
    initial_session_trials_plot = session_trials_plot.copy()
    initial_session_trials_plot_workspace = session_trials_plot_workspace.copy()

    # smooth speed trace for coloration
    smoothed_speed = np.concatenate((np.zeros(6 - 1), np.convolve(coordinates['speed'], np.ones(12), mode='valid'), np.zeros(6))) / 12

    # create windows to display clip
    if display_clip:
        cv2.namedWindow(savepath); cv2.moveWindow(savepath, 100, 100)
        cv2.namedWindow(savepath + ' DLC'); cv2.moveWindow(savepath + ' DLC', 1000, 100)

    # loop over each frame
    while True:
        # get the frame
        ret, frame = vid.read()
        if ret:
            #get the frame number
            frame_num = int(vid.get(cv2.CAP_PROP_POS_FRAMES))

            # apply the fisheye correction
            frame = register_frame(frame, x_offset, y_offset, registration, map1, map2)

            # prior to stimulus onset, refresh frame to initialized frame
            if frame_num < stim_frame:
                # model_flight_image = arena_fresh.copy()
                session_trials_plot = initial_session_trials_plot.copy()
                model_mouse_mask_previous = 0

            # at stimulus onset, refresh model arena is the obstacle comes up for down
            # elif frame_num == stim_frame and abs(trial_type) == 1:
            #     current_model_arena, _, _ = model_arena(frame.shape[0:2], trial_type > 0, False, obstacle_type)
            #     model_flight_image = cv2.cvtColor(current_model_arena, cv2.COLOR_GRAY2RGB)

            # extract DLC coordinates from the saved coordinates dictionary]
            body_angle = coordinates['body_angle'][frame_num - 1]
            shoulder_angle = coordinates['shoulder_angle'][frame_num - 1]
            head_angle = coordinates['head_angle'][frame_num - 1]
            neck_angle = coordinates['neck_angle'][frame_num - 1]
            nack_angle = coordinates['nack_angle'][frame_num - 1]
            head_location = tuple(coordinates['head_location'][:,frame_num-1].astype(np.uint16))
            nack_location = tuple(coordinates['nack_location'][:, frame_num - 1].astype(np.uint16))
            front_location = tuple(coordinates['front_location'][:, frame_num - 1].astype(np.uint16))
            shoulder_location = tuple(coordinates['shoulder_location'][:, frame_num - 1].astype(np.uint16))
            body_location = tuple(coordinates['center_body_location'][:, frame_num - 1].astype(np.uint16))

            # extract speed and use this to determine model mouse coloration
            # speed = coordinates['speed'][frame_num-1]
            speed = smoothed_speed[frame_num - 1]
            speed_color_light, speed_color_dark = set_up_speed_colors(speed)

            # set scale for size of model mouse
            back_butt_dist = 16

            # when turning, adjust relative sizes
            if abs(body_angle - shoulder_angle) > 35:
                shoulder = False
            else:
                shoulder = True

            # draw ellipses representing model mouse
            model_mouse_mask = cv2.ellipse(model_mouse_mask_initial.copy() , body_location, (int(back_butt_dist*.9), int(back_butt_dist*.5) ), 180 - body_angle, 0, 360, 100, thickness=-1)
            model_mouse_mask = cv2.ellipse(model_mouse_mask, nack_location, (int(back_butt_dist * .7), int(back_butt_dist * .35)), 180 - nack_angle, 0, 360,100, thickness=-1)
            model_mouse_mask = cv2.ellipse(model_mouse_mask, head_location, (int(back_butt_dist * .6), int(back_butt_dist * .3)), 180 - head_angle, 0, 360, 100,thickness=-1)
            model_mouse_mask = cv2.ellipse(model_mouse_mask, front_location, (int(back_butt_dist * .5), int(back_butt_dist * .33)), 180 - neck_angle, 0, 360,100, thickness=-1)

            if shoulder:
                model_mouse_mask = cv2.ellipse(model_mouse_mask, shoulder_location, (int(back_butt_dist), int(back_butt_dist * .4)), 180 - shoulder_angle, 0,360, 100, thickness=-1)


            # make a single large ellipse used to determine when do use the flight_color_dark
            mouse_mask = cv2.ellipse(model_mouse_mask_initial.copy(), body_location, (int(back_butt_dist*2.5), int(back_butt_dist*1.5)), 180 - shoulder_angle, 0, 360, 100, thickness=-1)

            # keep count of frames past stimulus
            frames_past_stimulus = frame_num - stim_frame
            frames_til_abort = count_down - frame_num

            # get contour of initial ellipse at stimulation time, to apply to images
            if frame_num == stim_frame:
                _, contours, _ = cv2.findContours(model_mouse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                x_stim = float(body_location[0]); y_stim = float(body_location[1])

            # stop after 2 secs of shelter
            if not frames_til_abort:
                finished_clip = True
                break

            # add dark mouse if distant from previous mouse, using the above mouse mask
            elif np.sum(mouse_mask * model_mouse_mask_previous) == 0 and not arrived_at_shelter and frames_past_stimulus:
                # add dark mouse to the session-trials plot and the session-trials-workspace plot
                session_trials_plot[model_mouse_mask.astype(bool)] = session_trials_plot[model_mouse_mask.astype(bool)] * speed_color_dark
                if frames_past_stimulus > 0:
                    dist_from_start = np.sqrt((x_stim - float(body_location[0]))**2 + (y_stim - float(body_location[1]))**2)
                    dist_to_make_red = 150
                    prev_homing_color = np.array([.98, .98, .98]) + np.max((0,dist_to_make_red - dist_from_start))/dist_to_make_red * np.array([.02, -.02, -.02])

                    session_trials_plot_workspace[model_mouse_mask.astype(bool)] = session_trials_plot_workspace[model_mouse_mask.astype(bool)] * prev_homing_color#[.9, .9, .9]
                    trial_plot[model_mouse_mask.astype(bool)] = trial_plot[model_mouse_mask.astype(bool)] * speed_color_dark

                # set the current model mouse mask as the one to be compared to to see if dark mouse should be added
                model_mouse_mask_previous = model_mouse_mask

            # continuous shading, after stimulus onset
            elif frames_past_stimulus >= 0:
                # once at shelter, end it
                if not 'no shelter' in videoname:
                    if np.sum(shelter_roi * mouse_mask) > 0: #temporary
                        if not arrived_at_shelter:
                            arrived_at_shelter = True
                            count_down = frame_num + fps * 2

                    # otherwise, shade in the current mouse position
                    elif not arrived_at_shelter:
                        dist_from_start = np.sqrt((x_stim - float(body_location[0])) ** 2 + (y_stim - float(body_location[1])) ** 2)
                        dist_to_make_red = 150
                        prev_homing_color = np.array([.98, .98, .98]) + np.max((0, dist_to_make_red - dist_from_start)) / dist_to_make_red * np.array(
                            [.02, -.02, -.02])

                        session_trials_plot[model_mouse_mask.astype(bool)] = session_trials_plot[model_mouse_mask.astype(bool)] * speed_color_light
                        session_trials_plot_workspace[model_mouse_mask.astype(bool)] = session_trials_plot_workspace[model_mouse_mask.astype(bool)] * prev_homing_color #* [.9, .9, .9]
                        trial_plot[model_mouse_mask.astype(bool)] = trial_plot[model_mouse_mask.astype(bool)] * speed_color_light




            # redraw this contour on each frame after the stimulus
            elif frame_num >= stim_frame:
                cv2.drawContours(session_trials_plot, contours, 0, (255,255,255), thickness = 1)

            # if wall falls or rises, color mouse in RED
            if (trial_type==1 or trial_type==-1) and (frame_num == wall_change_frame) and False:
                _, contours_wall_change, _ = cv2.findContours(model_mouse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(session_trials_plot, contours_wall_change, 0, (220,0,0), thickness=-1)
                cv2.drawContours(trial_plot, contours_wall_change, 0, (220,0,0), thickness=-1)

            # draw a dot for each body part, to verify DLC tracking
            # for bp in body_parts:
            #     session_trials_plot = cv2.circle(session_trials_plot, tuple(coordinates[bp][:2, frame_num - 1].astype(np.uint16)), 1, (200, 0, 0), -1)


            # place the DLC images within a background that includes a title and indication of current trial
            session_trials_plot_in_background[border_size:, 0:-border_size] = cv2.cvtColor(session_trials_plot, cv2.COLOR_BGR2RGB)

            # apply the border and count-down
            if counter:
                frame = apply_border_and_countdown(frame, frame_num, stim_frame, start_frame, end_frame, pre_stim_color, post_stim_color, border_size, savepath, fps, width)
            # just use a normal grayscale image instead
            else:
                frame = frame[:, :, 0]

            # add a looming spot - for actual loom
            # loom_radius = 30 * (frame_num - stim_frame) * ( (frame_num - stim_frame) < 10) * (frame_num > stim_frame)
            loom_on = np.tile(np.concatenate((np.ones(5), np.zeros(5))), 9).astype(int)


            if obstacle_type == 'wall' or obstacle_type == 'void':
                shape = 'circle'
                size = 340 #334

            elif obstacle_type == 'side wall':
                shape = 'square'
                size = 224 #218

            # elif obstacle_type == 'void':
            #     shape = 'circle'
            #     size = 353 #347

            elif obstacle_type == 'side wall 32' or obstacle_type == 'side wall 14':
                shape = 'square'
                size = 295
            else:
                shape = 'circle'
                size = 340  # 334
                loom_on = np.tile(np.concatenate((np.ones(5), np.zeros(5))), 9*3).astype(int)


            if frame_num < stim_frame or frame_num > stim_frame+89: loom_radius = 0 #89 for 3 secs, 269 for 9 secs
            else: loom_radius = size * loom_on[frame_num - stim_frame] #218 for planning #334 for barnes #347 for void

            session_trials_plot_show = session_trials_plot.copy()
            trial_plot_show = trial_plot.copy()
            if loom_radius:
                frame = frame.copy()
                loom_frame = frame.copy()
                loom_arena = session_trials_plot.copy()
                loom_arena2 = trial_plot.copy()

                # for actual loom
                # stimulus_location = tuple(coordinates['center_body_location'][:, stim_frame - 1].astype(np.uint16))
                # cv2.circle(loom_frame, stimulus_location, loom_radius, 100, -1)
                # cv2.circle(loom_arena, stimulus_location, loom_radius, (100,100,100), -1)

                center = (int(frame.shape[1] / 2), int(frame.shape[0] / 2))
                if shape == 'circle':
                    cv2.circle(loom_frame, center, loom_radius, 200, 12)
                    cv2.circle(loom_arena, center, loom_radius, (200, 200, 200), 12)
                    cv2.circle(loom_arena2, center, loom_radius, (200, 200, 200), 12)

                elif shape == 'square':
                    cv2.rectangle(loom_frame, tuple([c+loom_radius for c in center]), tuple([c-loom_radius for c in center]), 200, 12)
                    cv2.rectangle(loom_arena,  tuple([c+loom_radius for c in center]), tuple([c-loom_radius for c in center]), (200, 200, 200), 12)
                    cv2.rectangle(loom_arena2,  tuple([c+loom_radius for c in center]), tuple([c-loom_radius for c in center]), (200, 200, 200), 12)

                alpha = .3
                cv2.addWeighted(frame, alpha, loom_frame, 1 - alpha, 0, frame)
                cv2.addWeighted(session_trials_plot, alpha, loom_arena, 1 - alpha, 0, session_trials_plot_show)
                cv2.addWeighted(trial_plot, alpha, loom_arena, 1 - alpha, 0, trial_plot_show)

            session_trials_plot_show = cv2.cvtColor(session_trials_plot_show, cv2.COLOR_BGR2RGB)
            trial_plot_show = cv2.cvtColor(trial_plot_show, cv2.COLOR_BGR2RGB)

            # put minute of stimulation in clip
            stim_minute = str(int(np.round(stim_frame / 60 / 30))) + "'"
            frame = frame.copy()
            cv2.putText(frame, stim_minute, (20, 50), 0, 1, (255, 255, 255), thickness=2)

            # display current frames
            if display_clip:
                cv2.imshow(savepath, frame);
                cv2.imshow('hi', session_trials_plot_workspace);
                if counter: cv2.imshow(savepath + ' DLC', session_trials_plot_in_background);
                else: cv2.imshow(savepath + ' DLC', session_trials_plot_show);
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # write current frame to video
            video_clip.write(frame); session_video.write(frame)
            if counter:
                video_clip_dlc.write(session_trials_plot_in_background); session_trials_video.write(session_trials_plot_in_background)
            else:
                video_clip_dlc.write(session_trials_plot_show); session_trials_video.write(session_trials_plot_show)
            # end video
            if frame_num >= end_frame:
                finished_clip = True
                break
        else:
            print('Problem with movie playback'); cv2.waitKey(1000); break

    # wrap up
    vid.release(); video_clip.release(); video_clip_dlc.release()

    # save trial images
    if finished_clip:
        scipy.misc.imsave(os.path.join(savepath, videoname + '_dlc_history.tif'), cv2.cvtColor(session_trials_plot_in_background, cv2.COLOR_BGR2RGB))

        # format and save trial plot
        # cv2.drawContours(trial_plot, contours, 0, (0, 0, 0), thickness=2)
        cv2.drawContours(trial_plot_show, contours, 0, (255, 255, 255), thickness=1)

        # scipy.misc.imsave(os.path.join(savepath, videoname + '_dlc_pure.tif'), trial_plot)

        trial_plot_in_background = session_trials_plot_in_background.copy()
        trial_plot_in_background[border_size:, 0:-border_size] = trial_plot_show
        scipy.misc.imsave(os.path.join(savepath, videoname + '_dlc.tif'), cv2.cvtColor(trial_plot_in_background, cv2.COLOR_BGR2RGB))


    # draw contours on the session trials workspace plot
    # cv2.drawContours(session_trials_plot_workspace, contours, 0, (0, 0, 0), thickness=2)
    # cv2.drawContours(session_trials_plot_workspace, contours, 0, (0, 0, 0), thickness=1)

    cv2.drawContours(session_trials_plot_workspace, contours, 0, (220, 0, 0), thickness=-1)
    cv2.drawContours(session_trials_plot_workspace, contours, 0, (0,0,0), thickness=1)

    scipy.misc.imsave(os.path.join(savepath, videoname + '_history.tif'), session_trials_plot_workspace)

    # after the last trial, save the session workspace image
    if trial_num == (number_of_trials - 1):
        # make the trials less dark
        session_trials_plot_workspace = ((session_trials_plot_workspace.astype(float) * 2 + arena.copy().astype(float) * 2) / 4).astype(np.uint8)

        # add the session workspace image to the session-trials video
        session_trials_plot_in_background[border_size:, 0:-border_size] = session_trials_plot_workspace
        session_trials_video.write(session_trials_plot_workspace)

        # save the session workspace image
        scipy.misc.imsave(os.path.join(savepath, videoname + '_session_trials.tif'), session_trials_plot_workspace)

        # wrap up
        session_trials_video.release()
        session_video.release()

    return session_trials_plot_workspace