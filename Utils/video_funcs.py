# from Utils.imports import *
import cv2
import numpy as np
import scipy.misc
from termcolor import colored
from tqdm import tqdm
import glob
from Utils.registration_funcs import register_frame, model_arena, invert_fisheye_map
from Utils.obstacle_funcs import get_trial_types, initialize_wall_analysis, initialize_wall_analysis, set_up_colors, set_up_speed_colors
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
        cv2.namedWindow('Trial Clip'); cv2.moveWindow('Trial Clip',100,100)
    while True:
        # get the frame
        ret, frame = vid.read()
        if ret:
            # get the frame number
            frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES)

            # apply the fisheye correction
            if [registration]:
                frame = register_frame(frame, x_offset, y_offset, registration, map1, map2)

            # apply the border and count-down
            if counter:
                frame = apply_border_and_countdown(frame, frame_num, stim_frame, start_frame, end_frame, pre_stim_color, post_stim_color, border_size, videoname, fps, width)
            # just use a normal grayscale image instead
            else:
                frame = frame[:,:,0]

            # display the frame
            if display_clip:
                cv2.imshow('Trial Clip', frame)
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


'''


.....................................................................................................



'''




def peri_stimulus_analysis(coordinates, vidpath = '', videoname = '', savepath = '', dlc_config_settings = {}, session_video = None, previous_stim_frame = 0, x_offset = 300, y_offset = 100, obstacle_type = 'wall',
                           session_trials_video = None, session_trials_plot_workspace = None, session_trials_plot_in_background = None,
                           number_of_trials = 10, trial_num = 0, start_frame=0., end_frame=100., stim_frame = 0, trial_type = 1,
                           registration = 0, fps=False, border_size = 40, display_clip = False, counter = True, finished_clip = False):
    '''
    Generate and save peri-stimulus video clip and dlc clip on the idealized arena
    '''
    # Set up video acquisition and saving
    vid, video_clip, width, height, fourcc = set_up_video(vidpath, videoname, fps, savepath, border_size, counter)
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    video_clip_dlc = cv2.VideoWriter(os.path.join(savepath, videoname + '_dlc.avi'), fourcc,fps, (width + border_size, height + border_size), True)

    # set up dlc
    body_parts = dlc_config_settings['body parts']

    # load fisheye mapping
    maps = np.load(registration[3]); map1 = maps[:, :, 0:2]; map2 = maps[:, :, 2] * 0

    # set up border colors
    pre_stim_color = [0, 0, 0,]
    post_stim_color = [200, 200, 200]

    # set up flight path colors
    wall_color, probe_color, no_wall_color, flight_color, flight_color_light, flight_color_dark = set_up_colors(trial_type)

    # set up model arena
    arena, _, shelter_roi = model_arena((height, width), trial_type*(trial_type-1), False, obstacle_type)
    arena = cv2.cvtColor(arena, cv2.COLOR_GRAY2RGB)
    arena_fresh = arena.copy()

    # initialize images for model mouse to go onto
    model_flight_image = arena.copy()                                       # Single-trial image, with speed coded as color
    model_mouse_mask_previous = np.zeros(arena.shape[0:2]).astype(np.uint8)
    model_mouse_mask_initial = np.zeros(arena.shape[0:2]).astype(np.uint8)
    model_flight_in_background = session_trials_plot_in_background.copy()
    frames_past_stimulus = 0
    arrived_at_shelter = False

    # set up session trials plot
    if session_trials_plot_workspace is None:
        session_trials_plot = arena.copy()
        session_trials_plot_workspace = arena.copy()
    else:
        session_trials_plot = ((session_trials_plot_workspace.astype(float) * 1 + arena.copy().astype(float) * 4) / 5).astype(np.uint8)

    # set up pre-stimulus session trials plot
    initial_session_trials_plot = session_trials_plot.copy()
    initial_session_trials_plot_workspace = session_trials_plot_workspace.copy()

    # create windows to display clip
    if display_clip:
        cv2.namedWindow('Trial Clip'); cv2.moveWindow('Trial Clip', 100, 100)
        cv2.namedWindow('Session Flight'); cv2.moveWindow('Session Flight', 1000, 100)

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
                model_flight_image = arena_fresh.copy()
                session_trials_plot = initial_session_trials_plot.copy()
                model_mouse_mask_previous = 0

            # at stimulus onset, refresh model arena is the obstacle comes up for down
            elif frame_num == stim_frame and abs(trial_type) == 1:
                current_model_arena, _, _ = model_arena(frame.shape[0:2], trial_type > 0, False, obstacle_type)
                model_flight_image = cv2.cvtColor(current_model_arena, cv2.COLOR_GRAY2RGB)

            # extract DLC coordinates from the saved coordinates dictionary
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
            speed = coordinates['speed'][frame_num-1]
            speed_color_light, speed_color_dark = set_up_speed_colors(speed)

            # set scale for size of model mouse
            back_butt_dist = 16

            # draw ellipses representing model mouse
            model_mouse_mask = cv2.ellipse(model_mouse_mask_initial.copy() , head_location,(int(back_butt_dist*.6), int(back_butt_dist* .3)), 180 - head_angle, 0, 360, 100, thickness=-1)
            model_mouse_mask = cv2.ellipse(model_mouse_mask, front_location, (int(back_butt_dist * .5), int(back_butt_dist * .33)), 180 - neck_angle, 0,360, 100, thickness=-1)
            model_mouse_mask = cv2.ellipse(model_mouse_mask, nack_location, (int(back_butt_dist * .7), int(back_butt_dist * .35)), 180 - nack_angle, 0, 360, 100,thickness=-1)
            model_mouse_mask = cv2.ellipse(model_mouse_mask , shoulder_location, (int(back_butt_dist), int(back_butt_dist*.44)), 180 - shoulder_angle ,0, 360, 100, thickness=-1)
            model_mouse_mask = cv2.ellipse(model_mouse_mask , body_location, (int(back_butt_dist*.9), int(back_butt_dist*.5) ), 180 - body_angle, 0, 360, 100, thickness=-1)

            # make a single large ellipse used to determine when do use the flight_color_dark
            mouse_mask = cv2.ellipse(model_mouse_mask_initial.copy(), body_location, (int(back_butt_dist*2), back_butt_dist), 180 - shoulder_angle, 0, 360, 100, thickness=-1)

            # keep count of frames past stimulus
            frames_past_stimulus = frame_num - stim_frame

            # stop after 10 secs
            if frames_past_stimulus > 10*fps or arrived_at_shelter:
                pass

            # add dark mouse if distant from previous mouse, using the above mouse mask
            elif np.sum(mouse_mask * model_mouse_mask_previous) == 0:
                # add dark mouse to both the speed-trial plot and the session-trials plot and the session-trials-workspace plot
                model_flight_image[model_mouse_mask.astype(bool)] = model_flight_image[model_mouse_mask.astype(bool)] * speed_color_dark
                session_trials_plot[model_mouse_mask.astype(bool)] = session_trials_plot[model_mouse_mask.astype(bool)] * flight_color_dark
                if frames_past_stimulus > 0:
                    session_trials_plot_workspace[model_mouse_mask.astype(bool)] = session_trials_plot_workspace[model_mouse_mask.astype(bool)] * flight_color_dark

                # set the current model mouse mask as the one to be compared to to see if dark mouse should be added
                model_mouse_mask_previous = model_mouse_mask

            # continuous shading, after stimulus onset
            elif frames_past_stimulus > 0:
                # once at shelter, end it
                if np.sum(shelter_roi * model_mouse_mask) > 500:
                    arrived_at_shelter = True
                # otherwise, shade in the current mouse position
                else:
                    model_flight_image[model_mouse_mask.astype(bool)] = model_flight_image[model_mouse_mask.astype(bool)] * speed_color_light
                    session_trials_plot[model_mouse_mask.astype(bool)] = session_trials_plot[model_mouse_mask.astype(bool)] * flight_color_light
                    session_trials_plot_workspace[model_mouse_mask.astype(bool)] = session_trials_plot_workspace[model_mouse_mask.astype(bool)] * flight_color_light

            # get contour of initial ellipse at stimulation time, to apply to images
            if frame_num == stim_frame:
                _, contours, _ = cv2.findContours(model_mouse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # redraw this contour on each frame after the stimulus
            elif frame_num >= stim_frame:
                cv2.drawContours(model_flight_image, contours, 0, (0,0,0), 2)
                cv2.drawContours(model_flight_image, contours, 0, (255,255,255), 1)

                cv2.drawContours(session_trials_plot, contours, 0, tuple([int(x) for x in flight_color*.7]), thickness = 3)
                cv2.drawContours(session_trials_plot, contours, 0, (255,255,255), thickness = 1)

            # draw a dot for each body part, to verify DLC tracking
            # for bp in body_parts:
            #     session_trials_plot = cv2.circle(session_trials_plot, tuple(coordinates[bp][:2, frame_num - 1].astype(np.uint16)), 1, (0, 250,0), -1)

            # place the DLC images within a background that includes a title and indication of current trial
            session_trials_plot_in_background[border_size:, 0:-border_size] = cv2.cvtColor(session_trials_plot, cv2.COLOR_BGR2RGB) #session_trials_plot
            model_flight_in_background[border_size:, 0:-border_size,:] = model_flight_image

            # apply the border and count-down
            if counter:
                frame = apply_border_and_countdown(frame, frame_num, stim_frame, start_frame, end_frame, pre_stim_color, post_stim_color, border_size, videoname, fps, width)
            # just use a normal grayscale image instead
            else:
                frame = frame[:, :, 0]

            # display current frames
            if display_clip:
                cv2.imshow('Trial Clip', frame); cv2.imshow('Session Flight', session_trials_plot_in_background); cv2.imshow('Trial Flight', model_flight_in_background)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # write current frame to video
            video_clip.write(frame); session_video.write(frame); video_clip_dlc.write(model_flight_in_background); session_trials_video.write(session_trials_plot_in_background)

            # end video
            if frame_num >= end_frame:
                finished_clip = True
                break
        else:
            print('Problem with movie playback'); cv2.waitKey(1000); break

    # wrap up
    vid.release()
    video_clip.release()
    video_clip_dlc.release()

    # save trial images
    if finished_clip:
        scipy.misc.imsave(os.path.join(savepath, videoname + '_dlc.tif'), cv2.cvtColor(model_flight_in_background, cv2.COLOR_BGR2RGB))
        scipy.misc.imsave(os.path.join(savepath, videoname + '_dlc_history.tif'), cv2.cvtColor(session_trials_plot_in_background, cv2.COLOR_BGR2RGB))

    # draw contours on the session trials workspace plot
    cv2.drawContours(session_trials_plot_workspace, contours, 0, tuple([int(x) for x in flight_color * .7]), thickness=3)
    cv2.drawContours(session_trials_plot_workspace, contours, 0, (255, 255, 255), thickness=1)

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






