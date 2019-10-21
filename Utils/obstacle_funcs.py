import cv2
import numpy as np
import os
from termcolor import colored
from Utils.video_funcs import register_frame

def set_up_colors(trial_type):
    '''
    set up colors for DLC analysis
    '''
    # colors depending on trial type / obstacle
    wall_color = np.array([242, 102, 102])
    probe_color = np.array([200, 200, 200])
    no_wall_color = np.array([102, 102, 242])
    if abs(trial_type) == 1:
        flight_color = probe_color
    elif trial_type == 2:
        flight_color = wall_color
    else:
        flight_color = no_wall_color
    flight_color_light = 1 - ( 1 - flight_color / [255, 255, 255] ) / ( np.mean( 1 - flight_color / [255, 255, 255] ) / .05)
    flight_color_dark = 1 - (1 - flight_color / [255, 255, 255]) / ( np.mean(1 - flight_color / [255, 255, 255]) / .45)



    return wall_color, probe_color, no_wall_color, flight_color, flight_color_light, flight_color_dark


def set_up_speed_colors(speed, simulation = False, spontaneous = False, red = False):
    '''
    set up colors for speed-dependent DLC analysis
    '''
    # colors depending on speed
    if red:
        slow_color = np.array([240, 240, 240])
        medium_color = np.array([240, 10, 10])
        fast_color = np.array([220, 220, 0])
        super_fast_color = np.array([232, 0, 0])
    else:
        slow_color = np.array([240, 240, 240])
        medium_color = np.array([190, 190, 240])
        # fast_color = np.array([0, 192, 120])
        fast_color = np.array([0, 232, 120])
        super_fast_color = np.array([0, 232, 0])
    
    # vary color based on speed
    speed_threshold_3 = 20 #118 - not actually used?..
    speed_threshold_2 = 14 #18 #30 #14
    speed_threshold = 7
    # print(speed)
    if speed > speed_threshold_3:
        speed_color = super_fast_color
    elif speed > speed_threshold_2:
        speed_color = ((speed_threshold_3 - speed) * fast_color + (speed - speed_threshold_2) * super_fast_color) / (speed_threshold_3 - speed_threshold_2)
    elif speed > speed_threshold:
        speed_color = ((speed_threshold_2 - speed) * medium_color + (speed - speed_threshold) * fast_color) / (speed_threshold_2 - speed_threshold)
    else:
        speed_color = (speed * medium_color + (speed_threshold - speed) * slow_color) / speed_threshold
        
    speed_color_light = 1 - (1 - speed_color / [255, 255, 255]) / (np.mean(1 - speed_color / [255, 255, 255]) / .08)
    speed_color_dark = 1 - (1 - speed_color / [255, 255, 255]) / (np.mean(1 - speed_color / [255, 255, 255]) / .38)
    speed_color_dark = speed_color_dark**2

    if simulation:
        # speed_color_light, speed_color_dark = np.flip(speed_color_light), np.flip(speed_color_dark)
        speed_color_light, speed_color_dark = np.ones(3) * np.mean(speed_color_light) / 1.2, np.ones(3) * np.mean(speed_color_dark) / 1.2

    if spontaneous:
        speed_color_light = speed_color_light**.6; speed_color_dark = speed_color_dark**.5
        # speed_color_light = speed_color_light**.7; speed_color_dark = speed_color_dark**.6
        if speed < 9: speed_color_light = speed_color_light**.6; speed_color_dark = speed_color_dark**.7
        speed_color_light = speed_color_light[::-1]; speed_color_dark = speed_color_dark[::-1]

    return speed_color_light, speed_color_dark


# create a colorbar for speed
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
#
# scaling_factor = 720 / 100 / 30
# color_bar = np.zeros((80,1,3))
#
# for speed in range(color_bar.shape[0]):
#     pxl_speed = speed * scaling_factor
#     print(pxl_speed)
#     speed_color_light, speed_color_dark = set_up_speed_colors(pxl_speed)
#     color_bar[color_bar.shape[0] - speed - 1, 0, :] = speed_color_dark
#
# color_bar = cv2.cvtColor( cv2.resize( (color_bar*255).astype(np.uint8), (200, 800) ), cv2.COLOR_RGB2BGR)
# cv2.imshow('cb', color_bar)



def get_trial_details(self, stim_frame, trial_num, video_analysis_settings, stim_type, stims_video, previous_vid_duration):
    '''
    Get details like start time and end time for this trial
    '''
    start_frame = int(stim_frame - (video_analysis_settings['seconds pre stimulus'] * self.fps))
    end_frame = int(stim_frame + (video_analysis_settings['seconds post stimulus'] * self.fps))

    videoname = '{}_{}_{}-{} ({}\')'.format(self.session['Metadata'].experiment,
                                                 self.session['Metadata'].mouse_id,
                                                 stim_type, trial_num + 1, round((stim_frame + previous_vid_duration) / self.fps / 60))

    return start_frame, end_frame, videoname

def format_session_video(session_trials_plot_background, width, height, border_size, rectangle_thickness, trial_num, number_of_trials, videoname):
    '''
    Format the session video so that the current trial is selected and the title is right
    '''
    cv2.rectangle(session_trials_plot_background, (0, 0), (width, border_size), 0, -1)

    textsize = cv2.getTextSize(videoname, 0, .55, 1)[0]
    textX = int((width - textsize[0]) / 2)
    cv2.putText(session_trials_plot_background, videoname, (textX, border_size-5), 0, .55, (255,255,255), thickness=1)


    cv2.rectangle(session_trials_plot_background, (int( width + border_size / 4), int((trial_num-1) / number_of_trials * (height+2*border_size/4) + border_size/4)),
                  (int(width + 3*border_size/4), int((trial_num - 1 + 1) / number_of_trials * (height+2*border_size/4))), (0, 0, 0), rectangle_thickness)

    cv2.rectangle(session_trials_plot_background, (int( width + border_size / 4), int(trial_num / number_of_trials * (height+2*border_size/4) + border_size/4)),
                  (int(width + 3*border_size/4), int((trial_num + 1) / number_of_trials * (height+2*border_size/4))), (200, 200, 200), rectangle_thickness)

    return session_trials_plot_background


def get_trial_types(self, vid_num, number_of_vids, stims_video, stims, save_folder, x_offset, y_offset, obstacle_changes, video_analysis_settings, analysis_options, obstacle_type):
    '''
    Takes in a video and stimulus information, and outputs the type of trial (obstacle or none)
    and the background image for DLC trials to be plotted on top of
    '''

    # initialize trial types array
    trial_types = []
    number_of_trials = len(stims[vid_num])

    # set up the image and video that trial type information will modify
    vid = cv2.VideoCapture(self.session['Metadata'].video_file_paths[vid_num][0])
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # for the square arena, just do it based on time
    if obstacle_type == 'side wall 14' or obstacle_type == 'side wall 32':

        trial_types = [2*int(s>20*30*60) for s in stims_video]
        print(trial_types)

        wall_change_frame_store = None

    elif obstacle_type == 'void':

        void_up_mice = ['CA7505', 'CA7492', 'CA7502']
        if np.any([mouse==self.session.Metadata.mouse_id for mouse in void_up_mice]):
            trial_types = [2*int(s<36*30*60) for s in stims_video]
        else:
            trial_types = [2 for s in stims_video]

        wall_change_frame_store = None

    else:

        # If trial types are already saved correctly, just use those
        if ('Trial Types' in self.session['Tracking']):
            try:
                if len(self.session['Tracking']['Trial Types'][vid_num]) == len(stims_video):
                    trial_types = self.session['Tracking']['Trial Types'][vid_num]
            except:
                self.session['Tracking']['Trial Types'] = [[] for x in range(number_of_vids)]
                pass
        else:
            self.session['Tracking']['Trial Types'] = [[] for x in range(number_of_vids)]

        # loop through each trial in the session
        wall_change_frame_store = False
        for trial_num, stim_frame in enumerate(stims_video):

            start_frame = int(stim_frame - (video_analysis_settings['seconds pre stimulus'] * self.fps))
            end_frame = int(stim_frame + (video_analysis_settings['seconds post stimulus'] * self.fps))

            # If trial types depend on the trial, determine which trial is of which type
            if obstacle_changes and len(trial_types) < len(stims_video):
                wall_change_frame, trial_type = initialize_wall_analysis(True, stim_frame, start_frame, end_frame, self.registration,
                                                                              x_offset, y_offset, vid, width, height, trial_types, self.session['Metadata'].experiment)
                if wall_change_frame:
                    wall_change_frame_store = wall_change_frame

                trial_types.append(trial_type)
            # If all trials are the same, just add a 2 (obstacle) to trial type list
            elif len(trial_types) < len(stims_video):
                if obstacle_type == 'none': trial_types.append(0)
                else: trial_types.append(2)

    self.session['Tracking']['Trial Types'][vid_num] = trial_types
    vid.release()

    return trial_types, number_of_trials, wall_change_frame_store, height, width, video_duration, self


def setup_session_video(self, vid_num, stims_video, height, width, stims, save_folder, x_offset, y_offset, obstacle_changes,
                        video_analysis_settings, analysis_options, trial_types, number_of_trials, counter = False):
    '''
    Takes in a video and stimulus information, and outputs the type of trial (obstacle or none)
    and the background image for DLC trials to be plotted on top of
    '''

    # set up colors
    wall_color = np.array([102, 102, 242]) / 3
    probe_color = np.array([200, 200, 200]) / 3
    no_wall_color = np.array([242, 102, 102]) / 3
    wall_color = np.array([132, 132, 242]) / 3
    probe_color = np.array([200, 200, 200]) / 3
    no_wall_color = np.array([242, 132, 132]) / 3
    trial_colors = [probe_color, no_wall_color, probe_color, wall_color]
    border_size = 40
    rectangle_thickness = 3

    # set up the image and video that trial type information will modify
    session_trials_plot_background = np.zeros((height + border_size, width + border_size, 3)).astype(np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    session_videoname = '{}_{}'.format(self.session['Metadata'].experiment, self.session['Metadata'].mouse_id)

    if analysis_options['DLC clips']:
        session_trials_video = cv2.VideoWriter(os.path.join(save_folder, session_videoname + '_session_dlc.avi'),
                                               fourcc, self.fps, (width + border_size * counter, height + border_size * counter), True)
        session_video = cv2.VideoWriter(os.path.join(save_folder, session_videoname + '_session.avi'),
                                        fourcc, self.fps, (width + 2 * border_size * counter, height + 2 * border_size * counter), True*counter)
    else:
        session_trials_video = None; session_video = None

    # loop through each trial in the session
    for trial_num in range(number_of_trials):

        # draw rectangles corresponding to the trials on the right side of the plot
        cv2.rectangle(session_trials_plot_background,
                      (int(width + border_size / 4), int(trial_num / number_of_trials * (height + 2 * border_size / 4) + border_size / 4)),
                      (int(width + 3 * border_size / 4), int((trial_num + 1) / number_of_trials * (height + 2 * border_size / 4))),
                      trial_colors[trial_types[trial_num] + 1], -1)

        cv2.rectangle(session_trials_plot_background,
                      (int(width + border_size / 4), int(trial_num / number_of_trials * (height + 2 * border_size / 4) + border_size / 4)),
                      (int(width + 3 * border_size / 4), int((trial_num + 1) / number_of_trials * (height + 2 * border_size / 4))), (0, 0, 0),
                      rectangle_thickness)

    return session_trials_video, session_video, session_trials_plot_background, border_size, rectangle_thickness, trial_colors


def initialize_wall_analysis(analyze_wall, stim_frame, start_frame, end_frame, registration, x_offset, y_offset, vid, width, height, trial_types, experiment):
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
    y_wall_up_ROI = [int(x * height / 1000) for x in [494 - 30, 504 + 30]]

    # check state of wall on various frames
    frames_to_check = [start_frame, stim_frame - 1, stim_frame + 13, end_frame] # + 13
    wall_darkness = np.zeros((len(frames_to_check), 2))
    for i, frame_to_check in enumerate(frames_to_check):
        vid.set(cv2.CAP_PROP_POS_FRAMES, frame_to_check)
        ret, frame = vid.read()

        frame = register_frame(frame, x_offset, y_offset, registration, map1, map2)

        wall_darkness[i, 0] = sum(
            sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1], x_wall_up_ROI_left[0]:x_wall_up_ROI_left[1], 0] < 150))
        wall_darkness[i, 1] = sum(
            sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1], x_wall_up_ROI_right[0]:x_wall_up_ROI_right[1], 0] < 150))

    wall_darkness_pre = np.min(wall_darkness[0:int(len(frames_to_check) / 2), 0:2])
    wall_darkness_post = np.min(wall_darkness[int(len(frames_to_check) / 2):len(frames_to_check), 0:2])

    # use these darkness levels to detect whether wall is up, down, rising, or falling:
    wall_mouse_already_shown = False
    wall_mouse_show = False
    finished_clip = False
    if 'void' in experiment and 'up' in experiment: experiment = 'void down'

    if (wall_darkness_pre - wall_darkness_post) < -30:
        # print(colored('Wall rising trial!', 'green'))
        wall_height_timecourse = [0]
        trial_type = 1
    elif (wall_darkness_pre - wall_darkness_post) > 30:
        # print(colored('Wall falling trial', 'green'))
        wall_height_timecourse = [1]
        trial_type = -1
    elif 'down' in experiment and -1 in trial_types:
        trial_type = 0
        wall_height_timecourse = 0
    elif 'down' in experiment and not (-1 in trial_types):
        trial_type = 2
        wall_height_timecourse = 1
    elif 'up' in experiment and 1 in trial_types:
        trial_type = 2
        wall_height_timecourse = 1
    elif 'up' in experiment and not (1 in trial_types):
        trial_type = 0
        wall_height_timecourse = 0

    else:
        print('Uh-oh -- not sure what kind of trial!')


    # if its a wall rise or wall fall trial, get the timecourse and the index at which the wall rises or walls
    if trial_type == 1 or trial_type == -1:

        wall_change_frame = False

        for frame_num in range(stim_frame, stim_frame + 13):

            # read the frame
            vid.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = vid.read()
            frame = register_frame(frame, x_offset, y_offset, registration, map1, map2)

            # measure the wall edges
            left_wall_darkness = sum(sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1],
                                         x_wall_up_ROI_left[0]:x_wall_up_ROI_left[1]] < 200))
            right_wall_darkness = sum(sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1],
                                          x_wall_up_ROI_right[0]:x_wall_up_ROI_right[1]] < 200))

            # calculate overall change
            left_wall_darkness_change_overall = left_wall_darkness - wall_darkness[1, 0]
            right_wall_darkness_change_overall = right_wall_darkness - wall_darkness[1, 1]

            # show wall down timecourse
            if trial_type == -1:
                wall_height = round(1 - np.mean(
                    [left_wall_darkness_change_overall / (wall_darkness[2, 0] - wall_darkness[1, 0]),
                     right_wall_darkness_change_overall / (wall_darkness[2, 1] - wall_darkness[1, 1])]), 1)


                # show mouse when wall is down
                if abs(wall_height-1) >= .9: #wall_height <= .1
                    wall_change_frame = frame_num
                    break

            # show wall up timecourse
            if trial_type == 1:
                wall_height = round(np.mean(
                    [left_wall_darkness_change_overall / (wall_darkness[2, 0] - wall_darkness[1, 0]),
                     right_wall_darkness_change_overall / (wall_darkness[2, 1] - wall_darkness[1, 1])]), 1)
                print(wall_height)

                # show mouse when wall is up
                if wall_height >= .6:
                    wall_change_frame = frame_num
                    break
    else:
        wall_change_frame = False

    return wall_change_frame, trial_type



#
# def do_wall_analysis(trial_type, frame, frame_num, stim_frame, fps, x_wall_up_ROI_left, x_wall_up_ROI_right, y_wall_up_ROI, wall_darkness, wall_height_timecourse, wall_mouse_already_shown):
#     '''
#     Calculate how far up for down the obstacle is
#     '''
#     wall_mouse_show = False
#
#     if trial_type and (frame_num - stim_frame) < fps * .5:
#         left_wall_darkness = sum(sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1],
#                                      x_wall_up_ROI_left[0]:x_wall_up_ROI_left[1]] < 200))
#         right_wall_darkness = sum(sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1],
#                                       x_wall_up_ROI_right[0]:x_wall_up_ROI_right[1]] < 200))
#
#         # calculate overall change
#         left_wall_darkness_change_overall = left_wall_darkness - wall_darkness[1, 0]
#         right_wall_darkness_change_overall = right_wall_darkness - wall_darkness[1, 1]
#
#         # show wall up timecourse
#         if trial_type == -1:
#             wall_height = round(1 - np.mean(
#                 [left_wall_darkness_change_overall / (wall_darkness[2, 0] - wall_darkness[1, 0]),
#                  right_wall_darkness_change_overall / (wall_darkness[2, 1] - wall_darkness[1, 1])]),
#                                 1)
#             wall_height_timecourse.append(wall_height)
#
#             # show mouse when wall is down
#             if wall_height <= .1 and not wall_mouse_already_shown:
#                 wall_mouse_show = True
#
#         if trial_type == 1:
#             wall_height = round(np.mean(
#                 [left_wall_darkness_change_overall / (wall_darkness[2, 0] - wall_darkness[1, 0]),
#                  right_wall_darkness_change_overall / (wall_darkness[2, 1] - wall_darkness[1, 1])]),1)
#             wall_height_timecourse.append(wall_height)
#
#             # show mouse when wall is up
#             if wall_height >= .6 and not wall_mouse_already_shown:
#                 wall_mouse_show = True
#
#     return wall_mouse_show, wall_height_timecourse

