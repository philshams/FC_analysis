import cv2
import numpy as np
import os
from termcolor import colored

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


def set_up_speed_colors(speed):
    '''
    set up colors for speed-dependent DLC analysis
    '''
    # colors depending on speed
    slow_color = np.array([240, 240, 240])
    medium_color = np.array([190, 190, 240])
    fast_color = np.array([0, 192, 120])
    super_fast_color = np.array([0, 232, 0])
    
    # vary color based on speed
    speed_threshold_3 = 18
    speed_threshold_2 = 14
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

    return speed_color_light, speed_color_dark


def get_trial_details(self, stim_frame, trial_num, video_analysis_settings, stim_type, stims_video):
    '''
    Get details like start time and end time for this trial
    '''
    start_frame = int(stim_frame - (video_analysis_settings['seconds pre stimulus'] * self.fps))
    end_frame = int(stim_frame + (video_analysis_settings['seconds post stimulus'] * self.fps))
    if trial_num:
        previous_stim_frame = stims_video[trial_num - 1]
    else:
        previous_stim_frame = 0

    videoname = '{}_{}_{}-{} ({}\')'.format(self.session['Metadata'].experiment,
                                                 self.session['Metadata'].mouse_id,
                                                 stim_type, trial_num + 1, round(stim_frame / self.fps / 60))

    return start_frame, end_frame, previous_stim_frame, videoname

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


def get_trial_types(self, vid_num, stims_video, stims, save_folder, x_offset, y_offset, obstacle_changes, video_analysis_settings, analysis_options):
    '''
    Takes in a video and stimulus information, and outputs the type of trial (obstacle or none)
    and the background image for DLC trials to be plotted on top of
    '''
    # initialize trial types array
    trial_types = []
    number_of_trials = len(stims[0])

    # set up colors
    wall_color = np.array([102, 102, 242]) / 1.6
    probe_color = np.array([200, 200, 200]) / 1.6
    no_wall_color = np.array([242, 102, 102]) / 1.6
    trial_colors = [probe_color, no_wall_color, probe_color, wall_color]
    border_size = 40
    rectangle_thickness = 3

    # set up the image and video that trial type information will modify
    vid = cv2.VideoCapture(self.session['Metadata'].video_file_paths[vid_num][0])
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    session_trials_plot_background = np.zeros((height + border_size, width + border_size, 3)).astype(np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    session_videoname = '{}_{}'.format(self.session['Metadata'].experiment, self.session['Metadata'].mouse_id, )

    if analysis_options['DLC clips']:
        session_trials_video = cv2.VideoWriter(os.path.join(save_folder, session_videoname + '_session_dlc.avi'),
                                               fourcc, self.fps, (width + border_size, height + border_size), True)
        session_video = cv2.VideoWriter(os.path.join(save_folder, session_videoname + '_session.avi'),
                                        fourcc, self.fps, (width + 2 * border_size, height + 2 * border_size), True)
    else:
        session_trials_video = None; session_video = None

    # If trial types are already saved correctly, just use those
    if ('Trial Types' in self.session['Tracking']):
        if len(self.session['Tracking']) == len(stims_video):
            trial_types = self.session['Tracking']['Trial Types']

    # loop through each trial in the session
    for trial_num, stim_frame in enumerate(stims_video):

        start_frame = int(stim_frame - (video_analysis_settings['seconds pre stimulus'] * self.fps))
        end_frame = int(stim_frame + (video_analysis_settings['seconds post stimulus'] * self.fps))

        # If trial types depend on the trial, determine which trial is of which type
        if obstacle_changes and len(trial_types) < len(stims_video):
            _, trial_type, _, _, _, _, _, _, _ = initialize_wall_analysis(True, stim_frame, start_frame, end_frame, self.registration,
                                                                          x_offset, y_offset, vid, width, height)
            trial_types.append(trial_type)
        # If all trials are the same, just add a 2 (obstacle) to trial type list
        elif len(trial_types) < len(stims_video):
            trial_types.append(2)

        # draw rectangles corresponding to the trials on the right side of the plot
        cv2.rectangle(session_trials_plot_background,
                      (int(width + border_size / 4), int(trial_num / number_of_trials * (height + 2 * border_size / 4) + border_size / 4)),
                      (int(width + 3 * border_size / 4), int((trial_num + 1) / number_of_trials * (height + 2 * border_size / 4))),
                      trial_colors[trial_types[trial_num] + 1], -1)

        cv2.rectangle(session_trials_plot_background,
                      (int(width + border_size / 4), int(trial_num / number_of_trials * (height + 2 * border_size / 4) + border_size / 4)),
                      (int(width + 3 * border_size / 4), int((trial_num + 1) / number_of_trials * (height + 2 * border_size / 4))), (0, 0, 0),
                      rectangle_thickness)

    vid.release()
    return trial_types, session_trials_video, session_video, session_trials_plot_background, number_of_trials, height, width, border_size, rectangle_thickness, trial_colors



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
        # print(colored('Wall rising trial!', 'green'))
        wall_height_timecourse = [0]
        trial_type = 1
    elif (wall_darkness_pre - wall_darkness_post) > 30:
        # print(colored('Wall falling trial', 'green'))
        wall_height_timecourse = [1]
        trial_type = -1
    elif (wall_darkness_pre > 85) and (wall_darkness_post > 85):
        # print(colored('Obstacle trial', 'green'))
        trial_type = 2
        wall_height_timecourse = 1  # [1 for x in list(range(int(fps * .5)))]
    elif (wall_darkness_pre < 85) and (wall_darkness_post < 85):
        # print(colored('No obstacle trial', 'green'))
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
    '''
    Calculate how far up for down the obstacle is
    '''
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

