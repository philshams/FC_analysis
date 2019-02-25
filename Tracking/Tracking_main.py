from Utils.video_funcs import peri_stimulus_analysis, peri_stimulus_video_clip
from Utils.dlc_funcs import extract_dlc, filter_and_transform_dlc, compute_pose_from_dlc
from Utils.registration_funcs import get_arena_details
from Utils.obstacle_funcs import get_trial_types, initialize_wall_analysis, get_trial_details, format_session_video

import cv2
import scipy
import numpy as np
import os
from termcolor import colored
from Config import tracking_options, dlc_options
track_options, analysis_options, fisheye_map_location, video_analysis_settings = tracking_options()
dlc_config_settings = dlc_options()
if track_options['run DLC']:
    from deeplabcut.pose_estimation_tensorflow import analyze_videos
    # from deeplabcut.utils import create_labeled_video



class Tracking():
    """
    ................................SET UP TRACKING................................
    """
    def __init__(self, session):
        # set up class variables
        self.session = session

        # get frame rate
        self.fps = self.session['Metadata'].videodata[0]['Frame rate'][0]

        # Determine arena type
        self.x_offset, self.y_offset, self.obstacle_type, self.shelter_location, self.obstacle_changes \
            = get_arena_details(self.session['Metadata'].experiment)

        # do tracking
        self.main()


    def main(self):
        """
        ................................CONTROL TRACKING................................
        """
        # Create Tracking entry into session if not present
        if not isinstance(self.session['Tracking'], dict):
            self.session['Tracking'] = {}

        # Save raw video clips without any analysis
        if analysis_options['save stimulus clips'] and not analysis_options['DLC clips']:
            self.analyze_trials()

        # Check for arena registration
        if track_options['register arena']:
            self.registration = self.session['Registration']
        else:
            self.registration = []

        # run the video through DLC network
        self.video_path = self.session.Metadata.video_file_paths[0][0]
        if track_options['run DLC']:
            analyze_videos(dlc_config_settings['config_file'], self.video_path)

        # Track the session using DLC (extract data that was computed in the previous step)
        if track_options['track session'] or analysis_options['DLC clips'] or analysis_options['sub-goal'] or analysis_options['planning'] or analysis_options['procedural']:
            self.track_session()

        # Process tracking data into usable format
        if analysis_options['DLC clips'] or analysis_options['sub-goal'] or analysis_options['planning'] or analysis_options['procedural']:
            self.process_tracking_data()

            # Analyze each trial
            self.analyze_trials()


    def track_session(self):
        '''
        ................................PERFORM TRACKING................................
        '''
        if (dlc_config_settings['body parts'][0] in self.session.Tracking):
            # carry on to analysis if tracking has already been done
            print(colored(' - Already extracted DLC coordinates', 'green'))
        else:
            # get the original behaviour video
            print(colored(' - Tracking the whole session from ' + self.video_path, 'green'))

            # extract coordinates from DLC
            self.session['Tracking'] = extract_dlc(dlc_config_settings, self.video_path)


    def process_tracking_data(self):
        '''
        ................................PROCESS TRACKING DATA................................
        '''
        # filter coordinates and transform them to the common coordinate space
        print(colored(' - Processing DLC coordinates', 'green'))
        self.coordinates = filter_and_transform_dlc(dlc_config_settings, self.session['Tracking'], self.x_offset, self.y_offset, self.registration,
                                                    plot = False, filter_kernel = 21)

        # compute speed, angles, and pose from coordinates
        self.coordinates = compute_pose_from_dlc(dlc_config_settings['body parts'], self.coordinates, self.shelter_location,
                                                 self.session['Registration'][4][0], self.session['Registration'][4][1])



    def analyze_trials(self):
        '''
        ................................ANALYZE, DISPLAY, AND SAVE EACH TRIAL................................
        '''
        print(colored(' - Saving trial clips', 'green'))

        # Initialize a couple of variables
        session_trials_plot_workspace = None

        # Loop over each stimulus
        for stim_type, stims in self.session['Stimuli'].stimuli.items():

            # Only use stimulus modalities that were used during the session
            if not stims[0]:
                continue

            # Loop over each video in the session
            for vid_num, stims_video in enumerate(stims):

                # Save to a folder named after the experiment and the mouse
                save_folder = os.path.join(dlc_config_settings['clips_folder'], self.session['Metadata'].experiment, str(self.session['Metadata'].mouse_id))
                if not os.path.isdir(save_folder):
                    os.makedirs(save_folder)

                # Get the trial types and initiate the session video and image
                if analysis_options['DLC clips'] or True:
                    self.session['Tracking']['Trial Types'], session_trials_video, session_video, session_trials_plot_background, \
                    number_of_trials, height, width, border_size, rectangle_thickness = \
                        get_trial_types(self, vid_num, stims_video, stims, save_folder, self.x_offset, self.y_offset, self.obstacle_changes, video_analysis_settings)

                # Loop over each stim for each video
                for trial_num, stim_frame in enumerate(stims_video):

                    # get the trial details
                    start_frame, end_frame, previous_stim_frame, self.videoname = get_trial_details(self, stim_frame, trial_num, video_analysis_settings, stim_type, stims_video)

                    # analyze sub-goals
                    if analysis_options['sub-goal']:
                        pass

                    # analyze planning
                    if analysis_options['planning']:
                        pass

                    # analyze procedural learning
                    if analysis_options['procedural']:
                        pass

                    # do analysis and video saving
                    if analysis_options['DLC clips']:

                        # format the session video for this trial
                        session_trials_plot_background = format_session_video(session_trials_plot_background, width, height, border_size, rectangle_thickness, trial_num, number_of_trials, self.videoname)

                        # do analysis and video saving
                        session_trials_plot_workspace = peri_stimulus_analysis(
                                               self.coordinates, self.session['Metadata'].video_file_paths[vid_num][0],
                                               self.videoname, save_folder, dlc_config_settings, session_video, previous_stim_frame,
                                               self.x_offset, self.y_offset, self.obstacle_type, session_trials_video,
                                               session_trials_plot_workspace, session_trials_plot_background, number_of_trials, trial_num,
                                               start_frame, end_frame, stim_frame, self.session['Tracking']['Trial Types'][trial_num],
                                               self.registration, self.fps, border_size, display_clip=True, counter=True)

                    # do basic video saving
                    elif analysis_options['save stimulus clips']:
                        peri_stimulus_video_clip(self.video_path, self.videoname, save_folder, start_frame, end_frame, stim_frame,
                                                 self.registration, self.x_offset, self.y_offset, self.fps, display_clip=True, counter=False)

