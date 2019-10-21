from Utils.video_funcs import peri_stimulus_analysis, peri_stimulus_video_clip
from Utils.dlc_funcs import extract_dlc, filter_and_transform_dlc, compute_pose_from_dlc
from Utils.registration_funcs import get_arena_details, model_arena, get_background
from Utils.obstacle_funcs import get_trial_types, initialize_wall_analysis, get_trial_details, format_session_video, setup_session_video
from Utils.strategy_funcs import  spontaneous_homings, exploration, procedural_learning
from Utils.simulations import simulate
from Utils.loadsave_funcs import save_data
from Utils.proper_analysis import proper_analysis
import cv2
import scipy
import numpy as np
import os
from termcolor import colored
import dill as pickle
from Config import tracking_options, dlc_options
track_options, analysis_options, fisheye_map_location, video_analysis_settings = tracking_options()
dlc_config_settings = dlc_options()
if track_options['run DLC']:
    from deeplabcut.pose_estimation_tensorflow import analyze_videos
    from deeplabcut.utils import create_labeled_video



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
        self.x_offset, self.y_offset, self.obstacle_type, self.shelter_location, self.subgoal_location, self.obstacle_changes \
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

        # Check for arena registration
        if track_options['register arena']:
            self.registration = self.session['Registration']
        else:
            self.registration = []

        # get file paths
        for vid_num, video_path in enumerate(self.session['Metadata'].video_file_paths):
            self.video_path = video_path[0]
            self.processed_coordinates_file = os.path.join(os.path.dirname(self.video_path), 'coordinates')

            # run the video through DLC network
            if track_options['run DLC']:
                analyze_videos(dlc_config_settings['config_file'], self.video_path)
                # create_labeled_video(dlc_config_settings['config_file'], [self.video_path])

            # Track the session using DLC (extract data that was computed in the previous step)
            if track_options['track session']:
                self.track_session(vid_num, track = True)

        # Analyze each trial
        if any(x for x in analysis_options.values()):
            self.analyze_trials()


    def track_session(self, vid_num, track = False):
        '''
        ................................PERFORM TRACKING................................
        '''
        # find file path
        self.video_path = self.session['Metadata'].video_file_paths[vid_num][0]
        self.processed_coordinates_file = os.path.join(os.path.dirname(self.video_path), 'coordinates')

        # open saved coordinates if they exist
        if os.path.isfile( self.processed_coordinates_file ) and not track:
            print(colored(' - Already extracted DLC coordinates', 'green'))
            with open(self.processed_coordinates_file, "rb") as dill_file:
                self.coordinates = pickle.load(dill_file)

        # otherwise, extract the coordinates
        else:
            print(colored(' - Extracting DLC coordinates from ' + self.video_path, 'green'))
            self.coordinates = extract_dlc(dlc_config_settings, self.video_path)

            # process the extracted coordinates
            self.process_tracking_data()


    def process_tracking_data(self):
        '''
        ................................PROCESS TRACKING DATA................................
        '''
        # analyze the newly opened coordinates, if the processed coordinates haven't been saved yet
        print(colored(' - Processing DLC coordinates', 'green'))

        # filter coordinates and transform them to the common coordinate space
        self.coordinates = filter_and_transform_dlc(dlc_config_settings, self.coordinates, self.x_offset, self.y_offset, self.registration,
                                                    plot = False, filter_kernel = 7) #21

        # compute speed, angles, and pose from coordinates
        self.coordinates = compute_pose_from_dlc(dlc_config_settings['body parts'], self.coordinates, self.shelter_location,
                                                 self.session['Registration'][4][0], self.session['Registration'][4][1], self.subgoal_location)

        # save the processed coordinates to the video folder
        with open(self.processed_coordinates_file, "wb") as dill_file:
            pickle.dump(self.coordinates, dill_file)

        # set up the tracking dict
        self.session['Tracking']['coordinates'] = self.processed_coordinates_file #self.session['Tracking'] = {};


    def analyze_trials(self):
        '''
        ................................ANALYZE, DISPLAY, AND SAVE EACH TRIAL................................
        '''
        print(colored(' - Analyzing trials', 'green'))

        # Initialize a couple of variables
        session_trials_plot_workspace = None
        trial_types = []; number_of_trials = 0; trials_completed = 0

        # Loop over each stimulus
        for stim_type, stims in self.session['Stimuli'].stimuli.items():

            # Only use stimulus modalities that were used during the session
            if not stims[0]: # and not stims[1]:
                continue
            if trials_completed:
                break

            # Initialize list for multi-video analysis
            # self.session['Tracking']['Trial Types'] = [[],[]]
            video_durations = []; previous_vid_duration = 0

            # Loop over each video in the session
            for vid_num, stims_video in enumerate(stims):

                # Save to a folder named after the experiment and the mouse
                save_folder = os.path.join(dlc_config_settings['clips_folder'], self.session['Metadata'].experiment, str(self.session['Metadata'].mouse_id))
                if not os.path.isdir(save_folder):
                    os.makedirs(save_folder)

                # Get the trial types and number of trials
                trial_types, trials_in_video, wall_change_frame, height, width, video_duration, self = get_trial_types(self, vid_num, len(stims), stims_video,
                                   stims, save_folder, self.x_offset, self.y_offset, self.obstacle_changes, video_analysis_settings, analysis_options, self.obstacle_type)
                video_durations.append(video_duration)
                number_of_trials += trials_in_video

            # Get the trial types and initiate the session video and image
            trial_types = sum(self.session['Tracking']['Trial Types'],[])
            session_trials_video, session_video, session_trials_plot_background, border_size, rectangle_thickness, trial_colors = \
                setup_session_video(self, vid_num, stims_video, height, width, stims, save_folder, self.x_offset, self.y_offset,
                                self.obstacle_changes, video_analysis_settings, analysis_options, trial_types, number_of_trials)

            # Loop over each video in the session
            for vid_num, stims_video in enumerate(stims):

                if not stims_video:
                    continue
                print(trial_types)

                # get the dlc data
                self.track_session(vid_num)

                # set up needed analysis for simulations
                if (analysis_options['simulate'] or analysis_options['exploration']) and not ('start_index' in self.coordinates):
                    analysis_options['procedural'] = True
                if analysis_options['procedural']:
                    self.coordinates['start_index'] = np.array([]);
                    self.coordinates['end_index'] = np.array([]);
                if analysis_options['exploration']:
                    self.coordinates['SR'] = [[] for t in range(len(trial_types)) ]

                # Loop over each stim for each video
                previous_stim_frame = 0

                # fourcc = cv2.VideoWriter_fourcc(*"XVID")
                # vid = cv2.VideoWriter(os.path.join(save_folder, 'simulation.avi'), fourcc, self.fps,(width, height), True)

                for trial_num, stim_frame in enumerate(stims_video):

                    if trial_num > 2: break
                    if not trial_num: continue

                    if str(self.session['Metadata'].mouse_id) == 'CA6161':
                       if (stim_frame / 60 / 30) < 19:
                           continue

                    if str(self.session['Metadata'].mouse_id) == 'CA6131':
                       if (stim_frame / 60 / 30) < 7:
                           continue

                    if str(self.session['Metadata'].mouse_id) == 'CA6091':
                        if (stim_frame / 60 / 30) < 19 or (stim_frame / 60 / 30) > 27:
                            continue

                    # if not trial_num in [1,2,3,5]:
                    #     continue
                    # else: trials_completed = trial_num
                    # trial_types[0] = 2

                    # get the trial details
                    start_frame, end_frame, self.videoname = get_trial_details(self, stim_frame, trials_completed, video_analysis_settings, stim_type, sum(stims,[]), previous_vid_duration)
                    self.arena, _, _ = model_arena((height, width), trial_types[trials_completed], False, self.obstacle_type)

                    # format the session video for this trial
                    session_trials_plot_background = format_session_video(session_trials_plot_background, width, height, border_size, rectangle_thickness,
                                                                          trials_completed, number_of_trials, self.videoname)

                    # analyze spontaneous homings
                    if analysis_options['spontaneous homings'] or analysis_options['procedural']:
                        try: homing_arena
                        except: homing_arena = cv2.cvtColor(self.arena.copy(), cv2.COLOR_GRAY2RGB)

                        session_trials_plot_background = session_trials_plot_background * 0 #TEMPORARY
                        homing_arena, trial_groups = spontaneous_homings(homing_arena, session_trials_plot_background, border_size, self.coordinates,
                                                       previous_stim_frame, stim_frame, self.videoname, save_folder, self.subgoal_location, self.obstacle_type, width, height, make_vid = False)

                    # analyze procedural learning
                    if analysis_options['procedural']:
                        try:
                            procedural_arena
                        except:
                            if not ('start_index' in self.coordinates): self.coordinates['start_index'] = np.array([]); self.coordinates['end_index'] = np.array([]);
                            procedural_arena = cv2.cvtColor(self.arena.copy(), cv2.COLOR_GRAY2RGB)

                        procedural_arena, group_idx, distance_from_start, end_idx = procedural_learning(procedural_arena, session_trials_plot_background, border_size, self.coordinates,
                                                       previous_stim_frame, stim_frame, self.videoname, save_folder, self.subgoal_location, trial_groups, self.obstacle_type, self.shelter_location)

                        if len(self.coordinates['start_index']) < stim_frame:
                            self.coordinates['start_index'] = np.append(self.coordinates['start_index'], group_idx)
                            self.coordinates['end_index']   = np.append(self.coordinates['end_index'], end_idx)

                    # do simulations
                    if analysis_options['simulate']:
                        simulate(self.coordinates, stim_frame, self.shelter_location, self.arena, self.obstacle_type,
                                 self.subgoal_location, self.coordinates['start_index'], self.coordinates['end_index'],
                                 trial_types[trials_completed], stims_video, self.videoname, save_folder, session_trials_plot_background,
                                 border_size, strategy = 'all')

                    # analyze exploration
                    if analysis_options['exploration']:
                        session_trials_plot_background = session_trials_plot_background * 0  # TEMPORARY
                        exploration(session_trials_plot_background, border_size, self.coordinates,
                                                        previous_stim_frame, stim_frame, self.videoname, save_folder, self.arena)
                        # self.coordinates['SR'][trial_num] = successor_from_stimulus

                    # do analysis and video saving
                    if analysis_options['DLC clips']:

                        # FOR CONTROL (TEMPORARY): TAKE RANDOM POINT WHEN IN STIM ZONE (extreme y values)
                        # x_position = self.coordinates['center_location'][1].copy()
                        # stim_idx = np.array([], 'int')
                        # for st in stims_video:
                        #     stim_idx = np.append(stim_idx, np.arange(st - 300, st + 300))
                        # x_position[stim_idx] = np.nan; x_position[:300] = np.nan
                        # eligible_frames = np.where( (x_position > 540) + (x_position < 180) )[0]
                        # time_chase = stim_frame - np.random.choice(eligible_frames)
                        # start_frame, end_frame, stim_frame = start_frame - time_chase, end_frame - time_chase, stim_frame - time_chase

                        # do analysis and video saving
                        session_trials_plot_workspace = peri_stimulus_analysis(
                                               self.coordinates, self.session['Metadata'].video_file_paths[vid_num][0],
                                               self.videoname, save_folder, dlc_config_settings, session_video,
                                               self.x_offset, self.y_offset, self.obstacle_type, wall_change_frame, session_trials_video,
                                               session_trials_plot_workspace, session_trials_plot_background*0, number_of_trials, trial_num,
                                               start_frame, end_frame, stim_frame, trial_types[trials_completed],
                                               self.registration, self.fps, border_size, display_clip=True, counter=False)

                    # do basic video saving
                    elif analysis_options['raw clips']:
                        peri_stimulus_video_clip(self.video_path, self.videoname, save_folder, start_frame, end_frame, stim_frame,
                                                 self.registration, self.x_offset, self.y_offset, self.fps, display_clip=True, counter=True)

                    trials_completed += 1
                    previous_stim_frame = stim_frame



                previous_vid_duration += video_durations[vid_num]

                # vid.release()
                # save the processed coordinates to the video folder
                if analysis_options['procedural'] or analysis_options['exploration']:
                    with open(self.processed_coordinates_file, "wb") as dill_file:
                        pickle.dump(self.coordinates, dill_file)


                # do traditional analysis
                # FOR CONTROL (TEMPORARY): TAKE RANDOM POINT WHEN IN STIM ZONE (extreme y values)
                x_position = self.coordinates['center_location'][1].copy()
                stim_idx = np.array([], 'int')
                for st in stims_video:
                    stim_idx = np.append(stim_idx, np.arange(st - 300, st + 300))
                x_position[stim_idx] = np.nan;
                x_position[:300] = np.nan; x_position[-300:] = np.nan
                eligible_frames = np.where((x_position > 540) + (x_position < 180))[0]

                stims_video_control = []
                for st in stims_video:
                    stims_video_control.append(np.random.choice(eligible_frames))
                stims_video_control.sort()

                if analysis_options['proper analysis']:
                    proper_analysis(self, trial_types, stims_video, trial_colors, height, width, dlc_config_settings['clips_folder'],
                                    self.obstacle_type, self.shelter_location, stims_video_control = stims_video_control, control = False)





