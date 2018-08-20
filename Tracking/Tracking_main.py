# Import packages
import cv2
import numpy as np
import imageio
imageio.plugins.ffmpeg.download()


# Import functions and params from other scripts
from Tracking import dlc_analyseVideos
from Tracking.Tracking_functions import get_body_orientation, get_mvmt_direction, get_velocity
from Tracking.Tracking_utils import *
from Utils.Custom_funcs import cut_crop_video
from Utils import Data_rearrange_funcs
from Utils.utils_classes import Trial
from Utils.loadsave_funcs import load_yaml

from Config import startf, exp_type, track_options

# Import from DeepLabCut
# from AnalyzeVideos import getpose
# from nnet import predict

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


class Tracking():
    def __init__(self, session, database):
        # params for contour extraction
        if exp_type == 'maze':
            self.arena_floor = False
        else:
            self.arena_floor = session['Video']['User ROIs']['Tsk']

        # Load settings from config
        cfg = load_yaml(track_options['cfg_std'])
        dlc_config_settings = load_yaml(track_options['cfg_dlc'])
        dlc_config_settings['clips'] = {'visual': {}, 'audio': {}, 'digital': {}}

        self.stopframe = cfg['stopframe']
        self.fps = session['Video']['Frame rate']
        self.num_exp_cnts = cfg['num mice']
        self.preview = cfg['preview']
        self.threshold_scaling = cfg['th_scaling']  # Used in get_contours to get correct threhsolding
        self.tail_threshold_scaling = cfg['tail_th_scaling']
        self.min_cnt_area = cfg['min ctn area']
        self.max_cnt_area = cfg['max cnt area']
        self.fast_track_wnd_s = cfg['fast track wnd']

        # params from showing tracing results in preview
        self.trace_length = 10
        self.magnif_factor = 2

        # params for data saving
        self.coord_l = []

        #### NOW TRACK ####
        # tracking data stores the results of the tracking for the whole session, the exploration phase and the
        # individual frames
        if not isinstance(session['Tracking'], dict):
            session['Tracking'] = {}

        # Check if we want to track during the exploration only
        if track_options['track_exploration']:
            if track_options['track whole session']:
                # No need to track the exploration, we will be tracking the whole session anyway so we can extract
                # the exploration data from there
                pass
            else:
                print('     ... tracking Exploration')
                all_stims = session['Metadata'].stimuli.values()
                all_stims = [item for sublist in all_stims for item in sublist]
                all_stims = [item for sublist in all_stims for item in sublist]
                first_stim = min(all_stims)
                if first_stim:
                    start_frame = 0
                    stop_frame = first_stim-1
                else:
                    start_frame = 0
                    stop_frame = -1
                tracked = self.tracking(session['Video']['Background'], 0,
                                        start_frame=start_frame, stop_frame=stop_frame, video_fps=self.fps)
                session['Tracking']['Exploration'] = tracked

        # Check if tracking the whole session
        if track_options['track whole session']:
            print('     ... tracking the whole session')
            for idx, vid in enumerate(session['Metadata'].video_file_path):
                if idx == 0:
                    start_frame = startf
                else:
                    start_frame = 0
                tracked = self.tracking(session['Video']['Background'], vid,
                                        start_frame=start_frame, stop_frame=self.stopframe, video_fps=self.fps)
                session['Tracking']['Whole Session'] = tracked

                if track_options['track_exploration']:
                    print('Need to write a function to extract the exploration data from the whole session data')
                    pass

        if track_options['track_mouse_fast']:  # Track the individual trials
            print('     ... tracking individual trials')
            # Process only chunks of videos around the trials
            for stim_type, stims in session['Metadata'].stimuli.items():  # For each stim type get the list of stim frames
                if not stims:
                    continue

                for vid_num, stims_video in enumerate(stims):  # Loop over each video in the session
                    for idx, stim in enumerate(stims_video):  # Loop over each stim for each video
                        self.videoname = '{}-{}_{}-{}'.format(session['Metadata'].session_id, stim_type, vid_num, idx)
                        start_frame = int(stim-(self.fast_track_wnd_s*self.fps[0]))
                        stop_frame = int(stim+(self.fast_track_wnd_s*self.fps[0]))

                        # Generate empty trial object and put it into the database
                        trial_metadata = create_trial_metadata(self.videoname, stim_type, start_frame, stop_frame,
                                              session['Metadata'].video_file_paths[vid_num])
                        empty_trial = Trial()
                        empty_trial.metadata = trial_metadata
                        empty_trial.name = trial_metadata['Name']
                        session['Tracking'][empty_trial.metadata['Name']] = empty_trial

                        # STD TRACKING
                        if track_options['use_stdtracking']:
                            print('     ... processing trial {}'.format(self.videoname))

                            trial = self.tracking(session['Video']['Background'],
                                                  session['Metadata'].video_file_paths[vid_num],
                                                  start_frame=start_frame, stop_frame=stop_frame, video_fps=self.fps)

                            trial = Data_rearrange_funcs.restructure_trial_data(trial, start_frame, stop_frame,
                                                                                stim_type, idx, vid_num)
                            trial.metadata = trial_metadata

                            # Merge into database
                            old_trial = session['Tracking'][trial.metadata['Name']]
                            session['Tracking'][trial.metadata['Name']] = merge_std_dlc_trials(old_trial, trial)

                        # DLC TRACKING
                        if track_options['use_deeplabcut']:
                            # set up
                            start_sec = start_frame * (1 / session['Video']['Frame rate'][vid_num])
                            stop_sec = stop_frame * (1 / session['Video']['Frame rate'][vid_num])

                            # Extract trial clip and store it so that we can save all trials at the same time
                            trial_clip = cut_crop_video(session['Metadata'].video_file_paths[vid_num],
                                                        cut=True, starts=start_sec, fins=stop_sec,
                                                        save_format=None, ret=True)
                            dlc_config_settings['clips'][stim_type][self.videoname] = trial_clip

            # Process trial clips if we are using dlc_tracking
            if track_options['use_deeplabcut']:
                if dlc_config_settings['clips']['visual'] or dlc_config_settings['clips']['audio']:
                    print('        ... extracting trials video clips')
                    save_trial_clips(dlc_config_settings['clips'], dlc_config_settings['clips_folder'])

                    print('        ... extracting pose from clips')
                    dlc_analyseVideos.analyse(dlc_config_settings['clips_folder'])

                    print('        ... integrating results in database')
                    database = dlc_retreive_data(dlc_config_settings['clips_folder'], database)

                    print('        ... cleaning up')
                    if not dlc_config_settings['store trial videos']:
                        dlc_clear_folder(dlc_config_settings['clips_folder'], dlc_config_settings['store trial videos'])

        self.database = database

########################################################################################################################
########################################################################################################################

    def tracking(self, bg, video_path, start_frame=1, stop_frame=-1, video_fps=30, justCoM=True):
        # Create video capture
        cap = cv2.VideoCapture(video_path)
        ret, firstframe = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        if not ret:
            print("Something went wrong, couldn't read video file")
            return

        # Initialise empty arrays to store tracking data and other variables
        video_duration_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        setattr(self, 'x', np.zeros((1, video_duration_frames)))
        setattr(self, 'y', np.zeros((1, video_duration_frames)))
        setattr(self, 'orientation', np.zeros((1, video_duration_frames)))
        setattr(self, 'velocity', np.zeros((1, video_duration_frames)))
        setattr(self, 'direction', np.zeros((1, video_duration_frames)))

        # Display background
        if self.preview:
            cv2.namedWindow('bg')
            cv2.imshow('bg', bg)
            cv2.moveWindow('bg', 3200, 0)
            cv2.waitKey(1)

        # Start tracing
        f = start_frame  # keep track of frame number
        while True:
            # Check that we can proceed with the analysis
            f += 1
            if f % 1000 == 0:
                if not stop_frame == -1:
                    print('             ... processing frame {} of {}'.format(f, stop_frame))
                else:
                    print('             ... processing frame {} of {}'.format(f, video_duration_frames))

            if not stop_frame == -1 and f > stop_frame:
                return self

            ret, frame = cap.read()
            if not ret:
                return self

            # get contours from frame
            display, cnt = get_contours(bg, frame, self.threshold_scaling)

            # extract info from contour
            if cnt:
                (x, y), radius = cv2.minEnclosingCircle(cnt[0])
                centeroid = (int(x), int(y))
                self.coord_l.append(centeroid)
                self.x[0, f - 1] = centeroid[0]
                self.y[0, f - 1] = centeroid[1]

                # draw contours and trace and ROI
                if self.preview:
                    cv2.drawContours(frame, cnt, -1, (0, 255, 0), 1)
                    drawtrace(frame, self.coord_l, (255, 0, 0), self.trace_length)

                if not justCoM:
                    # get mouse orientation
                    orientation = get_body_orientation(f, cnt[0], bg, display, frame, start_frame, self.orientation,
                                                       self.arena_floor, self.tail_threshold_scaling)

                    # Get mouse velocity
                    velocity = get_velocity(video_fps, self.coord_l)

                    # Get direction of movement
                    direction = get_mvmt_direction(self.coord_l)

                    self.orientation[0, f - 1] = orientation
                    self.velocity[0, f - 1] = velocity
                    self.direction[0, f - 1] = direction

            if self.preview:
                display_results(f, frame, display, self.magnif_factor, self)

            #  Control framerate and exit
            # need to adjust this so that it waits for the correct amount to match fps
            key = cv2.waitKey(10) & 0xFF






