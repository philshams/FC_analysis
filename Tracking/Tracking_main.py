# Import packages
import pandas as pd
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

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


class Tracking():
    """"
    Tracks position or posture from video data based on the user selected parameters
    The results are saved in self.database
    """
    def __init__(self, session, database, TF_setup, TF_settings, clips_l):
        # set up class variables
        self.session = session
        self.database = database

        # TF set up -- check if TF is running already
        self.TF_setup = TF_setup
        self.TF_settings = TF_settings
        self.clips_l = clips_l

        # Load settings from config
        self.cfg = load_yaml(track_options['cfg_std'])
        self.dlc_config_settings = load_yaml(track_options['cfg_dlc'])
        self.dlc_config_settings['clips'] = {'visual': {}, 'audio': {}, 'digital': {}}

        # params for contour extraction
        self.arena_floor = False

        # params from showing tracing results in preview
        self.trace_length = 10
        self.magnif_factor = 2

        # params for data saving
        self.coord_l = []

        #### NOW TRACK ####
        self.main()

    def main(self):
        """
        Call functions to track using either the std or the dlc tracking.
        """
        # Create Tracking entry into session if not present
        if not isinstance(self.session['Tracking'], dict):
            self.session['Tracking'] = {}

        # Track exploration
        if track_options['track_exploration']:
            self.track_exploration()

        # Track whole session
        if track_options['track whole session']:
            self.track_wholesession()

        # Track single trials
        if track_options['track_mouse_fast']:
            self.track_trials()

    def track_exploration(self):
        if track_options['track whole session']:
            # No need to track the exploration, we will be tracking the whole session anyway so we can extract
            # the exploration data from there
            pass
        else:
            print('     ... tracking Exploration')
            all_stims = self.session['Metadata'].stimuli.values()
            all_stims = [item for sublist in all_stims for item in sublist]
            all_stims = [item for sublist in all_stims for item in sublist]
            first_stim = min(all_stims)
            if first_stim:
                start_frame = 0
                stop_frame = first_stim-1
            else:
                start_frame = 0
                stop_frame = -1
            tracked = self.tracking(self.session['Video']['Background'], 0,
                                    start_frame=start_frame, stop_frame=stop_frame, video_fps=self.fps)
            self.session['Tracking']['Exploration'] = tracked.data

    def track_wholesession(self):
        # Check if tracking the whole session
        if track_options['track whole session']:
            print('     ... tracking the whole session')
            for idx, vid in enumerate(self.session['Metadata'].video_file_path):
                if idx == 0:
                    start_frame = startf
                else:
                    start_frame = 0
                tracked = self.tracking(self.session['Video']['Background'], vid,
                                        start_frame=start_frame, stop_frame=self.stopframe, video_fps=self.fps)
                self.session['Tracking']['Whole Session'] = tracked.data

                if track_options['track_exploration']:
                    print('Need to write a function to extract the exploration data from the whole session data')
                    raise ValueError('This functionality is not implemented yet')

    def track_trials(self):
        cfg = self.cfg
        print('     ... tracking individual trials')
        for stim_type, stims in self.session['Metadata'].stimuli.items():
            # For each stim type get the list of stim frames
            if not stims:
                continue

            for vid_num, stims_video in enumerate(stims):  # Loop over each video in the session
                for idx, stim in enumerate(stims_video):  # Loop over each stim for each video
                    # SET UP - ######################################
                    self.videoname = '{}-{}_{}-{}'.format(self.session['Metadata'].session_id, stim_type, vid_num, idx)
                    fps = self.session['Video']['Frame rate'][vid_num]
                    start_frame = int(stim-(cfg['fast track wnd']*fps))
                    stop_frame = int(stim+(cfg['fast track wnd']*fps))

                    # Generate empty trial object and add it into the database
                    trial_metadata = create_trial_metadata(self.videoname, stim_type, start_frame, stop_frame,
                                                           self.session['Metadata'].video_file_paths[vid_num])
                    empty_trial = Trial()
                    empty_trial.metadata = trial_metadata
                    empty_trial.name = trial_metadata['Name']
                    self.session['Tracking'][empty_trial.metadata['Name']] = empty_trial

                    # STD TRACKING - ######################################
                    """
                    For STD tracking extract the position of the mouse contour using self.tracking()
                    """
                    if track_options['use_stdtracking']:
                        print('     ... processing trial {} - Standard tracking'.format(self.videoname))

                        trial = self.tracking(self.session['Video']['Background'],
                                              self.session['Metadata'].video_file_paths[vid_num],
                                              start_frame=start_frame, stop_frame=stop_frame, video_fps=fps)

                        trial = Data_rearrange_funcs.restructure_trial_data(trial, stim_type, idx, vid_num)
                        trial.metadata = trial_metadata

                        # Merge into database
                        old_trial = self.session['Tracking'][trial.metadata['Name']]
                        self.session['Tracking'][trial.metadata['Name']] = merge_std_dlc_trials(old_trial, trial)

                    # DLC TRACKING - PREPARE CLIPS - ######################################
                    """
                    for DLC tracking, extract videoclips for the peri-stimulus time which will be analysed using
                    DeepLabCut in a second moment
                    """
                    if track_options['use_deeplabcut']:
                        # set up
                        start_sec = start_frame * (1 / fps)
                        stop_sec = stop_frame * (1 / fps)

                        # Extract trial clip and store it so that we can save all trials at the same time
                        trial_clip = cut_crop_video(self.session['Metadata'].video_file_paths[vid_num],
                                                    cut=True, starts=start_sec, fins=stop_sec,
                                                    save_format=None, ret=True)
                        self.dlc_config_settings['clips'][stim_type][self.videoname] = trial_clip

        # DLC TRACKING - SAVE CLIPS - ######################################
        if track_options['use_deeplabcut']:
            if self.dlc_config_settings['clips']['visual'] or self.dlc_config_settings['clips']['audio']:
                print('        ... extracting trials video clips')
                # Update a list of sessions whose clips have been saved
                # These are the sessions that will be processed using DLC
                # If there are other videos from other sessions in the target folder, DLC
                # will ignore them
                session_clips_l = save_trial_clips(self.dlc_config_settings['clips'],
                                                self.dlc_config_settings['clips_folder'])
                self.clips_l.append(session_clips_l)


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

        indexes = ['x', 'y', 'orientation']
        if stop_frame == -1:
            array = np.full((video_duration_frames, len(indexes)), np.nan)
        else:
            array = np.full((stop_frame-start_frame, len(indexes)), np.nan)

        self.data = pd.DataFrame(array, columns=indexes)

        # Display background
        if self.cfg['preview']:
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
            display, cnt = get_contours(bg, frame, self.cfg['th_scaling'])

            # extract info from contour
            if cnt:
                (x, y), radius = cv2.minEnclosingCircle(cnt[0])
                centeroid = (int(x), int(y))
                self.coord_l.append(centeroid)
                self.data.loc[f-start_frame-1]['x'] = centeroid[0]
                self.data.loc[f-start_frame-1]['y'] = centeroid[1]

                # draw contours and trace and ROI
                if self.cfg['preview']:
                    cv2.drawContours(frame, cnt, -1, (0, 255, 0), 1)
                    drawtrace(frame, self.coord_l, (255, 0, 0), self.trace_length)

                if not justCoM:
                    # get mouse orientation
                    self.data.loc[f-start_frame-1]['orientation']  = get_body_orientation(
                                                        f, cnt[0], bg, display, frame, start_frame,
                                                        self.data['orientation'].values,
                                                        self.arena_floor, self.cfg['tail_th_scaling'])

            if self.cfg['preview']:
                display_results(f, frame, display, self.magnif_factor, self)
                #  Control framerate and exit
                # need to adjust this so that it waits for the correct amount to match fps
                key = cv2.waitKey(10) & 0xFF

    @staticmethod
    def tracking_use_dlc(database, clips_l):
        print('====================================\nExtracting Pose using DeepLabCut')

        dlc_config_settings = load_yaml(track_options['cfg_dlc'])

        print('        ... extracting pose from clips')
        TF_settings = dlc_setupTF(track_options)
        dlc_analyseVideos.analyse(TF_settings, dlc_config_settings['clips_folder'], clips_l)

        print('        ... integrating results in database')
        database = dlc_retreive_data(dlc_config_settings['clips_folder'], database, clips_l)

        print('        ... cleaning up')
        if not dlc_config_settings['store trial videos']:
            dlc_clear_folder(dlc_config_settings['clips_folder'], dlc_config_settings['store trial videos'])

        return database
