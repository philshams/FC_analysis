# Import packages

import imageio
imageio.plugins.ffmpeg.download()
from collections import namedtuple
from warnings import warn
# Import functions and params from other scripts
# from Tracking.std_tracking_functions import get_body_orientation, get_mvmt_direction, get_velocity

# from Tracking import dlc_analyseVideos
from Tracking.Tracking_utils import *

from Utils.video_funcs import cut_crop_video, register_arena, get_background
from Utils import Data_rearrange_funcs
from Utils.utils_classes import Trial
from Utils.loadsave_funcs import load_yaml

from Config import startf, track_options, dlc_config_settings, video_analysis_settings, fisheye_map_location
from termcolor import colored



class Tracking():
    """"
    Tracks position or posture from video data based on the user selected parameters
    The results are saved in self.database
    """
    def __init__(self, session, TF_setup, TF_settings):
        # set up class variables
        self.session = session

        # TF set up -- check if TF is running already
        self.TF_setup = TF_setup
        self.TF_settings = TF_settings

        # Load settings from config
        self.dlc_config_settings = dlc_config_settings
        self.dlc_config_settings['clips'] = {'visual': {}, 'audio': {}, 'digital': {}}

        # params for contour extraction
        self.fps = self.session['Metadata'].videodata[0]['Frame rate'][0]
        # self.background = self.session['Metadata'].videodata[0]['Background']
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

        # Track whole session
        if track_options['track whole session']:
            self.track_wholesession()

        # Track single trials and/or save video clips
        if (track_options['track stimulus responses'] and track_options['use standard tracking']) \
                or track_options['save stimulus clips'] or track_options['register arena']:
            self.track_trials()


    def track_wholesession(self):
        # Check if tracking the whole session
        print(colored('Tracking the whole session.', 'green'))
        self.session['Tracking']['Whole Session'] = {}
        video_tracking_data = namedtuple('coordinates', 'x y ori')

        for idx, vid in enumerate(self.session['Metadata'].video_file_paths):
            if idx == 0:
                start_frame = startf
            else:
                start_frame = 0
            tracked = self.tracking(self.background, vid[0],
                                    start_frame=start_frame, stop_frame=-1, video_fps=self.fps)
            tracking_data = video_tracking_data(tracked.data.x, tracked.data.y, tracked.data.orientation)
            self.session['Tracking']['Whole Session']['Video_{}'.format(idx)] = tracking_data


    def track_trials(self):

        if track_options['track stimulus responses'] and track_options['use standard tracking']:
            print(colored('Tracking individual trials.', 'green'))
        if track_options['save stimulus clips']:
            print(colored('Extracting clips.', 'green'))

        # GET BACKGROUND - #####################################
        if (not np.array(self.session['Metadata'].videodata[0]['Background']).size):
            # not self.session['Metadata'].videodata[0]['Background']
            print(colored('Fetching background', 'green'))
            self.session['Metadata'].videodata[0]['Background'] = get_background(
                self.session['Metadata'].video_file_paths[0][0],start_frame=1000, avg_over=100)

        # REGISTER ARENA - #####################################
        if track_options['register arena']:
            self.session['Metadata'].videodata[0]['Arena Transformation'] = register_arena(
                self.session['Metadata'].videodata[0]['Background'], fisheye_map_location)
            if not self.session['Metadata'].videodata[0]['Arena Transformation']:
                print(colored('Registering arena', 'green'))
                self.session['Metadata'].videodata[0]['Arena Transformation'] = register_arena(
                    self.session['Metadata'].videodata[0]['Background'], fisheye_map_location)
            registration = self.session['Metadata'].videodata[0]['Arena Transformation'][0]
        else:
            registration = False

        # LOOP OVER STIMULI - #######################################
        for stim_type, stims in self.session['Metadata'].stimuli.items():
            # For each stim type get the list of stim frames
            if not stims:
                continue

            for vid_num, stims_video in enumerate(stims):  # Loop over each video in the session
                for idx, stim_frame in enumerate(stims_video):  # Loop over each stim for each video
                    # try:
                        # SET UP - ######################################
                        start_frame = int(stim_frame-(video_analysis_settings['fast track wndw pre']*self.fps))
                        stop_frame = int(stim_frame+(video_analysis_settings['fast track wndw post']*self.fps))

                        self.videoname = '{}_{}_{}-{} ({}\')'.format(self.session['Metadata'].experiment,
                                                             self.session['Metadata'].mouse_id,
                                                             stim_type, idx + 1, round(start_frame / self.fps / 60))
                        # Generate empty trial object and add it into the database
                        trial_metadata = create_trial_metadata(self.videoname, stim_type, start_frame, stop_frame,
                                                               self.session['Metadata'].video_file_paths[vid_num])
                        empty_trial = Trial()
                        empty_trial.metadata = trial_metadata
                        empty_trial.name = trial_metadata['Name']
                        self.session['Tracking'][empty_trial.metadata['Name']] = empty_trial

                        # SAVE CLIPS - ######################################
                        if track_options['save stimulus clips']:
                            if self.session.Metadata.videodata[0]['Clips Directory'] and not track_options['do not overwrite']:
                                if not idx:
                                    print(colored('Video clips already saved', 'green'))
                            else:
                                cut_crop_video(self.session['Metadata'].video_file_paths[vid_num][0], self.videoname,
                                    self.dlc_config_settings['clips_folder'], start_frame, stop_frame, stim_frame, fisheye_map_location, registration,
                                    save_clip = True, display_clip = True, counter = True, make_flight_image = True)
                                # self.dlc_config_settings['clips'][stim_type][self.videoname] = trial_clip

                        # STD TRACKING - ######################################
                        if track_options['track stimulus responses'] and track_options['use standard tracking']:
                            print(colored('processing trial {} - standard tracking'.format(self.videoname),'green'))

                            trial = self.tracking(self.background, self.session['Metadata'].video_file_paths[vid_num][0],
                                                  start_frame=start_frame, stop_frame=stop_frame, video_fps=self.fps, justCoM=track_options['stdtracking_justCoM'])

                            trial = Data_rearrange_funcs.restructure_trial_data(trial, stim_type, idx, vid_num)
                            trial.metadata = trial_metadata

                            # Merge into database
                            old_trial = self.session['Tracking'][trial.metadata['Name']]
                            self.session['Tracking'][trial.metadata['Name']] = merge_std_dlc_trials(old_trial, trial)

            # Record location of clips
            self.session.Metadata.videodata[0]['Clips Directory'] = self.dlc_config_settings['clips_folder']


                        # DLC TRACKING - PREPARE CLIPS - ######################################
                        # """
                        # for DLC tracking, extract videoclips for the peri-stimulus time which will be analysed using
                        # DeepLabCut in a moment
                        # """
                        # if track_options['use_deeplabcut']:
                            # set up
                            # start_sec = start_frame * (1 / self.fps)
                            # stop_sec = stop_frame * (1 / self.fps)

                            # Extract trial clip and store it so that we can save all trials at the same time
                            # trial_clip = cut_crop_video(self.session['Metadata'].video_file_paths[vid_num][0],
                            #                             cut=True, starts=start_sec, fins=stop_sec,
                            #                             save_format=None, ret=True)
                            # self.dlc_config_settings['clips'][stim_type][self.videoname] = trial_clip
                    # except:
                    #     warn('Could not succesfully completestd tracking for session {}'.format(str(self.session)))

        # DLC TRACKING - SAVE CLIPS - ######################################
        # if track_options['use_deeplabcut']:
        #     if self.dlc_config_settings['clips']['visual'] or self.dlc_config_settings['clips']['audio']:
        #         print('        ... extracting trials video clips')
                # Update a list of sessions whose clips have been saved
                # These are the sessions that will be processed using DLC
                # If there are other videos from other sessions in the target folder, DLC
                # will ignore them
                # print('saving clips')
                # session_clips_l = save_trial_clips(self.dlc_config_settings['clips'],
                #                                 self.dlc_config_settings['clips_folder'])
                # self.clips_l.append(session_clips_l)


########################################################################################################################
########################################################################################################################
    # @clock
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
            stop_frame = video_duration_frames
        else:
            array = np.full((stop_frame-start_frame, len(indexes)), np.nan)

        self.data = pd.DataFrame(array, columns=indexes)

        # Display background
        if video_analysis_settings['preview']:
            cv2.namedWindow('bg')
            cv2.imshow('bg', bg)
            cv2.moveWindow('bg', 1200, 0)
            cv2.waitKey(1)

        # Start tracking
        for f in range(start_frame, stop_frame):
            if not stop_frame == -1 and f > stop_frame:
                return self

            ret, frame = cap.read()
            if not ret:
                return self

            # get contours from frame
            display, cnt = get_contours(bg, frame, self.video_analysis_settings['th_scaling'])

            # extract info from contour
            if cnt:
                (x, y), radius = cv2.minEnclosingCircle(cnt[0])
                centeroid = (int(x), int(y))
                self.coord_l.append(centeroid)
                self.data.loc[f-start_frame]['x'] = centeroid[0]
                self.data.loc[f-start_frame]['y'] = centeroid[1]

                # draw contours and trace and ROI
                if video_analysis_settings['preview']:
                    cv2.drawContours(frame, cnt, -1, (0, 255, 0), 1)
                    drawtrace(frame, self.coord_l, (255, 0, 0), self.trace_length)

                if not justCoM:
                    # get mouse orientation
                    self.data.loc[f-start_frame]['orientation']  = get_body_orientation(
                                                        f, cnt[0], bg, display, frame, start_frame,
                                                        self.data['orientation'].values,
                                                        self.arena_floor, self.video_analysis_settings['tail_th_scaling'])

            if video_analysis_settings['preview']:
                display_results(f, frame, display, self.magnif_factor, self)
                #  Control framerate and exit
                # need to adjust this so that it waits for the correct amount to match fps
                key = cv2.waitKey(10) & 0xFF
        return self

    @staticmethod
    def tracking_use_dlc(database):
        print('====================================\nExtracting Pose using DeepLabCut')

        # dlc_config_settings = load_yaml(track_options['cfg_dlc'])

        print('        ... extracting pose from clips')
        TF_settings = dlc_setupTF(track_options)
        dlc_analyseVideos.analyse(TF_settings, dlc_config_settings['clips_folder'])

        print('        ... integrating results in database')
        database = dlc_retreive_data(dlc_config_settings['clips_folder'], database)

        print('        ... cleaning up')
        if not dlc_config_settings['store trial videos']:
            dlc_clear_folder(dlc_config_settings['clips_folder'], dlc_config_settings['store trial videos'])

        return database
