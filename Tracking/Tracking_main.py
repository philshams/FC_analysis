# Import packages
from Config import startf, track_options, dlc_config_settings, video_analysis_settings, fisheye_map_location, y_offset, x_offset

from Utils.video_funcs import peri_stimulus_video_clip, peri_stimulus_video_clip, peri_stimulus_analysis, register_arena, get_background, \
    invert_fisheye_map, extract_coordinates_with_dlc

import imageio
imageio.plugins.ffmpeg.download()
from collections import namedtuple
from warnings import warn

# Import functions and params from other scripts
# from Tracking.std_tracking_functions import get_body_orientation, get_mvmt_direction, get_velocity
from Tracking.Tracking_utils import *


from Utils import Data_rearrange_funcs
from Utils.utils_classes import Trial
from Utils.loadsave_funcs import load_yaml
import h5py
import glob

from scipy import interpolate

from termcolor import colored
import yaml
import platform

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

        # Register arena
        if track_options['register arena']:
            self.register_arena()
        else:
            self.registration = False

        # Track whole session - DLC
        if track_options['track whole session']:
            self.track_wholesession()

        # Track single trials - DLC
        if track_options['track stimulus responses'] and not track_options['track whole session']:
            self.track_trials()

        # Save clips
        if track_options['save stimulus clips']:
            self.save_trials()


    # ========================================================
    #           REGISTER ARENA
    # ========================================================
    def register_arena(self):
        # GET BACKGROUND
        if (not np.array(self.session['Metadata'].videodata[0]['Background']).size):
            # not self.session['Metadata'].videodata[0]['Background']
            print(colored('Fetching background', 'green'))
            self.session['Metadata'].videodata[0]['Background'] = get_background(
                self.session['Metadata'].video_file_paths[0][0],start_frame=1000, avg_over=100)

        # REGISTER ARENA
        if not self.session['Metadata'].videodata[0]['Arena Transformation']:
            print(colored('Registering arena', 'green'))
            self.session['Metadata'].videodata[0]['Arena Transformation'] = register_arena(
                self.session['Metadata'].videodata[0]['Background'], fisheye_map_location)
        # self.session['Metadata'].videodata[0]['Arena Transformation'][3] = fisheye_map_location
        self.registration = self.session['Metadata'].videodata[0]['Arena Transformation']

    # ========================================================
    #           TRACK SESSION USING DLC
    # ========================================================
    def track_wholesession(self):

        if hasattr(self.session,'Coordinates'):
            print(colored('Coordinates have already been extracted', 'green'))
        else:
            # get the original behaviour video
            video = self.session.Metadata.video_file_paths[0]
            print(colored('Tracking the whole session from ' + video[0], 'green'))

            # extract the coordinates for the video
            self.session['Coordinates'], self.registration = extract_coordinates_with_dlc(dlc_config_settings, video, self.registration)

    # ========================================================
    #          EXTRACT FLIGHT FOR EACH TRIAL
    # ========================================================
    def save_trials(self):
        if track_options['track stimulus responses'] and track_options['use standard tracking']:
            print(colored('Tracking individual trials.', 'green'))
        if track_options['save stimulus clips']:
            print(colored('Extracting clips.', 'green'))

        # ========================================================
        #           LOOP OVER EACH STIMULUS
        # ========================================================
        for stim_type, stims in self.session['Metadata'].stimuli.items():
            # For each stim type get the list of stim frames
            if not stims:
                continue

            for vid_num, stims_video in enumerate(stims):  # Loop over each video in the session
                for idx, stim_frame in enumerate(stims_video):  # Loop over each stim for each video

                        # SET UP VIDEO/STIMULUS TO BE ANALYZED - ######################################
                        start_frame = int(stim_frame-(video_analysis_settings['fast track wndw pre']*self.fps))
                        stop_frame = int(stim_frame+(video_analysis_settings['fast track wndw post']*self.fps))

                        self.videoname = '{}_{}_{}-{} ({}\')'.format(self.session['Metadata'].experiment,
                                                             self.session['Metadata'].mouse_id,
                                                             stim_type, idx + 1, round(stim_frame / self.fps / 60))
                        # Generate empty trial object and add it into the database
                        trial_metadata = create_trial_metadata(self.videoname, stim_type, start_frame, stop_frame,
                                                               self.session['Metadata'].video_file_paths[vid_num])
                        empty_trial = Trial()
                        empty_trial.metadata = trial_metadata
                        empty_trial.name = trial_metadata['Name']
                        self.session['Tracking'][empty_trial.metadata['Name']] = empty_trial

                        # ========================================================
                        #           SAVE CLIPS AND FLIGHT IMAGES
                        # ========================================================
                        if track_options['save stimulus clips']:
                            if self.session.Metadata.videodata[0]['Clips Directory'] and track_options['do not overwrite']:
                                if not idx:
                                    print(colored('Video clips already saved', 'green'))
                            else:
                                if hasattr(self.session, 'Coordinates'):
                                    peri_stimulus_analysis(self.session.Coordinates, self.session['Metadata'].video_file_paths[vid_num][0], self.videoname,
                                                           os.path.join(self.dlc_config_settings['clips_folder'],self.session['Metadata'].experiment),
                                                           start_frame, stop_frame, stim_frame, self.registration, self.fps,
                                                           analyze_wall=track_options['analyze wall'], save_clip=True, display_clip=True, counter=True, make_flight_image=True)

                                else:
                                    peri_stimulus_video_clip(self.session['Metadata'].video_file_paths[vid_num][0], self.videoname,
                                                            os.path.join(self.dlc_config_settings['clips_folder'],self.session['Metadata'].experiment),
                                                            start_frame, stop_frame, stim_frame, self.registration, self.fps,
                                                            analyze_wall=track_options['analyze wall'],save_clip=True, display_clip=True, counter=True, make_flight_image=True)





            # Record location of clips
            self.session.Metadata.videodata[0]['Clips Directory'] = self.dlc_config_settings['clips_folder']
