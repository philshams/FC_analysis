# Import packages
from Config import startf, track_options, dlc_config_settings, video_analysis_settings, fisheye_map_location, y_offset, x_offset

from Utils.video_funcs import peri_stimulus_video_clip, peri_stimulus_video_clip, peri_stimulus_analysis, extract_coordinates_with_dlc, initialize_wall_analysis
from Utils.registration_funcs import register_arena, get_background, invert_fisheye_map

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
import scipy

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
            print(colored('Fetching background', 'green'))
            self.session['Metadata'].videodata[0]['Background'] = get_background(
                self.session['Metadata'].video_file_paths[0][0],start_frame=1000, avg_over=100)

        # REGISTER ARENA
        if not self.session['Metadata'].videodata[0]['Arena Transformation']:
            print(colored('Registering arena', 'green'))
            self.session['Metadata'].videodata[0]['Arena Transformation'] = register_arena(
                self.session['Metadata'].videodata[0]['Background'], fisheye_map_location)

        self.registration = self.session['Metadata'].videodata[0]['Arena Transformation']

    # ========================================================
    #           TRACK SESSION USING DLC
    # ========================================================
    def track_wholesession(self):

        if ('speed' in self.session.Tracking): # and False:
            print(colored('Coordinates have already been extracted', 'green'))
        else:
            # get the original behaviour video
            video = self.session.Metadata.video_file_paths[0]
            print(colored('Tracking the whole session from ' + video[0], 'green'))

            # extract the coordinates for the video
            self.session['Tracking'], self.registration = extract_coordinates_with_dlc(dlc_config_settings, video, self.registration)

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
        session_trials_plot_workspace = None
        exploration_arena_in_cum = None

        for stim_type, stims in self.session['Metadata'].stimuli.items():
            # For each stim type get the list of stim frames
            if not stims[0]:
                continue

            for vid_num, stims_video in enumerate(stims):  # Loop over each video in the session

                # SET UP TRIAL TYPES FOR BACKGROUND AND TABLE OF CONTENTS
                border_size = 40
                rectangle_thickness = 3

                save_folder = os.path.join(self.dlc_config_settings['clips_folder'], self.session['Metadata'].experiment, self.session['Metadata'].mouse_id)

                self.session['Tracking']['Trial Types'], session_trials_video, session_video, session_trials_plot_background, number_of_trials, height, width = \
                    self.get_trial_types(vid_num, stims_video, stims, border_size, rectangle_thickness, save_folder)




                for trial_num, stim_frame in enumerate(stims_video):  # Loop over each stim for each video

                    # SET UP VIDEO/STIMULUS TO BE ANALYZED - ######################################
                    start_frame = int(stim_frame-(video_analysis_settings['fast track wndw pre']*self.fps))
                    end_frame = int(stim_frame+(video_analysis_settings['fast track wndw post']*self.fps))
                    if trial_num:
                        previous_stim_frame = stims_video[trial_num - 1]
                    else:
                        previous_stim_frame = 0

                    self.videoname = '{}_{}_{}-{} ({}\')'.format(self.session['Metadata'].experiment,
                                                         self.session['Metadata'].mouse_id,
                                                         stim_type, trial_num + 1, round(stim_frame / self.fps / 60))

                    cv2.rectangle(session_trials_plot_background, (0, 0), (width, border_size), 0, -1)

                    textsize = cv2.getTextSize(self.videoname, 0, .55, 1)[0]
                    textX = int((width - textsize[0]) / 2)
                    cv2.putText(session_trials_plot_background, self.videoname, (textX, border_size-5), 0, .55, (255,255,255), thickness=1)


                    cv2.rectangle(session_trials_plot_background, (int( width + border_size / 4), int((trial_num-1) / number_of_trials * (height+2*border_size/4) + border_size/4)),
                                  (int(width + 3*border_size/4), int((trial_num - 1 + 1) / number_of_trials * (height+2*border_size/4))), (0, 0, 0), rectangle_thickness)

                    cv2.rectangle(session_trials_plot_background, (int( width + border_size / 4), int(trial_num / number_of_trials * (height+2*border_size/4) + border_size/4)),
                                  (int(width + 3*border_size/4), int((trial_num + 1) / number_of_trials * (height+2*border_size/4))), (200, 200, 200), rectangle_thickness)


                    # ========================================================
                    #           SAVE CLIPS AND FLIGHT IMAGES
                    # ========================================================
                    if track_options['save stimulus clips']:
                        if self.session.Metadata.videodata[0]['Clips Directory'] and track_options['do not overwrite']:
                            if not trial_num:
                                print(colored('Video clips already saved', 'green'))
                        else:
                            if ('nose' in self.session.Tracking):
                                session_trials_plot_workspace, exploration_arena_in_cum = peri_stimulus_analysis(
                                                       self.session.Tracking, self.session['Metadata'].video_file_paths[vid_num][0],
                                                       self.videoname, save_folder, session_video, previous_stim_frame,
                                                       session_trials_video, session_trials_plot_workspace, session_trials_plot_background, exploration_arena_in_cum,
                                                       len(stims[0]), trial_num, start_frame, end_frame, stim_frame, self.registration, self.fps, border_size,
                                                       analyze_wall=track_options['analyze wall'], save_clip=True, display_clip=True, counter=True, make_flight_image=True)

                            else:
                                peri_stimulus_video_clip(self.session['Metadata'].video_file_paths[vid_num][0], self.videoname,
                                                         save_folder,
                                                        start_frame, end_frame, stim_frame, self.registration, self.fps,
                                                        analyze_wall=track_options['analyze wall'],save_clip=True, display_clip=True, counter=True, make_flight_image=True)

            # Record location of clips
            self.session.Metadata.videodata[0]['Clips Directory'] = self.dlc_config_settings['clips_folder']

        scipy.misc.imsave(os.path.join(save_folder, self.videoname + '_session_trials.tif'), session_trials_plot_workspace)

        session_trials_video.release()
        session_video.release()


    def get_trial_types(self, vid_num, stims_video, stims, border_size, rectangle_thickness, save_folder):

        trial_types = []

        wall_color = np.array([222, 122, 122]) / 1.6
        probe_color = np.array([200, 200, 200]) / 1.6
        no_wall_color = np.array([122, 122, 222]) / 1.6

        trial_colors = [probe_color, no_wall_color, probe_color, wall_color]

        number_of_trials = len(stims[0])

        for sub_trial_num, sub_stim_frame in enumerate(stims_video):
            sub_start_frame = int(sub_stim_frame - (video_analysis_settings['fast track wndw pre'] * self.fps))
            sub_end_frame = int(sub_stim_frame + (video_analysis_settings['fast track wndw post'] * self.fps))
            vid = cv2.VideoCapture(self.session['Metadata'].video_file_paths[vid_num][0])
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if not sub_trial_num:
                session_trials_plot_background = np.zeros((height + border_size, width + border_size, 3)).astype(np.uint8)

                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                session_videoname = '{}_{}'.format(self.session['Metadata'].experiment, self.session['Metadata'].mouse_id, )
                session_trials_video = cv2.VideoWriter(os.path.join( save_folder, session_videoname + '_session_dlc.avi' ),
                                                       fourcc, self.fps, (width + border_size, height + border_size), True)
                session_video = cv2.VideoWriter(os.path.join( save_folder, session_videoname + '_session.avi'),
                                                       fourcc, self.fps, (width + 2*border_size, height + 2*border_size), True)

            if not ('Trial Types' in self.session['Tracking']):
                _, trial_type, _, _, _, _, _, _, _ = initialize_wall_analysis(True, sub_stim_frame, sub_start_frame, sub_end_frame, self.registration,
                                                                          x_offset, y_offset, vid, width, height)
                trial_types.append(trial_type)
            else:
                trial_types = self.session['Tracking']['Trial Types']

            cv2.rectangle(session_trials_plot_background,
                          (int(width + border_size / 4), int(sub_trial_num / number_of_trials * (height + 2 * border_size / 4) + border_size / 4)),
                          (int(width + 3 * border_size / 4), int((sub_trial_num + 1) / number_of_trials * (height + 2 * border_size / 4))),
                          trial_colors[trial_types[sub_trial_num] + 1], -1)

            cv2.rectangle(session_trials_plot_background,
                          (int(width + border_size / 4), int(sub_trial_num / number_of_trials * (height + 2 * border_size / 4) + border_size / 4)),
                          (int(width + 3 * border_size / 4), int((sub_trial_num + 1) / number_of_trials * (height + 2 * border_size / 4))), (0, 0, 0),
                          rectangle_thickness)


        return trial_types, session_trials_video, session_video, session_trials_plot_background, number_of_trials, height, width