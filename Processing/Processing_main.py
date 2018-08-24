from Utils.loadsave_funcs import load_yaml
from Processing.Processing_utils import *
from Plotting import Plt_individual_stimresponses
from Utils.maths import calc_angle_2d
import matplotlib.pyplot as plt

from Config import processing_options


class Processing():
    def __init__(self, session, database):
        # load processing settings from yaml file
        self.settings = load_yaml(processing_options['cfg'])

        # get arguments
        self.session = session
        self.database = database

        # Apply stuff to tracking data
        for tracking_data in list(self.session.Tracking.values()):
            # Get velocity
            self.extract_velocity(tracking_data)
            self.extract_location_relative_shelter(tracking_data)

    def extract_location_relative_shelter(self, data):
        """
        This function extracts the mouse position relative to the shelter

        """
        # Get the position of the centre of the shelter
        if not 'shelter location' in data.metadata.keys():
            if self.settings['shelter location'] == 'roi':
                # Get centre of shelter roi
                shelter_location = get_shelter_location(self.settings['shelter location'], self.session)
            else:
                # Get it from DLC tracking
                shelter_location = get_shelter_location(self.settings['shelter location'], data)
            data.metadata['shelter location'] = shelter_location
        else:
            shelter_location = data.metadata['shelter location']

        # Get position relative to shelter from STD data
        if self.settings['std']:
            adjusted_pos = calc_position_relative_point((data.std_tracking['x'].values,
                                                         data.std_tracking['y'].values), shelter_location)
            data.std_tracking['adjusted x'], data.std_tracking['adjusted y'] = adjusted_pos[0], adjusted_pos[1]

        # Get position relative to shelter from DLC data
        if self.settings['dlc']:
            for bp in data.dlc_tracking['Posture'].keys():
                if self.settings['dlc single bp']:
                    if bp != self.settings['dlc single bp']:
                        continue

                # Extract velocity for a single bodypart as determined by the user
                bp_data, _ = from_dlc_to_single_bp(data, bp)
                adjusted_pos = calc_position_relative_point((bp_data['x'].values,
                                                             bp_data['y'].values), shelter_location)
                bp_data['adjusted x'], bp_data['adjusted y'] = adjusted_pos[0], adjusted_pos[1]

    def extract_orientation(self, data):
        """
        This function extracts the mouse' orientation from the angle of two body parts [DLC tracking] and the
        position of the shelter.

        These bodyparts can be any tracked with DLC, but ideally they should be the centre of the body and the start of
        the tail. Other body parts might result in incorrect estimation of the orientation.
        The shelter position can either be given by the position of a user-defined ROI [during background extraction
        in video processing] or by landmarks tracked with DLC.

        :param data:
        :return:
        """

        # Get the position of the centre of the shelter
        if not data.metadata['shelter location']:
            if self.settings['shelter location'] == 'roi':
                # Get centre of shelter roi
                shelter_location = get_shelter_location(self.settings['shelter location'], self.session)
            else:
                # Get it from DLC tracking
                shelter_location = get_shelter_location(self.settings['shelter location'], data)
            data.metadata['shelter location'] = shelter_location
        else:
            shelter_location = data.metadata['shelter location']

        # Get the position of the two bodyparts
        body, _ = from_dlc_to_single_bp(data, self.settings['body'])
        tail, _ = from_dlc_to_single_bp(data, self.settings['tail'])

        # Get angle relative to frame
        absolute_angle = calc_angle_2d(body, tail)

    def extract_velocity(self, data):
        """
            Calculates the velocity of the mouse from tracking data.

            Should work for both individual sessions and whole cohorts

            Get get std tracking (i.e. center of mass) or DLC tracking (many bodyparts)

            Can give the velocity in px/frame, px/sec or bodylengths/s

            ***
            The results are saved in each tracking object's dataframe

            ***
            :return:
            """

        # Get the unit velocity will be calculated as and save it in the metadata
        unit = self.settings['velocity unit']
        if 'bl' in unit:
            # Need to calculate the avg length of the mouse
            if not data.dlc_tracking:
                print('Could not calculate velocity in bodylength per second as there are no DLC data\n'
                      'to extract the length of the body from.\nCalculating velocity as pixel per second instead')
                unit = 'pxpersec'
                bodylength = False
            else:
                if 'body length' in data.metadata.keys():
                    bodylength = data.metadata['body length']
                else:
                    # Extract body length from DLC data
                    bodylength = get_average_bodylength(data, head_tag=self.settings['head'], tail_tag=self.settings['tail'])
                    data.metadata['body length'] = bodylength
        else:
            bodylength=False

        data.metadata['Velocity unit'] = unit

        fps = self.session.Metadata.videodata[0]['Frame rate'][0]
        if self.settings['std']:
            # Extract velocity using std tracking data
            distance = calc_distance_2d((data.std_tracking['x'].values, data.std_tracking['y'].values))
            data.std_tracking['Velocity'] = distance
            data.std_tracking['Acceleration'] = calc_acceleration(distance, unit=unit, fps=fps,
                                                                  bodylength=bodylength)

        if self.settings['dlc']:
            if not data.dlc_tracking:
                return

            for bp in data.dlc_tracking['Posture'].keys():
                if self.settings['dlc single bp']:
                    if bp != self.settings['dlc single bp']:
                        continue

                # Extract velocity for a single bodypart as determined by the user
                bp_data, bodypart = from_dlc_to_single_bp(data, bp)

                if self.settings['dlc single bp']:
                    if bodypart != self.settings['dlc single bp']:
                        self.settings['dlc single bp'] = bodypart

                distance = calc_distance_2d((bp_data['x'], bp_data['y']))
                bp_data['Velocity'] = distance
                bp_data['Accelearation'] = calc_acceleration(distance, unit=unit, fps=fps, bodylength=bodylength)

    def extract_posture(self):
            """
            Reconstruct posture from DLC data

            Save in tracking.processing

            :param self:
            :return:
            """

            pass

            # bp_data = Processing_utils.from_dlc_to_single_bp(session, settings['center of mass bodypart'])
            # head_data = Processing_utils.from_dlc_to_single_bp(session, settings['head'])
            # tail_data = Processing_utils.from_dlc_to_single_bp(session, settings['tail'])
            #
            # # Calculate velocity and other stuff
            # bp_data = Processing_utils.single_bp_calc_stuff(bp_data)
            # head_data = Processing_utils.single_bp_calc_stuff(head_data)
            # tail_data = Processing_utils.single_bp_calc_stuff(tail_data)
            #
            # # Reconstruct posture
            # bp_data = Processing_utils.pose_reconstruction(head_data, bp_data, tail_data)
            #
            # # Plot
            # Plt_individual_stimresponses.plotter(bp_data)
