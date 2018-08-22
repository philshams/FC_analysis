import yaml

from Utils.loadsave_funcs import load_yaml
from Utils.utils_classes import Processing_class
from Processing.Processing_utils import calc_distance_2d, calc_velocity, from_dlc_to_single_bp, get_average_bodylength
from Plotting import Plt_individual_stimresponses

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
            # Check if .processing is already part of the session's tracking
            if not 'processing' in tracking_data.__dict__.keys():
                tracking_data.processing = Processing_class()

            # Get velocity
            self.extract_velocity(tracking_data)

        a = 1

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

        if self.settings['std']:
            # Extract velocity using std tracking data
            distance = calc_distance_2d((data.std_tracking['x'].values, data.std_tracking['y'].values))
            data.std_tracking['Velocity'] = calc_velocity(distance, unit=unit, fps=self.session.Video['Frame rate'],
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
                vel = calc_velocity(distance, unit=unit, fps=self.session.Video['Frame rate'], bodylength=bodylength)
                bp_data['Velocity'] = vel

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
