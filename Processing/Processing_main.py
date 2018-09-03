import matplotlib.pyplot as plt   # Used to test stuff while writing new functions

from multiprocessing.dummy import Pool as ThreadPool

from Utils.loadsave_funcs import load_yaml
from Processing.Processing_utils import *
from Utils.maths import calc_angle_2d, calc_ang_velocity
import time

from Config import processing_options


class Processing():
    def __init__(self, session, database):
        # load processing settings from yaml file
        self.settings = load_yaml(processing_options['cfg'])

        # get arguments
        self.session = session
        self.database = database

        # Apply stuff to tracking data
        for data_name, tracking_data in list(self.session.Tracking.items()):
            if data_name == 'Exploration' or data_name == 'Whole Session':
                continue

            # Process stuff [in parallel, 825x faster than calling each func one at the time]
            self.tracking_data = tracking_data
            funcs = [self.extract_bodylength, self.extract_velocity, self.extract_location_relative_shelter,
                     self.extract_orientation]
            pool = ThreadPool(4)
            _ = pool.apply_async(parallelizer, funcs, tracking_data)

            self.extract_ang_velocity(tracking_data)

            # Store info in metadata
            self.define_processing_metadata(tracking_data)

    def define_processing_metadata(self, tracking_data):
        tracking_data.metadata['Processing info'] = self.settings

    # FUNCTIONS ===========================================================================================

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
        head, _ = from_dlc_to_single_bp(data, self.settings['head'])
        body, _ = from_dlc_to_single_bp(data, self.settings['body'])
        tail, _ = from_dlc_to_single_bp(data, self.settings['tail'])

        # Get angle relative to frame
        absolute_angle = calc_angle_2d(body, tail, vectors=True)
        data.dlc_tracking['Posture']['body']['Orientation'] = [x+360 for x in absolute_angle]

        # Get head angle relative to body angle
        absolute_angle_head = calc_angle_2d(head, body, vectors=True)
        data.dlc_tracking['Posture']['body']['Head angle'] = [x+360 for x in absolute_angle_head]

    def extract_bodylength(self, data):
        # Get bodylength
        avg_bodylength, bodylength = get_average_bodylength(data, head_tag=self.settings['head'],
                                                            tail_tag=self.settings['tail'])

        # Store results
        data.metadata['avg body length'] = avg_bodylength

        for bp in data.dlc_tracking['Posture'].keys():
            if self.settings['dlc single bp']:
                if bp != self.settings['dlc single bp']:
                    continue

            # Extract velocity for a single bodypart as determined by the user
            bp_data, bodypart = from_dlc_to_single_bp(data, bp)
            bp_data['Body length'] = bodylength

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
        allunits = ['pxperframe', 'pxpersec', 'blpersec']
        if 'bl' in unit or 'all' in unit:
            # Need to calculate the avg length of the mouse
            if not data.dlc_tracking:
                print('Could not calculate velocity in bodylength per second as there are no DLC data\n'
                      'to extract the length of the body from.\nCalculating velocity as pixel per second instead')
                unit = 'pxpersec'
                bodylength = False
            else:
                if 'body length' in data.metadata.keys():
                    bodylength = data.metadata['avg body length']
                else:
                    # Extract body length from DLC data
                    bodylength, _ = get_average_bodylength(data, head_tag=self.settings['head'],
                                                            tail_tag=self.settings['tail'])
        else:
            bodylength=False

        data.metadata['Velocity unit'] = unit

        fps = self.session.Metadata.videodata[0]['Frame rate'][0]
        if self.settings['std']:
            # Extract velocity using std tracking data
            distance = calc_distance_2d((data.std_tracking['x'].values, data.std_tracking['y'].values))
            if not 'all' in unit:
                data.std_tracking['Velocity'] = scale_velocity_by_unit(distance, unit=unit, fps=fps, bodylength=bodylength)
                data.std_tracking['Acceleration'] = calc_acceleration(distance, unit=unit, fps=fps,
                                                                      bodylength=bodylength)
            else:
                for un in allunits:
                    data.std_tracking['Velocity_{}'.format(un)] = scale_velocity_by_unit(distance, unit=un, fps=fps,
                                                                           bodylength=bodylength)
                    data.std_tracking['Acceleration_{}'.format(un)] = calc_acceleration(distance, unit=un, fps=fps,
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
                if not 'all' in unit:
                    bp_data['Velocity'] = scale_velocity_by_unit(distance, unit=unit, fps=fps, bodylength=bodylength)
                    bp_data['Accelearation'] = calc_acceleration(distance, unit=unit, fps=fps, bodylength=bodylength)
                else:
                    for un in allunits:
                        bp_data['Velocity_{}'.format(un)] = scale_velocity_by_unit(distance, unit=un, fps=fps,
                                                                                             bodylength=bodylength)
                        bp_data['Acceleration_{}'.format(un)] = calc_acceleration(distance, unit=un, fps=fps,
                                                                                            bodylength=bodylength)

    def extract_ang_velocity(self, data):
        """
        Get orientation [calculated previously] and compute the velocity in it

        Returns the angular velocity in either deg per frame or deg per second

        :param data:
        :return:
        """
        if self.settings['ang vel unit'] == 'degperframe':
            orientation = data.dlc_tracking['Posture'][self.settings['body']]['Orientation']
            data.dlc_tracking['Posture'][self.settings['body']]['Body ang vel'] = calc_ang_velocity(orientation)

            orientation = data.dlc_tracking['Posture'][self.settings['body']]['Head angle']
            data.dlc_tracking['Posture'][self.settings['body']]['Head ang vel'] = calc_ang_velocity(orientation)
        else:
            fps = self.session.Metadata.videodata[0]['Frame rate'][0]
            orientation = data.dlc_tracking['Posture'][self.settings['body']]['Orientation']
            data.dlc_tracking['Posture'][self.settings['body']]['Body ang vel'] =\
                calc_ang_velocity(orientation, fps=fps)

            orientation = data.dlc_tracking['Posture'][self.settings['body']]['Head angle']
            data.dlc_tracking['Posture'][self.settings['body']]['Head ang vel'] =\
                calc_ang_velocity(orientation, fps=fps)


