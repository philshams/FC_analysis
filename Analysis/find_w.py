import numpy as np
import os
import pickle
import scipy.stats
import skfmm
import random
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from Utils.Data_rearrange_funcs import check_session_selected
from Utils.registration_funcs import get_arena_details, model_arena
from Config import setup
_, _, _, _, _, _, _, selector_type, selector = setup()

'''
---
----
Calculate the parameters to be used in the strategy simulations
---
----
'''

class parameterize():
    '''     Calculate the parameters to be used in the strategy simulations     '''

    def __init__(self, db):

        # initialize the input and likelihood
        self.input = np.zeros((0,9))
        self.likelihood = np.array([]);
        self.responsibilities = np.array([])
        self.HV_residual = np.array([])
        self.time_to_fit = True

        # loop across sessions, adding to the input and likelihood
        for session_name in db.index:

            # check if the session is selected
            self.session = db.loc[session_name]
            selected = check_session_selected(self.session.Metadata, selector_type, selector)

            # create the feature array and likelihood output for each selected session
            if selected:

                print(session_name)

                try: self.extract_coordinates()
                except: continue

                self.compute_quantities()

                self.compute_eligibility()

                self.compute_likelihood()

                self.compute_input()


        # fit the model to the data from all selected sessions
        self.standardize()
        self.fit_model()
        print('done')
        # self.get_angle_std(compute = True)

        # initialize the input and likelihood
        self.input = np.zeros((0, 9))
        self.likelihood = np.array([])
        self.responsibilities = np.array([])
        self.time_to_fit = False

        # loop across pairs of sessions, adding to the input and likelihood
        for session_name1 in db.index:
            for session_name2 in db.index:

                # check if the session is selected
                self.session1 = db.loc[session_name1]
                self.session2 = db.loc[session_name2]
                selected1 = check_session_selected(self.session1.Metadata, selector_type, selector)
                selected2 = check_session_selected(self.session2.Metadata, selector_type, selector)

                if selected1 and selected2 and session_name1 != session_name2:
                    print(session_name1 + ' and ' + session_name2)

                    self.extract_coordinates_dual()

                    self.compute_quantities_dual()

                    self.compute_eligibility_dual()

                    self.compute_likelihood_dual()

                    self.compute_input_dual()

        self.standardize()
        self.fit_model()



    def extract_coordinates(self):
        '''         EXTRACT THE SAVED COORDINATES FILE FOR THE CURRENT SESSION      '''

        # find the coordinates file
        video_path = self.session['Metadata'].video_file_paths[0][0]
        processed_coordinates_file = os.path.join(os.path.dirname(video_path), 'coordinates')

        # load the coordinates file
        with open(processed_coordinates_file, "rb") as dill_file:
            self.coordinates = pickle.load(dill_file)

        # get the arena details
        self.x_offset, self.y_offset, self.obstacle_type, self.shelter_location, self.subgoal_location, \
        self.infomark_location, self.obstacle_changes = get_arena_details(self.session['Metadata'].experiment)

        # load the distance and angle to obstacle data
        self.distance_arena = np.load('C:\\Drive\\DLC\\transforms\\distance_arena_' + self.obstacle_type + '.npy')
        self.angle_arena = np.load('C:\\Drive\\DLC\\transforms\\angle_arena_' + self.obstacle_type + '.npy')

        # get the arena
        self.arena, _, _ = model_arena((self.distance_arena.shape[0], self.distance_arena.shape[1]), 0, False, self.obstacle_type, simulate=False)
        self.shelter_location = [int(a * b / 1000) for a, b in zip(self.shelter_location, self.arena.shape)]

        # initialize the geodesic map
        phi = np.ones_like(self.arena)

        # mask the map wherever there's an obstacle
        mask = (self.arena == 90)
        self.phi_masked = np.ma.MaskedArray(phi, mask)

        # get the geodesic map of distance from the shelter
        phi_from_shelter = self.phi_masked.copy()
        phi_from_shelter[self.shelter_location[1], self.shelter_location[0]] = 0
        self.distance_from_shelter = np.array(skfmm.distance(phi_from_shelter))

    def extract_coordinates_dual(self):
        '''         EXTRACT THE SAVED COORDINATES FILE FOR THE CURRENT SESSION      '''

        # find the coordinates file
        video_path1 = self.session1['Metadata'].video_file_paths[0][0]
        processed_coordinates_file1 = os.path.join(os.path.dirname(video_path1), 'coordinates')

        # load the coordinates file
        with open(processed_coordinates_file1, "rb") as dill_file:
            self.coordinates1 = pickle.load(dill_file)

        # get the arena details
        self.x_offset, self.y_offset, self.obstacle_type1, self.shelter_location, self.subgoal_location, \
        self.infomark_location, self.obstacle_changes = get_arena_details(self.session1['Metadata'].experiment)

        # find the coordinates file
        video_path2 = self.session2['Metadata'].video_file_paths[0][0]
        processed_coordinates_file2 = os.path.join(os.path.dirname(video_path2), 'coordinates')

        # load the coordinates file
        with open(processed_coordinates_file2, "rb") as dill_file:
            self.coordinates2 = pickle.load(dill_file)

        # get the arena details
        self.x_offset, self.y_offset, self.obstacle_type2, self.shelter_location, self.subgoal_location, \
        self.infomark_location, self.obstacle_changes = get_arena_details(self.session2['Metadata'].experiment)

        if self.obstacle_type1 == self.obstacle_type2:
            self.obstacle_type = self.obstacle_type1

            # load the distance and angle to obstacle data
            self.distance_arena = np.load('C:\\Drive\\DLC\\transforms\\distance_arena_' + self.obstacle_type + '.npy')
            self.angle_arena = np.load('C:\\Drive\\DLC\\transforms\\angle_arena_' + self.obstacle_type + '.npy')

            # get the arena
            self.arena, _, _ = model_arena((self.distance_arena.shape[0], self.distance_arena.shape[1]), 0, False, self.obstacle_type, simulate=False)
            self.shelter_location = [int(a * b / 1000) for a, b in zip(self.shelter_location, self.arena.shape)]

            # initialize the geodesic map
            phi = np.ones_like(self.arena)

            # mask the map wherever there's an obstacle
            mask = (self.arena == 90)
            self.phi_masked = np.ma.MaskedArray(phi, mask)

            # get the geodesic map of distance from the shelter
            phi_from_shelter = self.phi_masked.copy()
            phi_from_shelter[self.shelter_location[1], self.shelter_location[0]] = 0
            self.distance_from_shelter = np.array(skfmm.distance(phi_from_shelter))

        else:
            # go to next pair of sessions
            print('DIFFERENT ARENAS -- FIX')

    def compute_quantities(self):
        '''         COMPUTE THE QUANTITIES NEEDED TO CALCULATE THE DIFFERENCE IN INITIAL CONDITIONS FUNCTION        '''

        # extract start-point and end-point frame number
        self.start_indices = np.where(self.coordinates['start_index'])[0]
        self.end_indices = self.coordinates['start_index'][self.start_indices].astype(int)

        # get the position
        self.start_position = self.coordinates['center_body_location'][0][self.start_indices], self.coordinates['center_body_location'][1][self.start_indices]
        self.end_position = self.coordinates['center_body_location'][0][self.end_indices], self.coordinates['center_body_location'][1][self.end_indices]

        # get the distance from the shelter
        self.shelter_distance = self.coordinates['distance_from_shelter'][self.start_indices]
        self.shelter_distance_end = self.coordinates['distance_from_shelter'][self.end_indices]

        # get the geodesic distance from the shelter
        self.shelter_distance_geodesic = self.distance_from_shelter[self.start_position[1].astype(int), self.start_position[0].astype(int)]
        self.shelter_distance_geodesic_end = self.distance_from_shelter[self.end_position[1].astype(int), self.end_position[0].astype(int)]

        # get the vector direction
        self.path_direction = np.angle((self.end_position[0] - self.start_position[0]) + (-self.end_position[1] + self.start_position[1]) * 1j, deg=True)

        # get the head direction
        self.head_direction = self.coordinates['body_angle'][self.start_indices]

        # get the distance from the obstacle
        self.distances_from_obstacle = self.distance_arena[self.start_position[1].astype(int), self.start_position[0].astype(int)]
        self.distances_from_obstacle[np.isnan(self.distances_from_obstacle)] = 0 # replace nan

        # get the angle to the obstacle
        self.angles_from_obstacle = self.angle_arena[self.start_position[1].astype(int), self.start_position[0].astype(int)]
        self.angles_from_obstacle[np.isnan(self.angles_from_obstacle)] = 0 # replace nan

        # get the stimulus-triggeredness
        stim_idx, no_wall_stim_idx = self.get_stim_indices()
        self.stimulus_evoked = np.array([(i in stim_idx) for i in self.start_indices])
        self.stimulus_evoked_no_wall = np.array([(i in no_wall_stim_idx) for i in self.start_indices])
        # print(np.where(self.stimulus_evoked)[0])

        # get the HV direction
        self.HV_direction = np.angle((self.shelter_location[0] - self.start_position[0]) + (-self.shelter_location[1] + self.start_position[1]) * 1j, deg=True)
        #self.coordinates['shelter_angle'][self.start_indices]

        # set the error of the gaussian orientation movement generation model
        # self.get_angle_std()
        self.angle_std = 20 #13.8

    def get_angle_std(self, compute = False):
        '''         COMPUTE THE STANDARD DEVIATION OF ANGLES GIVEN THE INTENDED DIRECTION OF THE SHELTER        '''

        # compute the STD
        if compute:
            # compute the std
            HV_STD = np.std(self.HV_residual)
            print(HV_STD)
            self.angle_std = HV_STD

        # aggregate the data for the STD
        else:
            # the first or last orientation movement
            first_movement = np.concatenate(([True], np.diff(self.start_indices) > 300))
            last_movement = np.concatenate((np.diff(self.start_indices) > 300, [True]))

            # only take complete homings
            ends_in_shelter = self.shelter_distance_end < 277

            # only take homings sufficiently far from shelter
            far_from_shelter = self.shelter_distance > 277

            # get the index of relevant orientation movements
            homing_vector_index = far_from_shelter * (self.stimulus_evoked_no_wall+self.stimulus_evoked) * ends_in_shelter

            # get the HV residual from these orientation movements
            actual_direction = self.path_direction[homing_vector_index]
            home_direction = self.HV_direction[homing_vector_index]
            residual = home_direction - actual_direction
            residual[residual > 180] = 360 - residual[residual > 180]
            residual[residual < -180] = 360 + residual[residual < -180]

            # residual = residual[abs(residual) < 30]

            self.HV_residual = np.append(self.HV_residual, residual)


    def compute_quantities_dual(self):
        '''         COMPUTE THE QUANTITIES NEEDED TO CALCULATE THE DIFFERENCE IN INITIAL CONDITIONS FUNCTION        '''

        # extract start-point and end-point frame number ------------------------------
        self.start_indices1 = np.where(self.coordinates1['start_index'])[0]
        self.end_indices1 = self.coordinates1['start_index'][self.start_indices1].astype(int)

        # get the position
        self.start_position1 = self.coordinates1['center_body_location'][0][self.start_indices1], self.coordinates1['center_body_location'][1][self.start_indices1]
        self.end_position1 = self.coordinates1['center_body_location'][0][self.end_indices1], self.coordinates1['center_body_location'][1][self.end_indices1]

        # get the distance from the shelter
        self.shelter_distance1 = self.coordinates1['distance_from_shelter'][self.start_indices1]
        self.shelter_distance_end1 = self.coordinates1['distance_from_shelter'][self.end_indices1]

        # get the geodesic distance from the shelter
        self.shelter_distance_geodesic1 = self.distance_from_shelter[self.start_position1[1].astype(int), self.start_position1[0].astype(int)]
        self.shelter_distance_geodesic_end1 = self.distance_from_shelter[self.end_position1[1].astype(int), self.end_position1[0].astype(int)]


        # get the vector direction
        self.path_direction1 = np.angle((self.end_position1[0] - self.start_position1[0]) + (-self.end_position1[1] + self.start_position1[1]) * 1j, deg=True)

        # get the head direction
        self.head_direction1 = self.coordinates1['body_angle'][self.start_indices1]

        # get the distance from the obstacle
        self.distances_from_obstacle1 = self.distance_arena[self.start_position1[1].astype(int), self.start_position1[0].astype(int)]
        self.distances_from_obstacle1[np.isnan(self.distances_from_obstacle1)] = 0 # replace nan

        # get the angle to the obstacle
        self.angles_from_obstacle1 = self.angle_arena[self.start_position1[1].astype(int), self.start_position1[0].astype(int)]
        self.angles_from_obstacle1[np.isnan(self.angles_from_obstacle1)] = 0 # replace nan

        # get the stimulus-triggeredness
        stim_idx1, _ = self.get_stim_indices(session_num = 1)
        self.stimulus_evoked1 = np.array([(i in stim_idx1) for i in self.start_indices1])

        # get the HV direction
        self.HV_direction1 = np.angle((self.shelter_location[0] - self.start_position1[0]) + (-self.shelter_location[1] + self.start_position1[1]) * 1j, deg=True)


        # extract start-point and end-point frame number ------------------------------
        self.start_indices2 = np.where(self.coordinates2['start_index'])[0]
        self.end_indices2 = self.coordinates2['start_index'][self.start_indices2].astype(int)

        # get the position
        self.start_position2 = self.coordinates2['center_body_location'][0][self.start_indices2], self.coordinates2['center_body_location'][1][self.start_indices2]
        self.end_position2 = self.coordinates2['center_body_location'][0][self.end_indices2], self.coordinates2['center_body_location'][1][self.end_indices2]

        # get the distance from the shelter
        self.shelter_distance2 = self.coordinates2['distance_from_shelter'][self.start_indices2]
        self.shelter_distance_end2 = self.coordinates2['distance_from_shelter'][self.end_indices2]

        # get the geodesic distance from the shelter
        self.shelter_distance_geodesic2 = self.distance_from_shelter[self.start_position2[1].astype(int), self.start_position2[0].astype(int)]
        self.shelter_distance_geodesic_end2 = self.distance_from_shelter[self.end_position2[1].astype(int), self.end_position2[0].astype(int)]


        # get the vector direction
        self.path_direction2 = np.angle((self.end_position2[0] - self.start_position2[0]) + (-self.end_position2[1] + self.start_position2[1]) * 1j, deg=True)

        # get the head direction
        self.head_direction2 = self.coordinates2['body_angle'][self.start_indices2]

        # get the distance from the obstacle
        self.distances_from_obstacle2 = self.distance_arena[self.start_position2[1].astype(int), self.start_position2[0].astype(int)]
        self.distances_from_obstacle2[np.isnan(self.distances_from_obstacle2)] = 0 # replace nan

        # get the angle to the obstacle
        self.angles_from_obstacle2 = self.angle_arena[self.start_position2[1].astype(int), self.start_position2[0].astype(int)]
        self.angles_from_obstacle2[np.isnan(self.angles_from_obstacle2)] = 0 # replace nan

        # get the stimulus-triggeredness
        stim_idx2, _ = self.get_stim_indices(session_num = 2)
        self.stimulus_evoked2 = np.array([(i in stim_idx2) for i in self.start_indices2])

        # get the HV direction
        self.HV_direction2 = np.angle((self.shelter_location[0] - self.start_position2[0]) + (-self.shelter_location[1] + self.start_position2[1]) * 1j, deg=True)

        # set the error of the gaussian orientation movement generation model
        self.angle_std = 20

    def compute_eligibility(self):
        '''         USE THE QUANTITIES COMPUTED ABOVE TO SELECT PAIRS OF PATHS SIMILAR ENOUGH TO QUALIFY FOR REPETITION           '''

        # reshape arrays for matrix multiplication
        self.angles_from_obstacle = np.reshape(self.angles_from_obstacle, (len(self.angles_from_obstacle), 1))
        self.distances_from_obstacle = np.reshape(self.distances_from_obstacle, (len(self.distances_from_obstacle), 1))

        # get the angle similarity arrays
        plus_90 = self.angles_from_obstacle == 90
        minus_90 = self.angles_from_obstacle == -90
        less_than_90 = abs(self.angles_from_obstacle) < 90
        more_than_90 = abs(self.angles_from_obstacle) > 90

        # get the angle similarity matrix
        different_obstacle_angle = plus_90 * minus_90.T + minus_90 * plus_90.T + less_than_90 * more_than_90.T + more_than_90 * less_than_90.T
        similar_obstacle_angle = ~different_obstacle_angle

        # get the eligibility matrix -- each row is the prior homing, and each column is the posterior homing
        eligible_pairs = similar_obstacle_angle

        # only include stimulus driven ones as the latter homing
        eligible_pairs = eligible_pairs * self.stimulus_evoked

        # make it an upper triangular matrix, and remove the identity band
        self.eligible_pairs = ((np.triu(eligible_pairs) - np.identity(eligible_pairs.shape[0])) > 0 ).astype(bool)
        # self.eligible_pairs = (np.triu(eligible_pairs) ).astype(bool)
        self.eligible_indices = np.where(self.eligible_pairs)

        # low enough absolute distance
        position_offset = np.sqrt( (self.start_position[0][self.eligible_indices[0]] - self.start_position[0][self.eligible_indices[1]])**2 + \
                                   (self.start_position[1][self.eligible_indices[0]] - self.start_position[1][self.eligible_indices[1]])**2 )

        # in the same quadrant
        in_same_quadrant = (self.angles_from_obstacle[self.eligible_indices[0]] == self.angles_from_obstacle[self.eligible_indices[1]]) + \
                           (abs(self.angles_from_obstacle[self.eligible_indices[0]]) < 90) * (abs(self.angles_from_obstacle[self.eligible_indices[1]]) < 90) + \
                           (abs(self.angles_from_obstacle[self.eligible_indices[0]]) > 90) * (abs(self.angles_from_obstacle[self.eligible_indices[1]]) > 90)
        in_same_quadrant = (in_same_quadrant[:, 0]).astype(bool)

        # turn at least 25 degrees
        # turn_angle = abs(self.path_direction[self.eligible_indices[0]] - self.head_direction[self.eligible_indices[1]])
        # turn_angle[turn_angle > 180] = abs(360 - turn_angle[turn_angle > 180])

        # target should decrease either geodesic or euclidean distance to shelter
        decrease_in_euclidean_distance = self.shelter_distance[self.eligible_indices[1]] - self.shelter_distance_end[self.eligible_indices[0]]
        decrease_in_geodesic_distance = self.shelter_distance_geodesic[self.eligible_indices[1]]  - self.shelter_distance_geodesic_end[self.eligible_indices[0]]

        # stimulus evoked get a pass for the distance requirement
        stimulus_evoked = self.stimulus_evoked[self.eligible_indices[0]]

        # turn angle
        turn_angle = abs(self.head_direction[self.eligible_indices[1]] - self.path_direction[self.eligible_indices[0]])
        turn_angle[turn_angle > 180] = abs(360 - turn_angle[turn_angle > 180])

        eligible = ( (decrease_in_euclidean_distance > 40) + (decrease_in_geodesic_distance > 40) ) * \
                     ( (position_offset < (np.sqrt(np.sum(self.arena > 0)) / 2 )) + stimulus_evoked ) * \
                      in_same_quadrant  * (turn_angle > 20) #* stimulus_evoked

        self.eligible_pairs[self.eligible_indices[0][~eligible], self.eligible_indices[1][~eligible]] = 0
        self.eligible_indices = np.where(self.eligible_pairs)

    def compute_eligibility_dual(self):
        '''         USE THE QUANTITIES COMPUTED ABOVE TO SELECT PAIRS OF PATHS SIMILAR ENOUGH TO QUALIFY FOR REPETITION           '''

        # reshape arrays for matrix multiplication
        self.angles_from_obstacle1 = np.reshape(self.angles_from_obstacle1, (len(self.angles_from_obstacle1), 1))
        self.distances_from_obstacle1 = np.reshape(self.distances_from_obstacle1, (len(self.distances_from_obstacle1), 1))
        # reshape arrays for matrix multiplication
        self.angles_from_obstacle2 = np.reshape(self.angles_from_obstacle2, (len(self.angles_from_obstacle2), 1))
        self.distances_from_obstacle2 = np.reshape(self.distances_from_obstacle2, (len(self.distances_from_obstacle2), 1))


        # get the angle similarity arrays
        plus_901 = self.angles_from_obstacle1 == 90
        minus_901 = self.angles_from_obstacle1 == -90
        less_than_901 = abs(self.angles_from_obstacle1) < 90
        more_than_901 = abs(self.angles_from_obstacle1) > 90
        # get the angle similarity arrays
        plus_902 = self.angles_from_obstacle2 == 90
        minus_902 = self.angles_from_obstacle2 == -90
        less_than_902 = abs(self.angles_from_obstacle2) < 90
        more_than_902 = abs(self.angles_from_obstacle2) > 90

        # get the angle similarity matrix
        different_obstacle_angle = plus_901 * minus_902.T + minus_901 * plus_902.T + less_than_901 * more_than_902.T + more_than_901 * less_than_902.T
        similar_obstacle_angle = ~different_obstacle_angle

        # get the eligibility matrix -- each row is the prior homing, and each column is the posterior homing
        eligible_pairs = similar_obstacle_angle

        # only include stimulus driven ones
        eligible_pairs = eligible_pairs * self.stimulus_evoked2

        # make it an upper triangular matrix, and remove the identity band
        # self.eligible_pairs = ((np.triu(eligible_pairs) - np.identity(eligible_pairs.shape[0])) > 0 ).astype(bool)
        self.eligible_pairs = (np.triu(eligible_pairs) ).astype(bool)
        self.eligible_indices = np.where(self.eligible_pairs)

        # low enough absolute distance
        position_offset = np.sqrt( (self.start_position1[0][self.eligible_indices[0]] - self.start_position2[0][self.eligible_indices[1]])**2 + \
                                   (self.start_position1[1][self.eligible_indices[0]] - self.start_position2[1][self.eligible_indices[1]])**2 )
        # in the same quadrant
        in_same_quadrant = (self.angles_from_obstacle1[self.eligible_indices[0]] == self.angles_from_obstacle2[self.eligible_indices[1]]) + \
                           (abs(self.angles_from_obstacle1[self.eligible_indices[0]]) < 90) * (abs(self.angles_from_obstacle2[self.eligible_indices[1]]) < 90) + \
                           (abs(self.angles_from_obstacle1[self.eligible_indices[0]]) > 90) * (abs(self.angles_from_obstacle2[self.eligible_indices[1]]) > 90)
        in_same_quadrant = (in_same_quadrant[:, 0]).astype(bool)

        # turn at least 25 degrees
        # turn_angle = abs(self.path_direction[self.eligible_indices[0]] - self.head_direction[self.eligible_indices[1]])
        # turn_angle[turn_angle > 180] = abs(360 - turn_angle[turn_angle > 180])

        # target should decrease either geodesic or euclidean distance to shelter
        decrease_in_euclidean_distance = self.shelter_distance2[self.eligible_indices[1]] - self.shelter_distance_end1[self.eligible_indices[0]]
        decrease_in_geodesic_distance = self.shelter_distance_geodesic2[self.eligible_indices[1]] - self.shelter_distance_geodesic_end1[self.eligible_indices[0]]

        # stimulus evoked get a pass for the distance requirement
        stimulus_evoked = self.stimulus_evoked1[self.eligible_indices[0]]

        # turn angle
        turn_angle = abs(self.head_direction2[self.eligible_indices[1]] - self.path_direction1[self.eligible_indices[0]])
        turn_angle[turn_angle > 180] = abs(360 - turn_angle[turn_angle > 180])

        eligible = ((decrease_in_euclidean_distance > 40) + (decrease_in_geodesic_distance > 40)) * \
                   ((position_offset < (np.sqrt(np.sum(self.arena > 0)) / 2)) + stimulus_evoked) * \
                   in_same_quadrant * (turn_angle > 20) #* stimulus_evoked

        self.eligible_pairs[self.eligible_indices[0][~eligible], self.eligible_indices[1][~eligible]] = 0
        self.eligible_indices = np.where(self.eligible_pairs)

    def compute_likelihood(self):
        '''         CALCULATE THE ANGLE PREDICTED BY EACH PREVIOUS ESCAPE       '''

        # initialize dual-strategy array
        self.strategy_error = np.zeros((len(self.eligible_indices[0]), 2))
        self.strategy_angle = np.zeros((len(self.eligible_indices[0]), 2))

        # get the angle from PR
        path_repetition_angle = self.path_direction[self.eligible_indices[0]]

        # get the actual angle
        path_angle = self.path_direction[self.eligible_indices[1]]
        self.strategy_angle[:, 0] = path_repetition_angle

        # get the endpoint from TR
        current_startpoint = self.start_position[0][self.eligible_indices[1]], self.start_position[1][self.eligible_indices[1]]
        previous_endpoint = self.end_position[0][self.eligible_indices[0]], self.end_position[1][self.eligible_indices[0]]
        previous_startpoint = self.start_position[0][self.eligible_indices[0]], self.start_position[1][self.eligible_indices[0]]

        # get the path distance
        prior_length = np.sqrt((previous_endpoint[0] - previous_startpoint[0])**2 + (previous_endpoint[1] - previous_startpoint[1])**2)

        # get the angle from TR
        target_repetition_angle = np.angle((previous_endpoint[0] - current_startpoint[0]) + (-previous_endpoint[1] + current_startpoint[1]) * 1j, deg=True)
        self.strategy_angle[:, 1] = target_repetition_angle

        # get the error in the angle
        self.strategy_error[:, 0] = (path_angle - path_repetition_angle)
        self.strategy_error[:, 1] = np.inf #(path_angle - target_repetition_angle)

        self.strategy_error[self.strategy_error > 180] = 360 - self.strategy_error[self.strategy_error > 180]
        self.strategy_error[self.strategy_error < -180] = 360 + self.strategy_error[self.strategy_error < -180]

        # get the best angle among PR and TR
        winning_strategy = np.argmin(abs(self.strategy_error), 1)
        repetition_error = np.array([self.strategy_error[i, s] for i, s in enumerate(winning_strategy)])
        repetition_angle = np.array([self.strategy_angle[i, s] for i, s in enumerate(winning_strategy)])

        # # get the HV error too
        HV_error = (path_angle - self.HV_direction[self.eligible_indices[1]])
        HV_error[HV_error > 180] = 360 - HV_error[HV_error > 180]; HV_error[HV_error < -180] = 360 + HV_error[HV_error < -180]

        # --only include homings that are better than the HV interpretation
        self.repetition_possibly_used = (abs(repetition_error) < abs(HV_error)) * (abs(repetition_error) < 10) * (prior_length > 60)

        # take the repetition that had low error
        flight_with_a_match = np.unique(self.eligible_indices[1][self.repetition_possibly_used])

        # set all flights preceding this repetition as applicable
        applicable_homings = np.zeros((1, len(self.start_indices)))
        applicable_homings[0, flight_with_a_match] = 1

        # update eligibility accordingly
        self.eligible_pairs = self.eligible_pairs * applicable_homings
        old_eligible_indices = self.eligible_indices
        self.eligible_indices = np.where(self.eligible_pairs)
        update_eligibility = [(t in self.eligible_indices[1]) for t in old_eligible_indices[1]]

        # update which errors are used
        repetition_error = repetition_error[update_eligibility]
        repetition_angle = repetition_angle[update_eligibility]

        # get the (normalized) likelihood that this was used, given our error model
        # self.likelihood = np.concatenate( (self.likelihood, scipy.stats.norm(0, self.angle_std).pdf(repetition_error) / scipy.stats.norm(0, self.angle_std).pdf(0) ))
        # self.likelihood = np.concatenate((self.likelihood, (abs(repetition_error) < 10).astype(int) ))

        # initialize input and output
        degrees = np.zeros((100, 1))
        degrees[:, 0] = np.linspace(0, 2*np.pi, 100)
        current_responsibility = np.zeros(len(repetition_error))
        kappa = 1 / (np.deg2rad(self.angle_std)**2)
        LR = linear_model.Ridge(alpha=.8)
        # LR = linear_model.LinearRegression()
        # self.compute_input()
        # self.standardize()

        # loop across each posterior homing
        for h in np.unique(self.eligible_indices[1]):
            # get the prior homings in use
            prior_homings = (self.eligible_indices[1]==h)

            # get their angles
            mu = (repetition_angle[prior_homings] + 180) * np.pi / 180

            # compute the von mises features (normalized so range is 0 to 1)
            degree_features = np.tile(degrees, (1, len(mu)) )
            vm_features = np.exp(kappa * np.cos(degree_features - mu)) / np.exp(kappa)

            # compute the desire output
            vm_likelihood = np.exp(kappa * np.cos(degrees - ((self.path_direction[h] + 180) * np.pi / 180)) ) / np.exp(kappa)

            # compute the weights
            LR.fit(vm_features, vm_likelihood)
            repetition_weights = LR.coef_[0,:]
            repetition_weights[repetition_weights < .01] = 0

            current_responsibility[prior_homings] = repetition_weights

            # mean_weighted_dist = np.sum(self.input_array[prior_homings,0] * repetition_weights / np.sum(repetition_weights) )
            # print(h)
            #
            # print('weighted: ' + str(np.round( (mean_weighted_dist), 3)))
            #
            # print('')

            # plt.plot(degrees, vm_features* (1/len(mu)), color='gray')
            # plt.plot(degrees, np.sum(vm_features * (1/len(mu)), axis = 1), color='black')
            # plt.plot(degrees, vm_likelihood, color='green', linewidth = 5)
            # plt.plot(degrees, vm_features * repetition_weights, color='blue')
            # plt.plot(degrees, np.sum(vm_features * repetition_weights, axis=1), color='red', linewidth = 5, linestyle = '--')
            # plt.pause(.3)
            # plt.close()

        # print(np.corrcoef(self.input_array[:, 0], current_responsibility)[0][1] )
        # plt.figure()
        # plt.scatter(self.input_array[:, 0], current_responsibility)
        # plt.xlabel('distance z-score')
        # plt.ylabel('responsibility')
        # plt.show()
        # plt.pause(3)

        # add the bonus for being stimulus triggered
        # prior_homing_stimulus_evoked = self.stimulus_evoked[self.eligible_indices[0]].astype(bool)
        # current_responsibility =current_responsibility[~prior_homing_stimulus_evoked]

        self.responsibilities = np.concatenate((self.responsibilities, current_responsibility))

        # print(np.corrcoef(self.input[:, 0], self.responsibilities)[0][1])
        # print(np.corrcoef(self.input[:, 0], self.responsibilities)[0][1])
        # print('')
        # plt.figure()
        # plt.scatter(self.input_array[:, 0], repetition_weights)
        # plt.pause(3)

    def compute_likelihood_dual(self):
        '''         CALCULATE THE ANGLE PREDICTED BY EACH PREVIOUS ESCAPE       '''

        # initialize dual-strategy array
        self.strategy_error = np.zeros((len(self.eligible_indices[0]), 2))
        self.strategy_angle = np.zeros((len(self.eligible_indices[0]), 2))

        # get the angle from PR
        path_repetition_angle = self.path_direction1[self.eligible_indices[0]]

        # get the actual angle
        path_angle = self.path_direction2[self.eligible_indices[1]]
        self.strategy_angle[:, 0] = path_repetition_angle

        # get the endpoint from TR
        current_startpoint = self.start_position2[0][self.eligible_indices[1]], self.start_position2[1][self.eligible_indices[1]]
        previous_endpoint = self.end_position1[0][self.eligible_indices[0]], self.end_position1[1][self.eligible_indices[0]]

        # get the angle from TR
        target_repetition_angle = np.angle((previous_endpoint[0] - current_startpoint[0]) + (-previous_endpoint[1] + current_startpoint[1]) * 1j, deg=True)
        self.strategy_angle[:, 1] = target_repetition_angle

        # get the error in the angle
        self.strategy_error[:, 0] = (path_angle - path_repetition_angle)
        self.strategy_error[:, 1] = np.inf  # (path_angle - target_repetition_angle)

        self.strategy_error[self.strategy_error > 180] = 360 - self.strategy_error[self.strategy_error > 180]
        self.strategy_error[self.strategy_error < -180] = 360 + self.strategy_error[self.strategy_error < -180]

        # get the best angle among PR and TR
        winning_strategy = np.argmin(abs(self.strategy_error), 1)
        repetition_error = np.array([self.strategy_error[i, s] for i, s in enumerate(winning_strategy)])
        repetition_angle = np.array([self.strategy_angle[i, s] for i, s in enumerate(winning_strategy)])

        # # get the HV error too
        HV_error = (path_angle - self.HV_direction2[self.eligible_indices[1]])
        HV_error[HV_error > 180] = 360 - HV_error[HV_error > 180];
        HV_error[HV_error < -180] = 360 + HV_error[HV_error < -180]

        # --only include homings that are better than the HV interpretation
        self.repetition_possibly_used = (abs(repetition_error) < abs(HV_error)) * (abs(repetition_error) < 6)

        # take the repetition that had low error
        flight_with_a_match = np.unique(self.eligible_indices[1][self.repetition_possibly_used])

        # set all flights preceding this repetition as applicable
        applicable_homings = np.zeros((1, len(self.start_indices2)))
        applicable_homings[0, flight_with_a_match] = 1

        # update eligibility accordingly
        self.eligible_pairs = self.eligible_pairs * applicable_homings
        old_eligible_indices = self.eligible_indices
        self.eligible_indices = np.where(self.eligible_pairs)
        update_eligibility = [(t in self.eligible_indices[1]) for t in old_eligible_indices[1]]

        # update which errors are used
        repetition_error = repetition_error[update_eligibility]
        repetition_angle = repetition_angle[update_eligibility]

        # get the (normalized) likelihood that this was used, given our error model
        # self.likelihood = np.concatenate( (self.likelihood, scipy.stats.norm(0, self.angle_std).pdf(repetition_error) / scipy.stats.norm(0, self.angle_std).pdf(0) ))
        # self.likelihood = np.concatenate((self.likelihood, (abs(repetition_error) < 10).astype(int) ))

        # initialize input and output
        degrees = np.zeros((100, 1))
        degrees[:, 0] = np.linspace(0, 2 * np.pi, 100)
        current_responsibility = np.zeros(len(repetition_error))
        kappa = 1 / (np.deg2rad(self.angle_std) ** 2)
        LR = linear_model.Ridge(alpha=.8)
        # LR = linear_model.LinearRegression()
        # self.compute_input()
        # self.standardize()

        # loop across each posterior homing
        for h in np.unique(self.eligible_indices[1]):
            # get the prior homings in use
            prior_homings = (self.eligible_indices[1] == h)

            # get their angles
            mu = (repetition_angle[prior_homings] + 180) * np.pi / 180

            # compute the von mises features (normalized so range is 0 to 1)
            degree_features = np.tile(degrees, (1, len(mu)))
            vm_features = np.exp(kappa * np.cos(degree_features - mu)) / np.exp(kappa)

            # compute the desire output
            vm_likelihood = np.exp(kappa * np.cos(degrees - ((self.path_direction2[h] + 180) * np.pi / 180))) / np.exp(kappa)

            # compute the weights
            LR.fit(vm_features, vm_likelihood)
            repetition_weights = LR.coef_[0, :]
            repetition_weights[repetition_weights < .01] = 0

            current_responsibility[prior_homings] = repetition_weights

            # mean_weighted_dist = np.sum(self.input_array[prior_homings,0] * repetition_weights / np.sum(repetition_weights) )
            # print(h)
            #
            # print('weighted: ' + str(np.round( (mean_weighted_dist), 3)))
            #
            # print('')

            # plt.plot(degrees, vm_features* (1/len(mu)), color='gray')
            # plt.plot(degrees, np.sum(vm_features * (1/len(mu)), axis = 1), color='black')
            # plt.plot(degrees, vm_likelihood, color='green', linewidth = 5)
            # plt.plot(degrees, vm_features * repetition_weights, color='blue')
            # plt.plot(degrees, np.sum(vm_features * repetition_weights, axis=1), color='red', linewidth = 5, linestyle = '--')
            # plt.pause(.3)
            # plt.close()

        # print(np.corrcoef(self.input_array[:, 0], current_responsibility)[0][1] )
        # plt.figure()
        # plt.scatter(self.input_array[:, 0], current_responsibility)
        # plt.xlabel('distance z-score')
        # plt.ylabel('responsibility')
        # plt.show()
        # plt.pause(3)
        # prior_homing_stimulus_evoked = self.stimulus_evoked1[self.eligible_indices[0]].astype(bool)
        # current_responsibility = current_responsibility[~prior_homing_stimulus_evoked]

        self.responsibilities = np.concatenate((self.responsibilities, current_responsibility))

        # print(np.corrcoef(self.input[:, 0], self.responsibilities)[0][1])
        # print(np.corrcoef(self.input[:, 0], self.responsibilities)[0][1])
        # print('')
        # plt.figure()
        # plt.scatter(self.input_array[:, 0], repetition_weights)
        # plt.pause(3)

    def compute_input(self):

        # get the position offsets
        position_offset = np.sqrt( (self.start_position[0][self.eligible_indices[0]] - self.start_position[0][self.eligible_indices[1]])**2 + \
                                   (self.start_position[1][self.eligible_indices[0]] - self.start_position[1][self.eligible_indices[1]])**2 )

        # normalize by trial
        position_array = np.ones_like(self.eligible_pairs) * np.nan
        position_array[self.eligible_indices[0], self.eligible_indices[1]] = position_offset
        position_array = position_array / np.nanmean(position_array, axis = 0)
        position_offset = position_array[~np.isnan(position_array)]

        shelter_distance_offset = self.shelter_distance_end[self.eligible_indices[0]] - self.shelter_distance[self.eligible_indices[1]]
        prior_distance_to_shelter = self.shelter_distance_end[self.eligible_indices[0]]
        post_distance_to_shelter = self.shelter_distance_end[self.eligible_indices[1]]

        # get the HD offsets
        # HD_offset = abs(self.path_direction[self.eligible_indices[0]] - self.head_direction[self.eligible_indices[1]])
        HD_offset = abs(self.head_direction[self.eligible_indices[0]] - self.head_direction[self.eligible_indices[1]])
        HD_offset[HD_offset > 180] = abs(360 - HD_offset[HD_offset > 180])

        # given that they're in the same quandrant...
        in_same_quadrant = (self.angles_from_obstacle[self.eligible_indices[0]] == self.angles_from_obstacle[self.eligible_indices[1]]) + \
                           (abs(self.angles_from_obstacle[self.eligible_indices[0]]) < 90) * (abs(self.angles_from_obstacle[self.eligible_indices[1]]) < 90) + \
                           (abs(self.angles_from_obstacle[self.eligible_indices[0]]) > 90) * (abs(self.angles_from_obstacle[self.eligible_indices[1]]) > 90)
        in_same_quadrant = (in_same_quadrant[:, 0]).astype(int)

        # get the obstacle distance offset
        distance_from_obstacle_offset = abs(self.distances_from_obstacle[self.eligible_indices[0]] - self.distances_from_obstacle[self.eligible_indices[1]])
        # distance_from_obstacle_offset[~in_same_quadrant] = distance_from_obstacle_offset[~in_same_quadrant] + np.max(distance_from_obstacle_offset)
        prior_distance_from_obstacle = self.distances_from_obstacle[self.eligible_indices[0]]
        post_distance_from_obstacle = self.distances_from_obstacle[self.eligible_indices[1]]

        # get the obstacle angle offset
        angle_from_obstacle_offset = abs(self.angles_from_obstacle[self.eligible_indices[0]] - self.angles_from_obstacle[self.eligible_indices[1]])
        angle_from_obstacle_offset[angle_from_obstacle_offset > 180] = abs(360 - angle_from_obstacle_offset[angle_from_obstacle_offset > 180])
        prior_angle_from_obstacle = self.angles_from_obstacle[self.eligible_indices[0]]
        post_angle_from_obstacle = self.angles_from_obstacle[self.eligible_indices[1]]

        # get the length of the bout
        prior_length = self.shelter_distance_geodesic[self.eligible_indices[0]] - self.shelter_distance_geodesic_end[self.eligible_indices[0]]

        # get the difference in time
        time_offset = (self.start_indices[self.eligible_indices[1]] - self.start_indices[self.eligible_indices[0]]) / self.start_indices[self.eligible_indices[1]]

        # add the bonus for being stimulus triggered
        prior_homing_stimulus_evoked = self.stimulus_evoked[self.eligible_indices[0]].astype(bool)


        # z-score and compile the offsets into a data-point x feature array
        self.input_array = np.zeros((len(position_offset), self.input.shape[1]))
        self.input_array[:, 0] = position_offset
        self.input_array[:, 1] = HD_offset
        # self.input_array[:, 2] = position_offset * HD_offset
        self.input_array[:, 3] = distance_from_obstacle_offset[:,0]
        self.input_array[:, 4] = prior_homing_stimulus_evoked
        self.input_array[:, 5] = position_offset * prior_homing_stimulus_evoked
        self.input_array[:, 6] = HD_offset * prior_homing_stimulus_evoked
        self.input_array[:, 7] = distance_from_obstacle_offset[:, 0] * prior_homing_stimulus_evoked
        # self.input_array[:, 8] = time_offset #* position_offset


        # self.input_array = self.input_array[~prior_homing_stimulus_evoked, :]


        # add to the cross-session input
        self.input = np.concatenate( (self.input, self.input_array))

    def compute_input_dual(self):

        # get the position offsets
        position_offset = np.sqrt( (self.start_position1[0][self.eligible_indices[0]] - self.start_position2[0][self.eligible_indices[1]])**2 + \
                                   (self.start_position1[1][self.eligible_indices[0]] - self.start_position2[1][self.eligible_indices[1]])**2 )

        # normalize by trial
        position_array = np.ones_like(self.eligible_pairs) * np.nan
        position_array[self.eligible_indices[0], self.eligible_indices[1]] = position_offset
        position_array = position_array / np.nanmean(position_array, axis = 0)
        position_offset = position_array[~np.isnan(position_array)]

        shelter_distance_offset = self.shelter_distance_end1[self.eligible_indices[0]] - self.shelter_distance2[self.eligible_indices[1]]
        prior_distance_to_shelter = self.shelter_distance_end1[self.eligible_indices[0]]
        post_distance_to_shelter = self.shelter_distance_end2[self.eligible_indices[1]]

        # get the HD offsets
        # HD_offset = abs(self.path_direction1[self.eligible_indices[0]] - self.head_direction2[self.eligible_indices[1]])
        HD_offset = abs(self.head_direction1[self.eligible_indices[0]] - self.head_direction2[self.eligible_indices[1]])
        HD_offset[HD_offset > 180] = abs(360 - HD_offset[HD_offset > 180])

        # get the obstacle distance offset
        distance_from_obstacle_offset = abs(self.distances_from_obstacle1[self.eligible_indices[0]] - self.distances_from_obstacle2[self.eligible_indices[1]])
        prior_distance_from_obstacle = self.distances_from_obstacle1[self.eligible_indices[0]]
        post_distance_from_obstacle = self.distances_from_obstacle2[self.eligible_indices[1]]

        # given that they're in the same quandrant...
        in_same_quadrant = (self.angles_from_obstacle1[self.eligible_indices[0]] == self.angles_from_obstacle2[self.eligible_indices[1]]) + \
                           (abs(self.angles_from_obstacle1[self.eligible_indices[0]]) < 90) * (abs(self.angles_from_obstacle2[self.eligible_indices[1]]) < 90) + \
                           (abs(self.angles_from_obstacle1[self.eligible_indices[0]]) > 90) * (abs(self.angles_from_obstacle2[self.eligible_indices[1]]) > 90)
        in_same_quadrant = (in_same_quadrant[:, 0]).astype(int)


        # get the obstacle angle offset
        angle_from_obstacle_offset = abs(self.angles_from_obstacle1[self.eligible_indices[0]] - self.angles_from_obstacle2[self.eligible_indices[1]])
        angle_from_obstacle_offset[angle_from_obstacle_offset > 180] = abs(360 - angle_from_obstacle_offset[angle_from_obstacle_offset > 180])
        prior_angle_from_obstacle = self.angles_from_obstacle1[self.eligible_indices[0]]
        post_angle_from_obstacle = self.angles_from_obstacle2[self.eligible_indices[1]]

        # get the difference in time
        time_offset = abs(self.start_indices2[self.eligible_indices[1]] - self.start_indices1[self.eligible_indices[0]]) / self.start_indices2[self.eligible_indices[1]]

        # add the bonus for being stimulus triggered
        prior_homing_stimulus_evoked = self.stimulus_evoked1[self.eligible_indices[0]].astype(bool)

        # z-score and compile the offsets into a data-point x feature array
        self.input_array = np.zeros((len(position_offset), self.input.shape[1]))
        self.input_array[:, 0] = position_offset
        self.input_array[:, 1] = HD_offset
        self.input_array[:, 2] = prior_homing_stimulus_evoked
        self.input_array[:, 3] = time_offset
        self.input_array[:, 4] = distance_from_obstacle_offset[:, 0]

        # self.input_array = self.input_array[~prior_homing_stimulus_evoked, :]

        # add to the cross-session input
        self.input = np.concatenate( (self.input, self.input_array))

    def standardize(self):


        # equalize number of stim and non-stim pairs
        stim_trials_to_add = int(np.sum(self.input[:, 4]==0) - np.sum(self.input[:, 2]))
        stim_trials = int(np.sum(self.input[:, 4]))

        if stim_trials_to_add > 0:
            stim_indices = np.where(self.input[:, 4])[0]
            non_stim_indices = np.where(self.input[:, 4]==0)[0]

            self.indices_to_add = np.random.choice(stim_indices, stim_trials_to_add).astype(int)
            self.indices_to_keep = np.random.choice(non_stim_indices, stim_trials).astype(int)

            # self.responsibilities = np.concatenate((self.responsibilities, self.responsibilities[indices_to_add]))
            # self.input = np.concatenate((self.input, self.input[indices_to_add, :]) )

            self.responsibilities = np.concatenate((self.responsibilities[stim_indices], self.responsibilities[self.indices_to_keep]))
            self.input = np.concatenate((self.input[stim_indices, :], self.input[self.indices_to_keep, :]) )

        # standarize the inputs and save the mean and std
        self.input_means = np.mean(self.input, axis = 0)
        self.input_stds = np.std(self.input, axis = 0)

        self.input = (self.input - self.input_means) / (self.input_stds + .000001)

        self.input[:, 4] = (self.input[:, 4] > 0).astype(float)




    def fit_model(self):
        '''         USE THE INPUT VECTOR AND THE LIKELIHOOD OUTPUT TO COMPUTE WEIGHTS USING LINEAR REGRESSION       '''

        # convert to polynomial features
        # poly = PolynomialFeatures(2, interaction_only = True)
        # self.input_ = poly.fit_transform(self.input)
        self.input_ = self.input

        # self.input_[:, [1,2,3,4,5]] = 0

        # np.corrcoef(self.input[:, 0], self.responsibilities)[0][1]
        # np.corrcoef(self.input[:, 1], self.responsibilities)[0][1]
        # plt.figure()
        # plt.scatter(self.input[:, 0], self.responsibilities)
        # plt.pause(3)


        # try different alpha values
        # alpha = [.01, .005, .0025, .001, .0003, 0]
        # alpha = [.02, .015, .01, .0075, .005, .001, 0]
        alpha = [.01, .005, .0025, .001, 0]
        # alpha = [.005]
        alpha = [.001]
        for a in alpha:
            print('Alpha: ' + str(a))
            score = 0

            # split the data into a training set and a testing set
            trials = 40
            for r in range(trials):
                if self.time_to_fit:
                    X_train, X_test, y_train, y_test = train_test_split(self.input_, self.responsibilities, test_size=0.2, random_state=r)
                else:
                    # X_train, X_test, y_train, y_test = train_test_split(self.input_, self.responsibilities, test_size=0.99, random_state=r)
                    # X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.2, random_state=r)
                    X_test, y_test = self.input_, self.responsibilities

                # generate the LR lasso model
                if self.time_to_fit:
                    if a < 0: self.LR = linear_model.LogisticRegression()
                    elif a: self.LR = linear_model.Lasso(alpha=a)
                    else: self.LR = linear_model.LinearRegression()

                    # fit it to the data
                    self.LR.fit(X_train, y_train)

                # look at the results
                if a < 0:
                    coef_ = self.LR.coef_[0]
                    print('Correct by chance: ' + str(np.round( np.max( (np.mean(y_train), 1 - np.mean(y_train)) ), 2)) )
                else: coef_ = self.LR.coef_
                score += self.LR.score(X_test, y_test) / trials
            print(np.round(score, 2))
            print(self.LR.coef_)
            print('')
        print(np.round(self.LR.intercept_, 2))

        if self.time_to_fit:
            self.LR.fit(self.input_, self.responsibilities)
            np.save('C:\\Drive\\DLC\\transforms\\feature_values_' + self.obstacle_type + '.npy', [self.input_means, self.input_stds, self.LR.intercept_])
            joblib.dump(self.LR, 'C:\\Drive\\DLC\\transforms\\regression_' + self.obstacle_type)

    def compute_cross_mouse_DIC(self):

        pass

    def get_stim_indices(self, session_num = 0):
        '''         GET INDICES WHEN A STIMULUS WAS JUST PLAYED         '''
        # choose the loaded session to use
        if not session_num: session = self.session
        elif session_num == 1: session = self.session1
        elif session_num == 2: session = self.session2

        # initialize list
        wall_stim_idx = np.array([])
        no_wall_stim_idx = np.array([])

        # get the stimuli
        stim_frames = []
        for stim_type, stims in session['Stimuli'].stimuli.items():
            for vid_num, stims_video in enumerate(stims):
                stim_frames.extend(stims[vid_num])
                if vid_num:
                    print('still need to make work for multiple videos!')

        # add each stim time and the following 10 seconds to the list
        for t, stim_frame in enumerate(stim_frames):
            # only include if there was a wall
            if session['Tracking']['Trial Types'][0][t]:
                wall_stim_idx = np.append(wall_stim_idx, np.arange(stim_frame - 100, stim_frame + 400))
            if not session['Tracking']['Trial Types'][0][t]:
                no_wall_stim_idx = np.append(no_wall_stim_idx, np.arange(stim_frame - 100, stim_frame + 400))

        return wall_stim_idx, no_wall_stim_idx





