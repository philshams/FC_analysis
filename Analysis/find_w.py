import numpy as np
import os
import pickle
import scipy.stats
from sklearn import linear_model
from sklearn.model_selection import train_test_split
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

        # # initialize the input and likelihood
        # self.input = np.zeros((0, 6))
        # self.likelihood = np.array([])
        #
        # # loop across pairs of sessions, adding to the input and likelihood
        # for session_name1 in db.index:
        #     for session_name2 in db.index:
        #
        #         # check if the session is selected
        #         self.session1 = db.loc[session_name1]
        #         self.session2 = db.loc[session_name2]
        #         selected1 = check_session_selected(self.session1.Metadata, selector_type, selector)
        #         selected2 = check_session_selected(self.session2.Metadata, selector_type, selector)
        #
        #         if selected1 and selected2 and session_name1 != session_name2:
        #             print(session_name1 + ' and ' + session_name2)
        #
        #             self.extract_coordinates_dual()
        #
        #             self.compute_quantities_dual()
        #
        #             self.compute_eligibility_dual()
        #
        #             self.compute_likelihood_dual()
        #
        #             self.compute_input_dual()
        #
        # self.fit_model()

        # initialize the input and likelihood
        self.input = np.zeros((0, 6))
        self.likelihood = np.array([])

        # loop across sessions, adding to the input and likelihood
        for session_name in db.index:

            # check if the session is selected
            self.session = db.loc[session_name]
            selected = check_session_selected(self.session.Metadata, selector_type, selector)

            # create the feature array and likelihood output for each selected session
            if selected:
                try:
                    print(session_name)

                    self.extract_coordinates()

                    self.compute_quantities()

                    self.compute_eligibility()
                except:
                    print('failure')
                    continue

                self.compute_likelihood()

                self.compute_input()

                print('success')

        # fit the model to the data from all selected sessions
        self.fit_model()
        print('done')




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
        stim_idx = self.get_stim_indices()
        self.stimulus_evoked = np.array([(i in stim_idx) for i in self.start_indices])

        # get the HV direction
        self.HV_direction = self.coordinates['shelter_angle'][self.start_indices]

        # get the arena
        self.arena, _, _ = model_arena((self.distance_arena.shape[0], self.distance_arena.shape[1]), 0, False, self.obstacle_type, simulate=False)

        # set the error of the gaussian orientation movement generation model
        self.angle_std = 20



    def compute_quantities_dual(self):
        '''         COMPUTE THE QUANTITIES NEEDED TO CALCULATE THE DIFFERENCE IN INITIAL CONDITIONS FUNCTION        '''

        # extract start-point and end-point frame number ------------------------------
        self.start_indices1 = np.where(self.coordinates1['start_index'])[0]
        self.end_indices1 = self.coordinates1['start_index'][self.start_indices1].astype(int)

        # get the position
        self.start_position1 = self.coordinates1['center_body_location'][0][self.start_indices1], self.coordinates1['center_body_location'][1][self.start_indices1]
        self.end_position1 = self.coordinates1['center_body_location'][0][self.end_indices1], self.coordinates1['center_body_location'][1][self.end_indices1]

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
        stim_idx1 = self.get_stim_indices(session_num = 1)
        self.stimulus_evoked1 = np.array([(i in stim_idx1) for i in self.start_indices1])

        # get the HV direction
        self.HV_direction1 = self.coordinates1['shelter_angle'][self.start_indices1]


        # extract start-point and end-point frame number ------------------------------
        self.start_indices2 = np.where(self.coordinates2['start_index'])[0]
        self.end_indices2 = self.coordinates2['start_index'][self.start_indices2].astype(int)

        # get the position
        self.start_position2 = self.coordinates2['center_body_location'][0][self.start_indices2], self.coordinates2['center_body_location'][1][self.start_indices2]
        self.end_position2 = self.coordinates2['center_body_location'][0][self.end_indices2], self.coordinates2['center_body_location'][1][self.end_indices2]

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
        stim_idx2 = self.get_stim_indices(session_num = 2)
        self.stimulus_evoked2 = np.array([(i in stim_idx2) for i in self.start_indices2])

        # get the HV direction
        self.HV_direction2 = self.coordinates2['shelter_angle'][self.start_indices2]

        # get the arena
        self.arena, _, _ = model_arena((self.distance_arena.shape[0], self.distance_arena.shape[1]), 0, False, self.obstacle_type, simulate=False)

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

        # get the angle similarity matrix
        different_obstacle_angle = plus_90 * minus_90.T + minus_90 * plus_90.T
        similar_obstacle_angle = ~different_obstacle_angle

        # get the distance similarity arrays
        next_to_obstacle = self.distances_from_obstacle < 50
        far_from_obstacle = self.distances_from_obstacle > 100
        mid_range_from_obstacle = (self.distances_from_obstacle >= 50) * (self.distances_from_obstacle <= 100)

        # get the distance similarity matrix
        similar_obstacle_distance = next_to_obstacle * next_to_obstacle.T + far_from_obstacle * far_from_obstacle.T + \
                                    mid_range_from_obstacle * np.ones_like(mid_range_from_obstacle).T + np.ones_like(mid_range_from_obstacle) * mid_range_from_obstacle.T

        # get the eligibility matrix -- each row is the prior homing, and each column is the posterior homing
        eligible_pairs = similar_obstacle_angle * similar_obstacle_distance

        # only include stimulus driven ones
        eligible_pairs = eligible_pairs * self.stimulus_evoked

        # make it an upper triangular matrix, and remove the identity band
        self.eligible_pairs = ((np.triu(eligible_pairs) - np.identity(eligible_pairs.shape[0])) > 0 ).astype(bool)
        # self.eligible_pairs = (np.triu(eligible_pairs) ).astype(bool)
        self.eligible_indices = np.where(self.eligible_pairs)

        # low enough absolute distance
        position_offset = np.sqrt( (self.start_position[0][self.eligible_indices[0]] - self.start_position[0][self.eligible_indices[1]])**2 + \
                                   (self.start_position[1][self.eligible_indices[0]] - self.start_position[1][self.eligible_indices[1]])**2 )

        too_far = position_offset > (np.sqrt(np.sum(self.arena > 0)) / 2.5 )
        self.eligible_pairs[self.eligible_indices[0][too_far], self.eligible_indices[1][too_far]] = 0
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
        # get the angle similarity arrays
        plus_902 = self.angles_from_obstacle2 == 90
        minus_902 = self.angles_from_obstacle2 == -90


        # get the angle similarity matrix
        different_obstacle_angle = plus_901 * minus_902.T + minus_901 * plus_902.T
        similar_obstacle_angle = ~different_obstacle_angle


        # get the distance similarity arrays
        next_to_obstacle1 = self.distances_from_obstacle1 < 50
        far_from_obstacle1 = self.distances_from_obstacle1 > 100
        mid_range_from_obstacle1 = (self.distances_from_obstacle1 >= 50) * (self.distances_from_obstacle1 <= 100)
        # get the distance similarity arrays
        next_to_obstacle2 = self.distances_from_obstacle2 < 50
        far_from_obstacle2 = self.distances_from_obstacle2 > 100
        mid_range_from_obstacle2 = (self.distances_from_obstacle2 >= 50) * (self.distances_from_obstacle2 <= 100)

        # get the distance similarity matrix
        similar_obstacle_distance = next_to_obstacle1 * next_to_obstacle2.T + far_from_obstacle1 * far_from_obstacle2.T + \
                                    mid_range_from_obstacle1 * np.ones_like(mid_range_from_obstacle2).T + np.ones_like(mid_range_from_obstacle1) * mid_range_from_obstacle2.T

        # get the eligibility matrix -- each row is the prior homing, and each column is the posterior homing
        eligible_pairs = similar_obstacle_angle * similar_obstacle_distance

        # only include stimulus driven ones
        eligible_pairs = eligible_pairs * self.stimulus_evoked2

        # make it an upper triangular matrix, and remove the identity band
        # self.eligible_pairs = ((np.triu(eligible_pairs) - np.identity(eligible_pairs.shape[0])) > 0 ).astype(bool)
        self.eligible_pairs = (np.triu(eligible_pairs) ).astype(bool)
        self.eligible_indices = np.where(self.eligible_pairs)

        # low enough absolute distance
        position_offset = np.sqrt( (self.start_position1[0][self.eligible_indices[0]] - self.start_position2[0][self.eligible_indices[1]])**2 + \
                                   (self.start_position1[1][self.eligible_indices[0]] - self.start_position2[1][self.eligible_indices[1]])**2 )
        too_far = position_offset > (np.sqrt(np.sum(self.arena > 0)) / 2.5)
        self.eligible_pairs[self.eligible_indices[0][too_far], self.eligible_indices[1][too_far]] = 0
        self.eligible_indices = np.where(self.eligible_pairs)




    def compute_likelihood(self):
        '''         CALCULATE THE ANGLE PREDICTED BY EACH PREVIOUS ESCAPE       '''

        # initialize dual-strategy array
        self.strategy_angle = np.zeros((len(self.eligible_indices[0]), 2))
        self.strategy_error = np.zeros((len(self.eligible_indices[0]), 2))

        # get the angle from PR
        path_repetition_angle = self.path_direction[self.eligible_indices[0]]
        self.strategy_angle[:, 0] = path_repetition_angle

        # get the endpoint from TR
        current_startpoint = self.start_position[0][self.eligible_indices[1]], self.start_position[1][self.eligible_indices[1]]
        previous_endpoint = self.end_position[0][self.eligible_indices[0]], self.end_position[1][self.eligible_indices[0]]

        # get the angle from TR
        target_repetition_angle = np.angle((previous_endpoint[0] - current_startpoint[0]) + (-previous_endpoint[1] + current_startpoint[1]) * 1j, deg=True)
        self.strategy_angle[:, 1] = target_repetition_angle

        # get the actual angle
        path_angle = self.path_direction[self.eligible_indices[1]]

        # get the error in the angle
        self.strategy_error[:, 0] = (path_angle - path_repetition_angle)
        self.strategy_error[:, 1] = (path_angle - target_repetition_angle)

        self.strategy_error[self.strategy_error > 180] = 360 - self.strategy_error[self.strategy_error > 180]
        self.strategy_error[self.strategy_error < -180] = 360 + self.strategy_error[self.strategy_error < -180]

        # get the best angle among PR and TR
        winning_strategy = np.argmin(abs(self.strategy_error), 1)
        # repetition_angle = [self.strategy_angle[i, s] for i, s in enumerate(winning_strategy)]
        repetition_error = np.array([self.strategy_error[i, s] for i, s in enumerate(winning_strategy)])

        # --only include homings where there is at least one match from PR or TR--
        # take low-error repetitions
        self.repetition_possibly_used = abs(repetition_error) < 15

        # take the repetition that had low error
        flight_with_a_match = np.unique(self.eligible_indices[1][self.repetition_possibly_used])

        # set all flights preceding this repetition as applicable
        applicable_homings = np.zeros((1, len(self.start_indices)))
        applicable_homings[0, flight_with_a_match] = 1

        # update eligibility accordingly
        self.eligible_pairs = self.eligible_pairs * applicable_homings
        old_eligible_indices = self.eligible_indices
        self.eligible_indices = np.where(self.eligible_pairs)
        update_eligibility = [(t in self.eligible_indices[1]) for t in  old_eligible_indices[1]]

        # update which errors are used
        repetition_error = repetition_error[update_eligibility]

        # get the (normalized) likelihood that this was used, given our error model
        self.likelihood = np.concatenate( (self.likelihood, scipy.stats.norm(0, self.angle_std).pdf(repetition_error) / scipy.stats.norm(0, self.angle_std).pdf(0) ))


    def compute_likelihood_dual(self):
        '''         CALCULATE THE ANGLE PREDICTED BY EACH PREVIOUS ESCAPE       '''

        # initialize dual-strategy array
        self.strategy_angle = np.zeros((len(self.eligible_indices[0]), 2))
        self.strategy_error = np.zeros((len(self.eligible_indices[0]), 2))

        # get the angle from PR
        path_repetition_angle = self.path_direction1[self.eligible_indices[0]]
        self.strategy_angle[:, 0] = path_repetition_angle

        # get the endpoint from TR
        current_startpoint = self.start_position2[0][self.eligible_indices[1]], self.start_position2[1][self.eligible_indices[1]]
        previous_endpoint = self.end_position1[0][self.eligible_indices[0]], self.end_position1[1][self.eligible_indices[0]]

        # get the angle from TR
        target_repetition_angle = np.angle((previous_endpoint[0] - current_startpoint[0]) + (-previous_endpoint[1] + current_startpoint[1]) * 1j, deg=True)
        self.strategy_angle[:, 1] = target_repetition_angle

        # get the actual angle
        path_angle = self.path_direction2[self.eligible_indices[1]]

        # get the error in the angle
        self.strategy_error[:, 0] = (path_angle - path_repetition_angle)
        self.strategy_error[:, 1] = (path_angle - target_repetition_angle)

        self.strategy_error[self.strategy_error > 180] = 360 - self.strategy_error[self.strategy_error > 180]
        self.strategy_error[self.strategy_error < -180] = 360 + self.strategy_error[self.strategy_error < -180]

        # get the best angle among PR and TR
        winning_strategy = np.argmin(abs(self.strategy_error), 1)
        # repetition_angle = [self.strategy_angle[i, s] for i, s in enumerate(winning_strategy)]
        repetition_error = np.array([self.strategy_error[i, s] for i, s in enumerate(winning_strategy)])

        # --only include homings where there is at least one match from PR or TR--
        # take low-error repetitions
        self.repetition_possibly_used = abs(repetition_error) < 15

        # take the repetition that had low error
        flight_with_a_match = np.unique(self.eligible_indices[1][self.repetition_possibly_used])

        # set all flights preceding this repetition as applicable
        applicable_homings = np.zeros((1, len(self.start_indices2)))
        applicable_homings[0, flight_with_a_match] = 1

        # update eligibility accordingly
        self.eligible_pairs = self.eligible_pairs * applicable_homings
        old_eligible_indices = self.eligible_indices
        self.eligible_indices = np.where(self.eligible_pairs)
        update_eligibility = [(t in self.eligible_indices[1]) for t in  old_eligible_indices[1]]

        # update which errors are used
        repetition_error = repetition_error[update_eligibility]

        # get the (normalized) likelihood that this was used, given our error model
        self.likelihood = np.concatenate( (self.likelihood, scipy.stats.norm(0, self.angle_std).pdf(repetition_error) / scipy.stats.norm(0, self.angle_std).pdf(0) ))




    def compute_input(self):

        # get the position offsets
        position_offset = np.sqrt( (self.start_position[0][self.eligible_indices[0]] - self.start_position[0][self.eligible_indices[1]])**2 + \
                                   (self.start_position[1][self.eligible_indices[0]] - self.start_position[1][self.eligible_indices[1]])**2 )

        # get the HD offsets
        HD_offset = abs(self.path_direction[self.eligible_indices[0]] - self.head_direction[self.eligible_indices[1]])
        HD_offset[HD_offset > 180] = 360 - HD_offset[HD_offset > 180]

        # get the obstacle distance offset
        distance_from_obstacle_offset = abs(self.distances_from_obstacle[self.eligible_indices[0]] - self.distances_from_obstacle[self.eligible_indices[1]])

        # get the obstacle angle offset
        angle_from_obstacle_offset = abs(self.angles_from_obstacle[self.eligible_indices[0]] - self.angles_from_obstacle[self.eligible_indices[1]])
        angle_from_obstacle_offset[angle_from_obstacle_offset > 180] = 360 - angle_from_obstacle_offset[angle_from_obstacle_offset > 180]

        # get the difference in time
        time_offset = self.start_indices[self.eligible_indices[1]] - self.start_indices[self.eligible_indices[0]]

        # add the bonus for being stimulus triggered
        prior_homing_stimulus_evoked = self.stimulus_evoked[self.eligible_indices[0]].astype(int)

        # z-score and compile the offsets into a data-point x feature array
        self.input_array = np.zeros((len(position_offset), 6))
        self.input_array[:, 0] = position_offset #scipy.stats.zscore(position_offset)
        self.input_array[:, 1] = HD_offset #scipy.stats.zscore(HD_offset)
        self.input_array[:, 2:3] = distance_from_obstacle_offset #scipy.stats.zscore(distance_from_obstacle_offset)
        self.input_array[:, 3:4] = angle_from_obstacle_offset #scipy.stats.zscore(angle_from_obstacle_offset)
        self.input_array[:, 4] = prior_homing_stimulus_evoked #scipy.stats.zscore(prior_homing_stimulus_evoked)
        self.input_array[:, 5] = time_offset #scipy.stats.zscore(time_offset)

        self.input_array[np.isnan(self.input_array)] = 0

        # add to the cross-session input
        self.input = np.concatenate( (self.input, self.input_array))


    def compute_input_dual(self):

        # get the position offsets
        position_offset = np.sqrt( (self.start_position1[0][self.eligible_indices[0]] - self.start_position2[0][self.eligible_indices[1]])**2 + \
                                   (self.start_position1[1][self.eligible_indices[0]] - self.start_position2[1][self.eligible_indices[1]])**2 )

        # get the HD offsets
        HD_offset = abs(self.path_direction1[self.eligible_indices[0]] - self.head_direction2[self.eligible_indices[1]])
        HD_offset[HD_offset > 180] = 360 - HD_offset[HD_offset > 180]

        # get the obstacle distance offset
        distance_from_obstacle_offset = abs(self.distances_from_obstacle1[self.eligible_indices[0]] - self.distances_from_obstacle2[self.eligible_indices[1]])

        # get the obstacle angle offset
        angle_from_obstacle_offset = abs(self.angles_from_obstacle1[self.eligible_indices[0]] - self.angles_from_obstacle2[self.eligible_indices[1]])
        angle_from_obstacle_offset[angle_from_obstacle_offset > 180] = 360 - angle_from_obstacle_offset[angle_from_obstacle_offset > 180]

        # get the difference in time
        time_offset = abs(self.start_indices2[self.eligible_indices[1]] - self.start_indices1[self.eligible_indices[0]])

        # add the bonus for being stimulus triggered
        prior_homing_stimulus_evoked = self.stimulus_evoked1[self.eligible_indices[0]].astype(int)

        # z-score and compile the offsets into a data-point x feature array
        self.input_array = np.zeros((len(position_offset), 6))
        self.input_array[:, 0] = position_offset #scipy.stats.zscore(position_offset)
        self.input_array[:, 1] = HD_offset #scipy.stats.zscore(HD_offset)
        self.input_array[:, 2:3] = distance_from_obstacle_offset #scipy.stats.zscore(distance_from_obstacle_offset)
        self.input_array[:, 3:4] = angle_from_obstacle_offset #scipy.stats.zscore(angle_from_obstacle_offset)
        self.input_array[:, 4] = prior_homing_stimulus_evoked #scipy.stats.zscore(prior_homing_stimulus_evoked)
        self.input_array[:, 5] = time_offset #scipy.stats.zscore(time_offset)

        self.input_array[np.isnan(self.input_array)] = 0

        # add to the cross-session input
        self.input = np.concatenate( (self.input, self.input_array))







    def fit_model(self):
        '''         USE THE INPUT VECTOR AND THE LIKELIHOOD OUTPUT TO COMPUTE WEIGHTS USING LINEAR REGRESSION       '''

        # standarize the inputs and save the mean and std
        input_means = np.mean(self.input, axis = 0)
        input_stds = np.std(self.input, axis = 0)



        # try different alpha values
        alpha = [.02, .01, .0001, 0]
        for a in alpha:
            print('Alpha: ' + str(a))

            # split the data into a training set and a testing set
            for r in range(2):
                X_train, X_test, y_train, y_test = train_test_split(self.input, self.likelihood, test_size=0.2, random_state=r)

                # generate the LR lasso model
                if a: LR = linear_model.Lasso(alpha=a)
                else: LR = linear_model.LinearRegression()

                # fit it to the data
                LR.fit(X_train, y_train)

                # look at the results
                print('Train Score: ' + str(np.round(LR.score(X_train, y_train), 2)) )
                print('Test Score: ' + str(np.round(LR.score(X_test, y_test), 2)) )
                print('Coefficients:' )
                print('  ' + str(np.round(LR.coef_[0], 2)) + ': position offset')
                print('  ' + str(np.round(LR.coef_[1], 2)) + ': HD offset')
                print('  ' + str(np.round(LR.coef_[2], 2)) + ': distance from obstacle offset')
                print('  ' + str(np.round(LR.coef_[3], 2)) + ': angle from obstacle offset')
                print('  ' + str(np.round(LR.coef_[4], 2)) + ': previous homing was stimulus evoked')
                print('  ' + str(np.round(LR.coef_[5], 2)) + ': time offset')

                print('')
            print('')

    def compute_cross_mouse_DIC(self):

        pass




    def get_stim_indices(self, session_num = 0):
        '''         GET INDICES WHEN A STIMULUS WAS JUST PLAYED         '''
        # choose the loaded session to use
        if not session_num: session = self.session
        elif session_num == 1: session = self.session1
        elif session_num == 2: session = self.session2

        # initialize list
        stim_idx = np.array([])

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
                stim_idx = np.append(stim_idx, np.arange(stim_frame - 100, stim_frame + 400))

        return stim_idx





