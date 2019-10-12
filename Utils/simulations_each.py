import numpy as np
import cv2
import scipy
from Utils.obstacle_funcs import set_up_speed_colors
from Utils.registration_funcs import model_arena
from sklearn.externals import joblib
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import os
import imageio
import skfmm
import time
import cupy as cp
cp.cuda.Device(0).use()

'''

......                                                                       ......
..............                                                       ..............
...........................                             ...........................
...SIMULATE WHAT A MOUSE FOLLOWING EACH OF THE STRATEGIES WOULD DO.................
...........................                             ...........................
..............                                                       ..............
......                                                                       ......

'''


class simulate():
    '''
    ...SIMULATE A MOUSE ESCAPE...
    '''
    def __init__(self, coordinates, stim_frame, infomark_location, shelter_location, arena, obstacle_type,
                    subgoal_location, start_idx, end_idx, trial_type, stims_video, videoname, save_folder,
                    session_trials_plot_background, border_size, strategy = 'all', vid=0):

        # set up class variables
        self.height, self.width = arena.shape[0], arena.shape[1]
        self.arena, _, self.shelter_roi = model_arena((self.height, self.width), trial_type > 0, False, obstacle_type, simulate=True)
        self.obstacle_type = obstacle_type
        self.color_arena_initialize = cv2.cvtColor(self.arena.copy(), cv2.COLOR_GRAY2RGB)
        self.color_arena_initialize = cv2.cvtColor(self.color_arena_initialize, cv2.COLOR_RGB2BGR)
        self.current_speed = 0
        self.shelter_location = [int(a*b/1000) for a, b in zip(shelter_location, self.arena.shape)]
        self.body_angle = 0
        self.body_location = (0,0)
        self.large_mouse_mask = arena * 0
        self.model_mouse_mask = arena * 0
        self.model_mouse_mask_previous = arena * 0
        self.coordinates = coordinates
        self.distance_arena = np.load('C:\\Drive\\DLC\\transforms\\distance_arena_' + obstacle_type + '.npy')
        self.angle_arena = np.load('C:\\Drive\\DLC\\transforms\\angle_arena_' + obstacle_type + '.npy')
        self.stim_frame = stim_frame
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.obstacle_in_the_way = 0
        self.shelter_nearby = False
        self.stim_idx = self.get_stim_indices(stims_video)
        self.video_name = videoname
        self.save_folder = save_folder
        self.background = session_trials_plot_background
        self.border_size = border_size
        self.obstacle_increase = 0
        self.overall_offset = []
        self.predicted_weights = []
        self.start_position = []
        self.starting_indices = []
        self.achieved_subgoals = []
        self.only_subgoal = []
        self.intended_angle = 90
        self.subgoal_locations = [ [int(y[1]*self.arena.shape[0]/1000) for y in subgoal_location['sub-goals']],
                                   [int(x[0]*self.arena.shape[1]/1000) for x in subgoal_location['sub-goals']] ]
        self.distance_to_shelter = np.inf
        x, y = np.meshgrid(np.arange(-50, 51), np.arange(-50, 51))
        self.angle_from_center = ((np.angle((x) + (-y) * 1j, deg=True) + 180) * np.pi / 180 * 99 / (2 * np.pi)).astype(int)
        self.ring = np.zeros((101, 101), np.uint8)
        cv2.circle(self.ring, (50, 50), 40, 1, 10)
        self.ring = self.ring.astype(bool)
        self.no_ring = ~self.ring
        self.degrees = np.zeros((100, 1))
        self.degrees[:, 0] = np.linspace(0, 2 * np.pi, 100)


        self.strategies = [strategy]

        # set parameters
        self.trial_duration = 300
        self.dist_to_whiskers = 0
        self.whisker_radius = 40
        self.body_length = 16
        self.large_body_length = 40
        self.max_shelter_proximity = 190 #120 #180
        self.arrived_at_shelter_proximity = 55
        self.angle_std = 13.8
        self.kappa = 1 / (np.deg2rad(self.angle_std) ** 2)

        # commence simulation
        self.main()





    def main(self):
        '''        RUN THE SIMULATION        '''

        # pick strategies to use
        if self.strategies == ['all']:
            self.strategies = ['homing_vector', 'vector_repetition', 'target_repetition', 'experience_model', 'geodesic_model']

        # loop across all strategies
        for i, strategy in enumerate(self.strategies):
            self.strategy = strategy

            # initialize mouse position and arena array
            self.initialize_mouse_position(self.stim_frame)
            self.color_arena = self.color_arena_initialize.copy()
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            vid = cv2.VideoWriter(os.path.join(self.save_folder, self.video_name + self.strategy + '.avi'), fourcc, 30,
                                         (self.width, self.height), True)

            # initialize environmental model
            if 'model' in self.strategy or 'repetition' in self.strategy:
                self.initialize_model()

            # loop across post-stimulus frames
            for self.frame_num in range(self.stim_frame, self.stim_frame + self.trial_duration):

                # make the mouse masks
                self.make_mouse_masks()

                # once at shelter, end it
                # if np.sum(self.shelter_roi * self.model_mouse_mask): break
                if self.distance_to_shelter < self.arrived_at_shelter_proximity: break

                # shade in the mouse
                self.shade_in_mouse()

                # show the previous frame
                cv2.imshow(self.strategies[i], self.color_arena)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                # update the position by each strategy

                if self.strategy == 'homing_vector': self.homing_vector()
                if self.strategy == 'vector_repetition': self.vector_repetition()
                if self.strategy == 'target_repetition': self.target_repetition()
                # if self.strategy == 'experience_model': self.experience_model()
                if self.strategy == 'geodesic_model': self.geodesic_model()
                # except:
                #     print('strategy failed')
                #     break
                try: vid.write(self.point_array)
                except: vid.write(self.color_arena)

            vid.release()
            self.save_image(self.strategies[i])





    def vector_repetition(self):
        '''        UPDATE ANGLE AND POSITION ACCORDING TO THE PATH REPETITION STRATEGY      '''

        # check if there is an obstacle in the way and adjust accordingly
        self.obstacle_guidance()

        # check if nearby to shelter and adjust accordingly
        self.check_for_shelter()

        # check if enough distance has passed to perform a new orienting movement
        self.time_for_new_orienting_movement()

        # if not, find the repeated vector to the shelter
        if not (self.obstacle_in_the_way or self.shelter_nearby):

            # use the state-action table to determine a which previous paths are worthy of being mimicked
            if self.its_time_to_orient:
                self.get_previous_homing_offsets()

            # determine the angle to turn based on which previous paths are most similar
            self.determine_vector_repetition_angle()

            # now shift the position according to the strategy
            self.move_body()


    def target_repetition(self):
        '''        UPDATE ANGLE AND POSITION ACCORDING TO THE TARGET REPETITION STRATEGY      '''

        # check if there is an obstacle in the way and adjust accordingly
        self.obstacle_guidance()

        # check if nearby to shelter and adjust accordingly
        self.check_for_shelter()

        # check if enough distance has passed to perform a new orienting movement
        self.time_for_new_orienting_movement()

        # use the state-action table to determine a which previous paths are worthy of being mimicked
        if self.its_time_to_orient:
            self.get_previous_homing_offsets()

        # if not, find the repeated sub-goal to get to the shelter
        if not (self.obstacle_in_the_way or self.shelter_nearby):

            # determine the sub-goal to go to based on which previous paths most similar
            self.determine_target_angle(type='end point')

            # now shift the position according to the strategy
            self.move_body()


    def experience_model(self):
        '''         UPDATE ANGLE AND POSITION ACCORDING TO THE EXPERIENCE MODEL STRATEGY        '''

        # check if nearby to shelter and adjust accordingly
        self.check_for_shelter()

        # if not, find the repeated vector to the shelter
        if not (self.shelter_nearby):

            # get utility value of previous homing start points
            if self.frame_num == self.stim_frame: self.get_previous_homing_utility()

            # determine the angle to turn based on which previous paths are best
            self.determine_target_angle(type = 'start point')

            # check if there is an obstacle in the way and adjust accordingly
            self.obstacle_guidance()

            # if not, move according to the experience model
            if not self.obstacle_in_the_way:
                # now shift the position according to the strategy
                self.move_body()


    def homing_vector(self):
        '''        UPDATE ANGLE AND POSITION ACCORDING TO THE HOMING VECTOR STRATEGY        '''

        # check if there is an obstacle in the way and adjust accordingly
        self.obstacle_guidance()

        # if not, find the homing vector to the shelter
        if not self.obstacle_in_the_way:

            # find the angle with respect to the shelter
            self.intended_angle = self.angle_to_shelter()

            # find the angle needed to turn
            self.get_turn_angle()

            # now shift the position according to the strategy
            self.move_body()


    def geodesic_model(self):
        '''         UPDATE ANGLE AND POSITION ACCORDING TO THE GEODESIC MODEL STRATEGY        '''

        # check if nearby to shelter and adjust accordingly
        self.check_for_shelter()

        # check if there is an obstacle in the way and adjust accordingly
        self.obstacle_guidance()

        # if not, find the repeated vector to the shelter
        if not (self.obstacle_in_the_way or self.shelter_nearby):

            # determine the angle to turn based on the geodesic gradient
            self.evaluate_geodesic_gradient()

            # now shift the position according to the strategy
            self.move_body()









    def initialize_model(self):
        '''         GERNERATE THE GEODESIC MAP USED FOR THE INTERNAL MODEL      '''

        # initialize the map
        phi = np.ones_like(self.arena)

        # mask the map wherever there's an obstacle
        mask = (self.arena == 90)
        self.phi_masked = np.ma.MaskedArray(phi, mask)

        # get the geodesic map of distance from the shelter
        phi_from_shelter = self.phi_masked.copy()
        phi_from_shelter[self.shelter_location[1], self.shelter_location[0]] = 0
        self.distance_from_shelter = np.array(skfmm.distance(phi_from_shelter))

        # calculate the gradient along this map
        self.geodesic_gradient = np.gradient(self.distance_from_shelter)
        self.geodesic_gradient[0][abs(self.geodesic_gradient[0]) > 1.1] = 0
        self.geodesic_gradient[1][abs(self.geodesic_gradient[1]) > 1.1] = 0

        # get the geodesic map of distance from shelter, given one option blocked
        if np.sum(mask):
            # generate a new, hypothetical mask RIGHT
            mask_right = mask.copy()
            mask_right[np.min(np.where(mask)[0]), np.max(np.where(mask)[1]):] = 1
            phi_masked_right = np.ma.MaskedArray(phi, mask_right)

            # get the geodesic map of distance from the shelter
            phi_masked_right[self.shelter_location[1], self.shelter_location[0]] = 0
            self.distance_from_shelter_right = np.array(skfmm.distance(phi_masked_right))

            # calculate the gradient along this map
            self.geodesic_gradient_right = np.gradient(self.distance_from_shelter_right)
            self.geodesic_gradient_right[0][abs(self.geodesic_gradient_right[0]) > 1.1] = 0
            self.geodesic_gradient_right[1][abs(self.geodesic_gradient_right[1]) > 1.1] = 0

            # generate a new, hypothetical mask LEFT
            mask_left = mask.copy()
            mask_left[np.min(np.where(mask)[0]), :np.min(np.where(mask)[1])] = 1
            phi_masked_left = np.ma.MaskedArray(phi, mask_left)

            # get the geodesic map of distance from the shelter
            phi_masked_left[self.shelter_location[1], self.shelter_location[0]] = 0
            self.distance_from_shelter_left = np.array(skfmm.distance(phi_masked_left))

            # calculate the gradient along this map
            self.geodesic_gradient_left = np.gradient(self.distance_from_shelter_left)
            self.geodesic_gradient_left[0][abs(self.geodesic_gradient_left[0]) > 1.1] = 0
            self.geodesic_gradient_left[1][abs(self.geodesic_gradient_left[1]) > 1.1] = 0
        else:
            self.distance_from_shelter_left, self.distance_from_shelter_right = False, False





    def evaluate_geodesic_gradient(self):
        '''     CALCULATE THE GRADIENT OF THE GEODESIC MAP AT THE CURRENT LOCATION      '''

        if np.sum(self.distance_from_shelter_left):
            # get the angle of the gradient of the geodesic map
            dP_dx_right = -self.geodesic_gradient_right[1][self.body_location[1], self.body_location[0]]
            dP_dy_right = self.geodesic_gradient_right[0][self.body_location[1], self.body_location[0]]

            # the angle to turn to is the angle of the gradient
            self.intended_angle = np.angle(dP_dx_right + dP_dy_right * 1j, deg=True)

            # go to landmark biases the turn angle
            self.go_to_landmark()

            # get the angle to turn by
            self.get_turn_angle()

            # get the output from this scenario
            intended_angle_right = self.intended_angle.copy()
            turn_angle_right = self.current_turn_angle.copy()

            # get the total distance from this scenario
            distance_right = self.distance_from_shelter_right[self.body_location[1], self.body_location[0]]
            if not distance_right: distance_right = np.inf

            # get the angle of the gradient of the geodesic map
            dP_dx_left = -self.geodesic_gradient_left[1][self.body_location[1], self.body_location[0]]
            dP_dy_left = self.geodesic_gradient_left[0][self.body_location[1], self.body_location[0]]

            # the angle to turn to is the angle of the gradient
            self.intended_angle = np.angle(dP_dx_left + dP_dy_left * 1j, deg=True)

            # go to landmark biases the turn angle
            self.go_to_landmark()

            # get the angle to turn by
            self.get_turn_angle()

            # get the output from this scenario
            intended_angle_left = self.intended_angle.copy()
            turn_angle_left = self.current_turn_angle.copy()

            # get the total distance from this scenario
            distance_left = self.distance_from_shelter_left[self.body_location[1], self.body_location[0]]
            if not distance_left: distance_left = np.inf

            # decide which to use: left or right - currently set 2 pixels per degree
            intended_angles = [intended_angle_left, intended_angle_right]
            turn_angles = [turn_angle_left, turn_angle_right]
            total_costs = [distance_left + 2 * abs(turn_angle_left), distance_right + 2 * abs(turn_angle_right)]
            self.intended_angle = intended_angles[np.argmin(total_costs)]
            self.current_turn_angle = turn_angles[np.argmin(total_costs)]


        else:
            # get the angle of the gradient of the geodesic map
            dP_dx = -self.geodesic_gradient[1][self.body_location[1], self.body_location[0]]
            dP_dy = self.geodesic_gradient[0][self.body_location[1], self.body_location[0]]

            # the angle to turn to is the angle of the gradient
            self.intended_angle = np.angle(dP_dx + dP_dy * 1j, deg=True)

            # go to landmark biases the turn angle
            self.go_to_landmark()

            # get the angle to turn by
            self.get_turn_angle()




    def angle_to_shelter(self):
        '''         GET THE ANGLE TO THE SHELTER            '''

        # calculate the angle to the shelter from the current position
        angle_to_shelter = -np.degrees(np.arctan2(self.shelter_location[1] - self.body_location[1], self.shelter_location[0] - self.body_location[0]))
        return angle_to_shelter

    def go_to_landmark(self):
        '''        BIAS SUB-GOAL LOCATION BY SALIENT OBJECT LOCATION        '''

        # get the angle to the landmark
        current_angle_from_obstacle = self.angle_arena[self.body_location[1], self.body_location[0]]

        # get the angle to the shelter
        angle_to_shelter = self.angle_to_shelter()

        # get the deviation from the intended angle from the homing vector
        intended_minus_shelter = abs(self.intended_angle - angle_to_shelter)
        if intended_minus_shelter > 180: intended_minus_shelter = 360 - intended_minus_shelter

        # get the deviation from the obstacle angle from the homing vector
        obstacle_minus_shelter = abs(current_angle_from_obstacle - angle_to_shelter)
        if obstacle_minus_shelter > 180: obstacle_minus_shelter = 360 - obstacle_minus_shelter

        # adjust the turn angle accordingly, if it helps getting to the shelter
        if np.sign(self.intended_angle) != np.sign(current_angle_from_obstacle):
            current_angle_from_obstacle -= 360*np.sign(current_angle_from_obstacle)
        if obstacle_minus_shelter < intended_minus_shelter:
            self.intended_angle = (self.intended_angle * 3 + current_angle_from_obstacle * 1) / 4



    def obstacle_guidance(self):
        '''     SEE IF THERE IS AN OBSTACLE IN THE WAY        '''

        # first, make a circle to indicate whisker sensing
        whisking_center = (int(self.body_location[0] + self.dist_to_whiskers * np.cos(np.radians(self.body_angle))),
                           int(self.body_location[1] - self.dist_to_whiskers * np.sin(np.radians(self.body_angle))))
        whisking_circle = cv2.ellipse(self.arena * 0, whisking_center, (self.whisker_radius, self.whisker_radius), 180 - self.body_angle, 90, 270, 1, -1)

        # check if it is touching an obstacle
        obstacle_contact = whisking_circle * (self.arena < 255) * (self.arena > 0)
        current_obstacle_in_the_way = np.sum(obstacle_contact).astype(float)

        # get the increase in obstacle touching for decceleration
        self.obstacle_increase = current_obstacle_in_the_way - self.obstacle_in_the_way

        # get the direction to the obstacle -- if it's above, then dont do obstacle guidance?
        current_angle_from_obstacle = self.angle_arena[self.body_location[1], self.body_location[0]]

        # is the obstacle in the way?
        self.obstacle_in_the_way = current_obstacle_in_the_way * (current_angle_from_obstacle != 90)

        # also get the distance to the shelter
        self.distance_to_shelter = np.sqrt(
            (self.body_location[0] - self.shelter_location[0]) ** 2 + (self.body_location[1] - self.shelter_location[1]) ** 2)

        # if it is touching an obstacle, see whether there is more obstacle sensed on the left vs right, front vs back
        if self.obstacle_in_the_way:
            front_left_circle = cv2.ellipse(self.arena * 0, whisking_center, (self.whisker_radius, self.whisker_radius), 180 - self.body_angle, 90, 180, 1,-1)
            front_right_circle = cv2.ellipse(self.arena * 0, whisking_center, (self.whisker_radius, self.whisker_radius), 180 - self.body_angle, 180, 270,1, -1)
            back_left_circle = cv2.ellipse(self.arena * 0, whisking_center, (self.whisker_radius, self.whisker_radius), 180 - self.body_angle, 0, 90, 1, -1)
            back_right_circle = cv2.ellipse(self.arena * 0, whisking_center, (self.whisker_radius, self.whisker_radius), 180 - self.body_angle, 270, 360, 1,-1)

            front_left_contact = np.sum(front_left_circle * (self.arena < 255))
            front_right_contact = np.sum(front_right_circle * (self.arena < 255))
            back_left_contact = np.sum(back_left_circle * (self.arena < 255))
            back_right_contact = np.sum(back_right_circle * (self.arena < 255))
            all_whiskers = [front_left_contact, front_right_contact, back_left_contact, back_right_contact]

            # get the angle to turn by based on the obstacle sensing:
            # if front left is highest, turn right
            if np.argmax(all_whiskers) == 0:
                left_edge_radar = front_left_circle * (self.arena < 255)
                right_wall_radar = front_right_circle * (self.arena < 255)
                left_edge = np.sum(left_edge_radar[self.subgoal_locations])
                right_wall = np.sum(right_wall_radar) > 10

                if abs(self.intended_angle) < 60 or (left_edge and right_wall) :  # unless goal or edge is on the left
                    self.current_turn_angle = 90 * front_left_contact / front_right_contact
                else:
                    self.current_turn_angle = -90 * front_right_contact / front_left_contact
            # if front right is highest, turn left
            elif np.argmax(all_whiskers) == 1:
                right_edge_radar = front_right_circle * (self.arena < 255)
                left_wall_radar = front_left_circle * (self.arena < 255)
                right_edge = np.sum(right_edge_radar[self.subgoal_locations])
                left_wall = np.sum(left_wall_radar) > 10

                if abs(self.intended_angle) > 120 or (right_edge and left_wall):
                    self.current_turn_angle = -90 * front_right_contact / front_left_contact
                else:
                    self.current_turn_angle = 90 * front_left_contact / front_right_contact
            # if back left is highest, turn left
            if np.argmax(all_whiskers) == 2:
                if front_left_contact:
                    self.current_turn_angle = 0
                else:
                    self.current_turn_angle = 90 * front_left_contact / back_left_contact
            # if back right is highest, turn right
            elif np.argmax(all_whiskers) == 3:
                if front_right_contact:
                    self.current_turn_angle = 0
                else:
                    self.current_turn_angle = -90 * front_right_contact / back_right_contact

            # now shift the position according to the strategy
            # print(np.argmax(all_whiskers))
            self.move_body()

            # reorient when exiting the obstacle
            # self.its_time_to_orient = True
            # self.current_escape_bout_lengths[0] = 100 #[0 for f in range(self.trial_duration)]


            '''     TO DO: GET DISTRIBUTION OF TURNS BASED ON ANGLE OF INDICDENCE, OR JUST ADD A BIT OF NOISE TO THE ANGLE       '''








    def move_body(self):
        '''         MOVE BODY LOCATION IN THE DIRECTION OF BODY ANGLE           '''

        # now shift the body angle according to the strategy
        self.angular_speed = min(30, abs(self.current_turn_angle / ( 2 + 3 * ((self.frame_num - self.stim_frame) < 6) )))
        '''TO DO: ANGULAR SPEED FROM AN EMPIRICAL DISTRIBUTION (here, the '3' constant)'''

        # now shift the body angle according to the strategy
        self.body_angle = (self.body_angle + self.angular_speed * np.sign(self.current_turn_angle)) % 360

        # get the curent speed as a function of angle speed
        '''TO DO: GET ACCELERATION FROM AN EMPIRICAL DISTRIBUTION, PER STRATEGY'''
        self.current_acceleration = ((self.angular_speed < 10) * (11 - self.current_speed) + \
                                     (self.angular_speed > 10) * (2 - self.current_speed) + \
                                     (abs(self.obstacle_increase) > 10) * (2 - self.current_speed) ) / 3

        self.current_speed += self.current_acceleration

        # move body forward
        self.body_location[0] += int(self.current_speed * np.cos(np.radians(self.body_angle)))
        self.body_location[1] -= int(self.current_speed * np.sin(np.radians(self.body_angle)))

    def get_current_escape_bout_lengths(self):
        '''         GET THE DISTANCE BETWEEN EACH ORIENTING MOVEMENT OF THE CURRENT ESCAPE      '''
        # get the current amount of distance between orienting movements
        current_escape_epoch = self.start_idx[self.stim_frame: self.stim_frame + 600]
        current_escape_start_indices = np.where(current_escape_epoch)[0] + self.stim_frame
        current_escape_end_indices = current_escape_epoch[current_escape_start_indices - self.stim_frame]

        # get the distances in between bouts
        # self.current_escape_bout_lengths = [ np.sqrt( (self.coordinates['center_body_location'][0][e] - self.coordinates['center_body_location'][0][s])**2 + \
        #                                          (self.coordinates['center_body_location'][1][e] - self.coordinates['center_body_location'][1][s])**2 ) \
        #                                          for s,e in zip(current_escape_start_indices.astype(int), current_escape_end_indices.astype(int)) ]
        self.current_escape_bout_lengths = [20]

        # after the last one, don't update
        # self.current_escape_bout_lengths.append(np.inf)
        try: self.current_escape_bout_lengths[-1] = np.inf
        except: pass



    def time_for_new_orienting_movement(self):
        '''         CHECK IF ENOUGH DISTANCE HAS PASSED TO DO A NEW ORIENTING MOVEMENT       '''

        # get the index of the current path being followed
        distance_traveled = np.sqrt( (self.body_location[0] - self.starting_body_location[0])**2 + (self.body_location[1] - self.starting_body_location[1])**2 )

        # if the distance of a bout has been traveled
        if distance_traveled >= self.current_escape_bout_lengths[0]:
            self.its_time_to_orient = True
            # self.current_escape_bout_lengths[0] = 100 # all subsequent bouts must be at least 30 mm long
            # self.current_escape_bout += 1
            self.starting_body_location = self.body_location.copy().astype(float)



    def get_previous_homing_offsets(self):
        '''         DETERMINE THE PATH TO FOLLOW BASED ON THE STATE ACTION TABLE             '''
        # extract start-point and end-point indices
        self.starting_indices = np.where(self.start_idx[:self.stim_frame])[0]
        repetition_end_indices = self.start_idx[self.starting_indices].astype(int)

        # get the start and end position for each one
        self.start_position = self.coordinates['center_body_location'][0][self.starting_indices], self.coordinates['center_body_location'][1][self.starting_indices]
        self.end_position = self.coordinates['center_body_location'][0][repetition_end_indices], self.coordinates['center_body_location'][1][repetition_end_indices]

        # get the vector direction of the previous path
        path_direction = np.angle((self.end_position[0] - self.start_position[0]) + (-self.end_position[1] + self.start_position[1]) * 1j, deg=True)
        head_direction = self.coordinates['body_angle'][self.starting_indices]

        # get the position offset for each one
        position_offset = np.sqrt((self.start_position[0] - self.body_location[0]) ** 2 + (self.start_position[1] - self.body_location[1]) ** 2)

        # get the distance from the shelter
        shelter_distances = self.coordinates['distance_from_shelter'][repetition_end_indices]
        current_distance_to_shelter = np.sqrt( (self.body_location[0] - self.shelter_location[0])**2 + (self.body_location[1] - self.shelter_location[1])**2 )

        geodesic_shelter_distances_start = self.distance_from_shelter[self.start_position[1].astype(int), self.start_position[0].astype(int)]
        geodesic_shelter_distances = self.distance_from_shelter[self.end_position[1].astype(int), self.end_position[0].astype(int)]
        current_geodesic_shelter_distance = self.distance_from_shelter[self.body_location[1], self.body_location[0]]


        # get the HD-vs-vector-direction offset
        # HD_offset = abs(self.body_angle - path_direction)
        HD_offset = abs(self.body_angle - head_direction)
        HD_offset[HD_offset > 180] = abs(360 - HD_offset[HD_offset > 180])

        # get the distance from the obstacle
        distances_from_obstacle = self.distance_arena[self.start_position[1].astype(int), self.start_position[0].astype(int)]
        current_distance_from_obstacle = self.distance_arena[self.body_location[1], self.body_location[0]]
        distance_from_obstacle_offset = abs(current_distance_from_obstacle - distances_from_obstacle)

        # get the angle to the obstacle
        angles_from_obstacle = self.angle_arena[self.start_position[1].astype(int), self.start_position[0].astype(int)]
        current_angle_from_obstacle = self.angle_arena[self.body_location[1], self.body_location[0]]
        angle_from_obstacle_offset = abs(current_angle_from_obstacle - angles_from_obstacle)
        angle_from_obstacle_offset[angle_from_obstacle_offset > 180] = abs(360 - angle_from_obstacle_offset[angle_from_obstacle_offset > 180])

        # check if it was in the same section
        in_same_quadrant = (angles_from_obstacle == current_angle_from_obstacle) + \
                           (abs(angles_from_obstacle) < 90) * (abs(current_angle_from_obstacle) < 90) + \
                           (abs(angles_from_obstacle) > 90) * (abs(current_angle_from_obstacle) > 90)

        # get a measure of recency
        time_offset = (self.stim_frame - self.starting_indices) / (self.stim_frame)

        # see if it's stimulus-evoked or spontaneous
        stimulus_evoked = np.array([(i in self.stim_idx) for i in self.starting_indices])

        # remove the ineligible prior homings -------------------
        # get the change in euclidean shelter distance
        decrease_in_euclidean_distance = current_distance_to_shelter - shelter_distances

        # get the change in geodesic shelter distance
        decrease_in_geodesic_distance = current_geodesic_shelter_distance - geodesic_shelter_distances



        # get the turn angle
        turn_angle = abs(self.body_angle - path_direction)
        turn_angle[turn_angle > 180] = abs(360 - turn_angle[turn_angle > 180])

        # distance too great
        too_far = position_offset > (np.sqrt(np.sum(self.arena > 0)) / 4)
        too_far_short = position_offset > (np.sqrt(np.sum(self.arena > 0)) / 8)

        # do differently if there's no wall -- just too_far

        # different distances from obstacle
        # different_obstacle_positions = (current_distance_from_obstacle < 50) * (distances_from_obstacle > 150) + \
        #                                (current_distance_from_obstacle > 150) * (distances_from_obstacle < 50)

        # pairs to exclude #in_same_quadrant ~too_far+stimulus_evoked
        pairs_to_keep = ( ((~in_same_quadrant * ~too_far_short) + (in_same_quadrant * (~too_far + stimulus_evoked))) * \
                        ((decrease_in_euclidean_distance > 40) + (decrease_in_geodesic_distance > 40)) ).astype(bool) # * (turn_angle > 20)

        position_offset = position_offset / np.mean(position_offset[pairs_to_keep])

        # calculate the overall offset
        self.input = np.zeros((len(self.starting_indices),9))
        self.input[:, 0] = position_offset
        self.input[:, 1] = HD_offset
        # self.input[:, 2] = position_offset * HD_offset
        self.input[:, 3] = distance_from_obstacle_offset
        # self.input[:, 4] = stimulus_evoked
        self.input[:, 5] = stimulus_evoked * position_offset
        self.input[:, 6] = stimulus_evoked * HD_offset
        self.input[:, 7] = stimulus_evoked * distance_from_obstacle_offset
        # self.input[:, 8] = time_offset

        # standardize the variables
        self.input = (self.input - self.feature_values[0]) / (self.feature_values[1] + .000001)
        self.input[:, 4] = (self.input[:, 4]>0).astype(float)

        # poly = PolynomialFeatures(2, interaction_only=True)
        # self.input = poly.fit_transform(self.input)

        # use model to compute model output
        # self.predicted_probabilities = self.LR.predict_proba(self.input)[:,1]
        try:
            self.predicted_weights = self.LR.predict(self.input)

            # don't use the ineligible pairs
            self.predicted_weights[~pairs_to_keep] = 0
            self.predicted_weights[self.predicted_weights<0] = 0

            # don't include sub-goals that have already been achieved
            self.predicted_weights[self.achieved_subgoals] = 0

            # only include sub-goals that came from the experience model
            self.predicted_weights[self.only_subgoal] = 1

            # stick with this evaluation until its time to orient again
            # self.its_time_to_orient = False

            # show what's going on in this mouse's head
            self.show_points()
        except:
            print('homing')
            self.homing_vector()














    def show_points(self):
        '''         SHOW ALL POTENTIAL REPETITION VECTORS        '''
        self.point_array = self.color_arena.copy()
        eligible_paths = np.where(self.predicted_weights)[0]
        arrow_color = np.array([230, 230, 230])

        if self.strategy == 'vector_repetition':

            for path_num in eligible_paths:
                # get the path locations
                point1 = int(self.start_position[0][path_num]), int(self.start_position[1][path_num])
                point2 = int(self.end_position[0][path_num]), int(self.end_position[1][path_num])
                confidence = 1
                if self.predicted_weights[path_num] > .07: confidence = 1.4
                if self.predicted_weights[path_num] > .09: confidence = 1.6
                if self.predicted_weights[path_num] > .11: confidence = 2.4
                cv2.arrowedLine(self.point_array, point1, point2, arrow_color/confidence, thickness=int(np.round(confidence)), tipLength=.1)

            # target direction
            path_directions = np.angle((self.end_position[0][eligible_paths] - self.start_position[0][eligible_paths]) + \
                                         (-self.end_position[1][eligible_paths] + self.start_position[1][eligible_paths]) * 1j, deg=True)

            # get their angles
            mu = (path_directions + 180) * np.pi / 180

            # compute the von mises features (normalized so range is 0 to 1)
            degree_features = np.tile(self.degrees, (1, len(mu)))
            vm_features = np.exp(self.kappa * np.cos(degree_features - mu)) / np.exp(self.kappa) * self.predicted_weights[eligible_paths]
            predicted_angle = np.sum(vm_features, 1)
            # if self.its_time_to_orient:
            self.intended_angle = self.degrees[np.argmax(predicted_angle)] * 180 / np.pi - 180

            # use the predicted weights to add them together
            # plt.plot(self.degrees, predicted_angle , color='green', linewidth = 5)
            # plt.plot(self.degrees, vm_features, color='blue')

            # show a circle indicating predicted intended angles
            desired_angle_square = (predicted_angle[self.angle_from_center] / np.max(predicted_angle) * 255).astype(np.uint8)
            desired_angle_square[self.no_ring] = 0
            desired_angle_square = cv2.applyColorMap(desired_angle_square, cv2.COLORMAP_JET)

            point_array_roi = self.point_array[self.body_location[1]-50:self.body_location[1]+51,
                                               self.body_location[0]-50:self.body_location[0]+51]

            point_array_roi[self.ring] = desired_angle_square[self.ring]

            self.point_array[self.body_location[1] - 50:self.body_location[1] + 51,
                            self.body_location[0] - 50:self.body_location[0] + 51] = point_array_roi

        if self.strategy == 'target_repetition':
            # initialize arrays
            self.heat_map = cp.zeros((self.color_arena.shape[0], self.color_arena.shape[1]))
            dist = cp.zeros((self.color_arena.shape[0], self.color_arena.shape[1], 2), np.int16)
            x, y = cp.meshgrid(cp.arange(self.color_arena.shape[1]), cp.arange(self.color_arena.shape[0]))

            # draw an arrowed line
            for path_num in eligible_paths:

                # get the path locations
                point1 = int(self.start_position[0][path_num]), int(self.start_position[1][path_num])
                point2 = int(self.end_position[0][path_num]), int(self.end_position[1][path_num])

                # make the map of distance from path endpoint
                dist[:, :, 0] = x - point2[0]
                dist[:, :, 1] = y - point2[1]
                dist_map = cp.linalg.norm(dist, axis=2)

                # draw a heat map of target locations
                heat_map = cp.zeros_like(self.heat_map)
                heat_map[int(self.end_position[1][path_num]), int(self.end_position[0][path_num])] = \
                    self.predicted_weights[path_num] * 255

                # give it the appropriate amount of blurring
                target_distance = np.sqrt((self.end_position[0][path_num] - self.body_location[0]) ** 2 + (self.end_position[1][path_num] - self.body_location[1]) ** 2)
                spatial_std = 2 * target_distance * np.tan( np.deg2rad(self.angle_std/2) )
                heat_map = cp.exp(-( dist_map**2 / ( 2.0 * spatial_std **2 ) ) )

                # add to the overall map
                self.heat_map += heat_map / cp.max(heat_map) * self.predicted_weights[path_num]

            # make the blurred heat map for this path
            heat_map_zero = cp.asnumpy(self.heat_map == 0)
            heat_map = cp.asnumpy(255 - self.heat_map * 255 / np.max(self.heat_map)).astype(np.uint8)
            heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_PINK)

            # show it
            alpha = .1
            cv2.addWeighted(self.point_array, alpha, heat_map, 1 - alpha, 0, self.point_array)
            self.point_array[heat_map_zero] = self.color_arena[heat_map_zero]
            self.point_array[self.color_arena[:,:,0] < 255] = self.color_arena[self.color_arena[:,:,0] < 255]

        cv2.imshow('coordinate', self.point_array)




















    def get_previous_homing_utility(self):
        '''         DETERMINE THE PATH TO FOLLOW BASED ON THE ACTION UTILITY TABLE             '''

        # extract start-point indices
        self.starting_indices = np.where(self.start_idx[:self.stim_frame])[0]

        # get the position and HD for each one
        self.start_position = np.zeros((2, len(self.starting_indices)))
        self.start_position[0] = self.coordinates['center_body_location'][0][self.starting_indices]
        self.start_position[1] = self.coordinates['center_body_location'][1][self.starting_indices]

        # get the final position of each path
        repetition_end_indices = self.start_idx[self.starting_indices].astype(int)
        repetition_end_points = self.coordinates['center_body_location'][0][repetition_end_indices], self.coordinates['center_body_location'][1][repetition_end_indices]

        # set current location as 0 contour
        phi_from_self = self.phi_masked.copy()
        phi_from_self[self.body_location[1], self.body_location[0]] = 0

        # get distances from current location
        distance_from_body_location = np.array(skfmm.distance(phi_from_self))

        # get the distance to each startpoint
        distance_to_subgoal = distance_from_body_location[self.start_position[1].astype(int), self.start_position[0].astype(int)]

        # get the distance from the shelter of each endpoint
        distance_from_shelter = self.distance_from_shelter[repetition_end_points[1].astype(int), repetition_end_points[0].astype(int)]

        # make sure it gets you close enough to do a DV
        distance_from_shelter[distance_from_shelter > self.max_shelter_proximity] = np.inf
        distance_from_shelter[distance_from_shelter <= self.max_shelter_proximity] = 0

        # get the angle to each starting point and add a penalty per degree turn required
        angle_to_start_point = np.angle((self.start_position[0] - self.body_location[0]) + (-self.start_position[1] + self.body_location[1]) * 1j, deg=True)

        # get the distance from the start of the repeated path to its end
        distance_along_path = np.sqrt( (repetition_end_points[0] - self.start_position[0])**2 + (repetition_end_points[1] - self.start_position[1])**2 ) \
                                        + abs(angle_to_start_point)*2

        # give distance from shelter a high weighting
        self.predicted_weights = 1 / (distance_to_subgoal + distance_along_path + distance_from_shelter)




    def determine_vector_repetition_angle(self):
        '''         DETERMINE THE ANGLE TO TURN BASED ON THE STATE ACTION TABLE             '''

        # sort start indices by overall offset
        # sorted_start_indices = np.argsort(overall_offset)
        # best_start_index = np.argmax(self.predicted_weights)
        #
        # # for now, just choose the closest start point to mimic
        # ''' -------------  TO DO: INTEGRATE MULTIPLE PATHS THAT CAN BE REPEATED'''
        # repetition_start_point = (self.start_position[0][best_start_index], self.start_position[1][best_start_index])
        # repetition_end_index = int(self.start_idx[self.starting_indices][best_start_index])
        # repetition_end_point = (
        #     self.coordinates['center_body_location'][0][repetition_end_index], self.coordinates['center_body_location'][1][repetition_end_index])
        #
        # # get the distance to the end point
        # repetition_end_indices = self.start_idx[self.starting_indices].astype(int)
        # repetition_end_points_all = [
        #     self.coordinates['center_body_location'][0][repetition_end_indices], self.coordinates['center_body_location'][1][repetition_end_indices] ]
        #
        # distance_to_subgoals = np.sqrt((repetition_end_points_all[0] - self.body_location[0]) ** 2 + (repetition_end_points_all[1] - self.body_location[1]) ** 2)

        # switch path once the subgoal is attained (PATH REPETITION) -- just a convencience for the simulation
        # achieved_subgoals = np.where(distance_to_subgoals < 20)[0]
        # self.achieved_subgoals.extend(achieved_subgoals)
        # self.achieved_subgoals = list(np.unique(self.achieved_subgoals))

        # show the point we're mimicking
        # self.show_point(repetition_start_point, repetition_end_point)

        # get the vector direction and distance to travel
        # self.intended_angle = np.angle((repetition_end_point[0] - repetition_start_point[0]) + (-repetition_end_point[1] + repetition_start_point[1]) * 1j, deg=True)

        #

        # get the angle to turn by
        self.get_turn_angle()


    def determine_target_angle(self, type = 'end point'):
        '''         DETERMINE THE ANGLE TO TURN BASED ON THE ACTION ENDPOINT TABLE             '''

        # sort start indices by overall offset
        # sorted_start_indices = np.argsort(overall_offset)
        best_start_index = np.argmax(self.predicted_weights)

        # for now, just choose the closest start point to mimic
        ''' -------------  TO DO: INTEGRATE MULTIPLE PATHS THAT CAN BE REPEATED'''
        if type == 'end point':
            target_point = np.flip(np.unravel_index(int(cp.argmax(self.heat_map)), self.heat_map.shape))

            # target_index = int(self.start_idx[self.starting_indices[best_start_index]])
            # target_point = (self.coordinates['center_body_location'][0][target_index], self.coordinates['center_body_location'][1][target_index])
            # repetition_start_point = (self.start_position[0][best_start_index], self.start_position[1][best_start_index])


            # get the distance to the end point
            distance_to_subgoal = np.sqrt((target_point[0] - self.body_location[0]) ** 2 + (target_point[1] - self.body_location[1]) ** 2)

            # switch sub-goals once the subgoal is attained (TARGET REPETITION) -- just a convencience for the simulation
            if distance_to_subgoal < 20 and not best_start_index in self.achieved_subgoals:
                self.achieved_subgoals.append(best_start_index)
                self.its_time_to_orient = True
                self.starting_body_location = self.body_location.copy().astype(float)
                # self.current_escape_bout_lengths[0] = 100

            # switch strategies once the subgoal is attained (EXPERIENCE MODEL) -- just a convencience for the simulation
            if distance_to_subgoal < 20 and 1 in self.predicted_weights:
                self.strategy = 'homing_vector'
            
        elif type == 'start point':
            target_point = (self.start_position[0][best_start_index], self.start_position[1][best_start_index])
            repetition_start_point = target_point

            # get the distance to the startpoint
            distance_to_subgoal = np.sqrt((self.start_position[0][best_start_index] - self.body_location[0]) ** 2 + (self.start_position[1][best_start_index] - self.body_location[1]) ** 2)

            # switch strategies once the subgoal is attained
            if distance_to_subgoal < 20:
                self.strategy = 'target_repetition'; self.only_subgoal = best_start_index

        # show the point we're mimicking
        # self.show_point(repetition_start_point, target_point)

        # get the vector direction and distance to travel
        # self.intended_angle = degrees[np.argmax(predicted_angle)] * 180 / np.pi - 180
        self.intended_angle = np.angle((target_point[0] - self.body_location[0]) + (-target_point[1] + self.body_location[1]) * 1j, deg=True)

        # go to landmark biases the turn angle
        self.go_to_landmark()

        # get the angle to turn by
        self.get_turn_angle()



    def get_turn_angle(self):
        '''         DETERMINE THE TURN ANGLE BASED ON THE TARGET ANGLE          '''

        # compute the amount needed to turn based on current and target angle
        self.current_turn_angle = (self.intended_angle - self.body_angle)
        if self.current_turn_angle > 180: self.current_turn_angle = 360 - self.current_turn_angle
        if self.current_turn_angle < -180: self.current_turn_angle = 360 + self.current_turn_angle


    def show_point(self, point1, point2):
        '''         SHOW A PARTICULAR COORDINATE        '''
        # initialize array
        self.point_array = self.color_arena.copy()

        # draw an arrowed line
        if point1 == point2:
            cv2.circle(self.point_array, tuple(int(x) for x in point1), 40, (255, 150, 150), 5)
        else:
            cv2.arrowedLine(self.point_array, tuple(int(x) for x in point1), tuple(int(x) for x in point2), (255, 150, 150), thickness=3, tipLength=.1)

        # show it
        cv2.imshow('coordinate', self.point_array)

    def save_image(self, starting_strategy):
        '''         SAVE THE IMAGE OF THE SIMULATED PATH        '''

        # Save to a new folder named after the experiment and the mouse with the word 'simulate'
        new_save_folder = os.path.join(self.save_folder + '_simulate')
        if not os.path.isdir(new_save_folder): os.makedirs(new_save_folder)

        # put the image in the background
        color_arena_in_background = self.background.copy()
        color_arena_in_background[self.border_size:, :-self.border_size] = self.color_arena

        # save the image
        imageio.imwrite(os.path.join(new_save_folder, self.video_name + '_' + starting_strategy + '.tif'),
                        cv2.cvtColor(color_arena_in_background, cv2.COLOR_BGR2RGB))

    def check_for_shelter(self):
        '''         CHECK IF THE SHELTER IS NEARBY; IF SO, SWITCH FROM REPETITION TO HOMING VECTOR          '''

        # get the distance to the shelter
        # self.distance_to_shelter = np.sqrt( (self.body_location[0] - self.shelter_location[0])**2 + (self.body_location[1] - self.shelter_location[1])**2 )

        # check if this distance is too close
        if self.distance_to_shelter < self.max_shelter_proximity:
            self.shelter_nearby = True
            self.homing_vector()
        else:
            self.shelter_nearby = False



    def shade_in_mouse(self):
        '''        SHADE IN THE MOUSE ON THE COLOR ARENA PLOT        '''

        # get the colors to use
        speed_color_light, speed_color_dark = set_up_speed_colors(self.current_speed, simulation=True)

        # add dark mouse if applicable
        if np.sum(self.large_mouse_mask * self.model_mouse_mask_previous) == 0:
            self.color_arena[self.model_mouse_mask.astype(bool)] = self.color_arena[self.model_mouse_mask.astype(bool)] * speed_color_dark
            self.model_mouse_mask_previous = self.model_mouse_mask
        # otherwise, shade in the current mouse position
        else:
            self.color_arena[self.model_mouse_mask.astype(bool)] = self.color_arena[self.model_mouse_mask.astype(bool)] * speed_color_light

    def initialize_mouse_position(self, frame_num):
        '''        GENERATE MASK CORRESPONDING TO THE MOUSE'S POSITION        '''
        # extract the location and angle from the coordinates
        self.body_location = self.coordinates['center_body_location'][:, frame_num].astype(np.uint16)
        self.body_angle = self.coordinates['body_angle'][frame_num]
        self.current_speed = 0
        self.starting_body_location = self.body_location.copy().astype(float)
        self.get_current_escape_bout_lengths()
        self.its_time_to_orient = True
        self.current_escape_bout = 0
        self.distance_to_shelter = np.inf
        # self.feature_values = np.load('C:\\Drive\\DLC\\transforms\\feature_values_' + self.obstacle_type + '.npy')
        self.feature_values = np.load('C:\\Drive\\DLC\\transforms\\feature_values_wall.npy')
        # self.LR = joblib.load('C:\\Drive\\DLC\\transforms\\regression_' + self.obstacle_type)
        self.LR = joblib.load('C:\\Drive\\DLC\\transforms\\regression_wall')



    def make_mouse_masks(self):
        '''        MAKE MASKS REPRESENTING THE MOUSE POSITIONS        '''
        # set up variable
        body_location = tuple(self.body_location)

        # make a single large ellipse used to determine when do use the flight_color_dark
        self.large_mouse_mask = cv2.ellipse(self.arena*0, body_location, (int(self.large_body_length), int(self.large_body_length * 3 / 5)), 180 - self.body_angle,0, 360, 100, thickness=-1)

        # calculate the straightforward body part locations
        head_location, front_location, nack_location = list(body_location), list(body_location), list(body_location)

        head_location[0] += int(1.2 * self.body_length * np.cos(np.radians(self.body_angle)))
        head_location[1] -= int(1.2 * self.body_length * np.sin(np.radians(self.body_angle)))

        front_location[0] += int(1 * self.body_length * np.cos(np.radians(self.body_angle)))
        front_location[1] -= int(1 * self.body_length * np.sin(np.radians(self.body_angle)))

        nack_location[0] += int(.8 * self.body_length * np.cos(np.radians(self.body_angle)))
        nack_location[1] -= int(.8 * self.body_length * np.sin(np.radians(self.body_angle)))

        # make the mouse ellipse
        model_mouse_mask = cv2.ellipse(self.arena*0, body_location, (int(self.body_length * .9), int(self.body_length * .5)), 180 - self.body_angle, 0, 360, 100, thickness=-1)
        model_mouse_mask = cv2.ellipse(model_mouse_mask, tuple(head_location), (int(self.body_length * .6), int(self.body_length * .3)), 180 - self.body_angle, 0,360, 100, thickness=-1)
        model_mouse_mask = cv2.ellipse(model_mouse_mask, tuple(front_location), (int(self.body_length * .5), int(self.body_length * .33)), 180 - self.body_angle, 0, 360, 100,thickness=-1)
        self. model_mouse_mask = cv2.ellipse(model_mouse_mask, tuple(nack_location), (int(self.body_length * .7), int(self.body_length * .38)), 180 - self.body_angle, 0, 360, 100,thickness=-1)

    def get_stim_indices(self, stims_video):
        '''         GET INDICES WHEN A STIMULUS WAS JUST PLAYED         '''
        # initialize list
        stim_idx = np.array([])

        # add each stim time and the following 10 seconds to the list
        for stim_frame in stims_video:
            stim_idx = np.append(stim_idx, np.arange( stim_frame - 100, stim_frame + 400) )

        return stim_idx