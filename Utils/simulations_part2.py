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
        self.arena, _, self.shelter_roi = model_arena((self.height, self.width), trial_type, False, obstacle_type, simulate=True)
        self.obstacle_type = obstacle_type
        self.color_arena_initialize = cv2.cvtColor(self.arena.copy(), cv2.COLOR_GRAY2RGB)
        self.color_arena_initialize = cv2.cvtColor(self.color_arena_initialize, cv2.COLOR_RGB2BGR)
        self.color_arena = self.color_arena_initialize.copy()
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
        self.get_stim_indices(stims_video)
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
        self.trial_type = trial_type
        self.frozen_weights = False
        self.intended_angle = 90
        self.previous_strategy = ' '
        self.subgoal_locations = [ [int(y[1]*self.arena.shape[0]/1000) for y in subgoal_location['sub-goals']],
                                     [int(x[0]*self.arena.shape[1]/1000) for x in subgoal_location['sub-goals']] ]
        # self.subgoal_location = [(np.array(a) * b / 1000).astype(int) for a, b in zip(subgoal_location['sub-goals'], self.arena.shape)]
        self.distance_to_shelter = np.inf
        self.degrees = np.zeros((360, 1))
        self.degrees[:, 0] = np.linspace(0, 2 * np.pi, 360)
        x, y = np.meshgrid(np.arange(-44, 45), np.arange(-44, 45))
        self.angle_from_center = ((np.angle((x) + (-y) * 1j, deg=True) + 180) * np.pi / 180 * (len(self.degrees)-1) / (2 * np.pi)).astype(int)
        self.ring = np.zeros((89, 89), np.uint8)
        cv2.circle(self.ring, (44, 44), 40, 1, 10)
        self.ring = self.ring.astype(bool)
        self.no_ring = ~self.ring
        self.strategy_colors = {}
        self.strategy_colors['homing vector'] = (40, 40, 255)
        self.strategy_colors['obstacle guidance'] = (40, 40, 255)
        self.strategy_colors['repetition'] = (255, 100, 100)  #(20, 200, 240)
        # self.strategy_colors['vector_repetition'] = (255, 100, 100)
        self.strategy_colors['spatial planning'] = (40, 255, 40)
        self.strategy_colors['not escape'] = (120, 120, 120) #(20, 20, 20)

        self.strategies = [strategy]

        # set parameters
        self.trial_duration = 300
        self.dist_to_whiskers = 20
        self.whisker_radius = 50
        self.body_length = 16
        self.large_body_length = 40
        self.max_shelter_proximity = 190 #120 #180
        self.arrived_at_shelter_proximity = 55
        self.angle_std = 7 # 5 # 13.8
        self.kappa = 1 / (np.deg2rad(self.angle_std) ** 2)
        self.kappa_rand = 1 / (np.deg2rad(60) ** 2)

        # commence simulation
        self.main()





    def main(self):
        '''        RUN THE SIMULATION        '''

        # initialize mouse position and arena array
        # self.initialize_mouse_position(self.stim_frame)
        new_save_folder = os.path.join(self.save_folder + '_simulate')
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        vid = cv2.VideoWriter(os.path.join(new_save_folder, self.video_name + '_classifier.avi'), fourcc, 30, (self.width, self.height), True)
        plt.style.use('dark_background'); plt.rcParams.update({'font.size': 22})

        # initialize environmental model
        self.initialize_model()

        # get the body locations and angles and the orientation angles
        self.get_trial_info()

        self.initialize_mouse_position(self.stim_frame)

        # loop across post-stimulus frames
        for bout in range(len(self.TS_starting_indices)):
            self.strategies = ['homing_vector', 'spatial planning', 'repetition', 'not escape'] #'vector_repetition',
            self.bout = bout
            likelihoods = []
            self.calculating = True
            self.frozen_weights = False
            self.initialize_bout()
            self.color_arena_initialize = (( 1* cv2.cvtColor(self.arena.copy(), cv2.COLOR_GRAY2RGB).astype(float) + 3 * self.color_arena.astype(float) )/ 4).astype(np.uint8)

            plt.figure(figsize = (18,8))
            self.ax = plt.subplot(1,1,1)


            for i, strategy in enumerate(self.strategies):

                self.strategy = strategy

                if self.strategy == 'homing_vector':
                    self.homing_vector()
                    likelihoods.append(self.likelihood)
                # if self.strategy == 'vector_repetition':
                #     self.vector_repetition()
                #     likelihoods.append(self.likelihood)
                if self.strategy == 'repetition':
                    self.target_repetition()
                    likelihoods.append(self.likelihood)
                if self.strategy == 'spatial planning':
                    self.geodesic_model()
                    likelihoods.append(self.likelihood)


            self.ax.set_title('Likelihood of this Orientation Movement for Each Strategy'); self.ax.set_xlabel('allocentric head direction (degrees)');
            self.ax.set_ylabel('likelihood of HD given strategy')
            self.ax.set_xlim([0, 180]); self.ax.set_ylim([0, self.plot_height])
            self.ax.plot([self.degrees[self.current_path_idx] * 180 / np.pi, self.degrees[self.current_path_idx] * 180 / np.pi], [0, self.plot_height],color='white', linestyle='--')

            # get the likelihood of the random strategy
            mu = (self.TS_head_direction[self.bout] + 180) * np.pi / 180

            # compute the von mises features (normalized so range is 0 to 1)
            predicted_angle = np.exp(self.kappa_rand * np.cos(self.degrees - mu)) / np.exp(self.kappa_rand)
            predicted_angle = predicted_angle[:,0] / np.sum(predicted_angle)

            # get the likelihood
            self.likelihood = predicted_angle[self.current_path_idx]

            # predicted_angle = np.ones((len(self.degrees)))/len(self.degrees)
            likelihoods.append(self.likelihood)


            self.ax.fill_between(self.degrees[:,0] * 180 / np.pi, predicted_angle, color = 'grey', linewidth = 2, linestyle = '--', alpha = .2)

            leg = self.ax.legend(("mouse's choice", self.innate_strategy, 'spatial planning', 'target repetition', 'not escape'))
            leg.draggable()
            # plt.show()


            plt.savefig(os.path.join(new_save_folder, self.video_name + '_like' + str(self.bout+1) + '.tif'))
            plt.close('all')
            # automatically switch between obstacle guidance and homing vector (or both?..)

            # use the winning strategy
            self.strategies[0] = self.innate_strategy
            self.strategy = self.strategies[np.argmax(likelihoods)]
            if self.strategy == 'spatial planning' and self.same_as_HV:
                self.strategy = 'homing vector'

            # display the mouse running around with this representation in mind
            if not self.bout: start_frame = self.TS_starting_indices[0] - 30
            else:
                start_frame = self.TS_starting_indices[self.bout] - 10 * (self.previous_strategy != self.strategy)
            self.previous_strategy = self.strategy
            self.calculating = False
            # self.color_arena = self.color_arena_initialize.copy()
            for frame_num in range(start_frame, self.TS_end_indices[self.bout]):
                self.frame_num = frame_num
                
                # attained starting point?
                self.frozen_weights = (self.frame_num > self.TS_starting_indices[self.bout])

                # make the mouse masks
                self.make_mouse_masks()

                # shade in the mouse
                self.shade_in_mouse()

                # show the internal representation
                if self.strategy == 'homing vector':
                    self.homing_vector_representation()
                if self.strategy == 'obstacle guidance':
                    self.obstacle_guidance_representation()
                # if self.strategy == 'vector_repetition':
                #     self.vector_repetition_representation()
                if self.strategy == 'repetition':
                    self.target_repetition_representation()
                if self.strategy == 'spatial planning':
                    self.geodesic_model_representation()
                if self.strategy == 'not escape':
                    self.random_representation()

                # add text
                cv2.putText(self.rep_arena, self.strategy, (20,30), 0, .75, self.strategy_colors[self.strategy], thickness = 1 )


                if frame_num == self.TS_starting_indices[self.bout] or \
                    frame_num == self.TS_end_indices[self.bout]-1:

                    rep_arena_flash = self.rep_arena.copy()

                    if frame_num == self.TS_end_indices[self.bout] - 1:
                        _, newcontours, _ = cv2.findContours(self.model_mouse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(rep_arena_flash, newcontours, 0, self.strategy_colors[self.strategy], thickness=2)
                        cv2.drawContours(rep_arena_flash, contours, 0, self.strategy_colors[self.strategy], thickness=2)
                    else:
                        _, contours, _ = cv2.findContours(self.model_mouse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(rep_arena_flash, contours, 0, self.strategy_colors[self.strategy], thickness=2)



                    if frame_num == self.TS_starting_indices[self.bout]:
                        flash_on = np.tile(np.concatenate((np.zeros(10), np.ones(10), )), 3).astype(int)
                    else: flash_on = np.ones(30)

                    # pause here
                    for flash in flash_on:
                        # fade out the old trajectory

                        # but keep the current position / representation

                        if flash: rep_arena_show = rep_arena_flash
                        else: rep_arena_show = self.rep_arena

                        # show the previous frame
                        cv2.imshow('strategy cam', rep_arena_show)
                        vid.write(rep_arena_show)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                # show the previous frame
                cv2.imshow('strategy cam', self.rep_arena)
                vid.write(self.rep_arena)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            dotted_line_arena = self.rep_arena.copy()
            cv2.line(dotted_line_arena, (int(self.TS_start_position[0][self.bout]), int(self.TS_start_position[1][self.bout])),
                     (int(self.TS_end_position[0][self.bout]), int(self.TS_end_position[1][self.bout])), (5, 5, 5), 7)
            cv2.line(dotted_line_arena, (int(self.TS_start_position[0][self.bout]), int(self.TS_start_position[1][self.bout])),
                     (int(self.TS_end_position[0][self.bout]), int(self.TS_end_position[1][self.bout])), (255, 255, 255), 5)
            cv2.line(dotted_line_arena, (int(self.TS_start_position[0][self.bout]), int(self.TS_start_position[1][self.bout])),
                     (int(self.TS_end_position[0][self.bout]), int(self.TS_end_position[1][self.bout])), self.strategy_colors[self.strategy], 3)

            imageio.imwrite(os.path.join(new_save_folder, self.video_name + '_' + str(self.bout+1) + '.tif'), \
                            cv2.cvtColor(dotted_line_arena, cv2.COLOR_BGR2RGB))


        #     if self.distance_to_shelter < self.arrived_at_shelter_proximity: break
        #
        #     try: vid.write(self.rep_arena)
        #     except: vid.write(self.color_arena)
        #
        vid.release()
        # self.save_image(self.strategies[i])



    def homing_vector_representation(self):
        '''         DISPLAY THE HOMING VECTOR           '''

        # initialize arrays
        line_arena = self.arena*0
        self.rep_arena = self.color_arena.copy()
        # draw line
        cv2.line(line_arena, tuple(self.shelter_location), self.head_location, 1, 2)
        # shade in line
        self.rep_arena[line_arena.astype(bool)] = self.color_arena[line_arena.astype(bool)] * [.5,.5,1]
        # draw line
        cv2.line(line_arena, tuple(self.shelter_location), self.head_location, 1, 10)
        # shade in line
        self.rep_arena[line_arena.astype(bool)] = self.rep_arena[line_arena.astype(bool)] * [.9, .9, 1]


    def obstacle_guidance_representation(self):
        '''         DISPLAY THE OBSTACLE                '''

        # initialize arrays
        obstacle_arena = self.arena*0
        self.rep_arena = self.color_arena.copy()

        # shade in obstacle
        self.rep_arena[self.arena==90] = 255*np.array([.1,.1,1])


    def vector_repetition_representation(self):
        '''         DISPLAY THE PRIOR VECTOR            '''

        if not self.frozen_weights: self.get_previous_homing_offsets()
        self.show_vector_repetition()


    def target_repetition_representation(self):
        '''         DISPLAY THE PRIOR TARGETS           '''

        if not self.frozen_weights: self.get_previous_homing_offsets()
        self.show_target_repetition()


    def geodesic_model_representation(self):
        '''         DISPLAY THE SPATIAL PLANNING        '''

        # initialize arrays
        circle_arena = self.arena * 0
        self.rep_arena = self.color_arena.copy()

        # get colored distance map
        distance_from_shelter_bright = (250 * (self.distance_from_shelter / np.max(self.distance_from_shelter))**2 ).astype(np.uint8)
        distance_from_shelter_dim = (250 * (1 - self.distance_from_shelter / np.max(self.distance_from_shelter))**2 ).astype(np.uint8)

        dist_glow_arena = self.color_arena.copy()
        dist_glow_arena[:, :, 2] = 10
        dist_glow_arena[:, :, 1] = distance_from_shelter_dim #10 #distance_from_shelter_bright
        dist_glow_arena[:, :, 0] = 10 #np.min(dist_glow_arena[:, :, 1:], 2)

        # shade into rep map
        shade_in_zone = ((self.color_arena == 255)).astype(bool)
        self.rep_arena[shade_in_zone] = dist_glow_arena[shade_in_zone]

        cv2.imshow('strategy cam', self.rep_arena)


    def random_representation(self):
        '''         DISPLAY THE SPATIAL PLANNING        '''

        # initialize arrays
        circle_arena = self.arena * 0
        self.rep_arena = self.color_arena.copy()

        # get distance map
        dist = np.zeros((self.color_arena.shape[0], self.color_arena.shape[1], 2), np.int16)
        x, y = np.meshgrid(np.arange(self.color_arena.shape[1]), np.arange(self.color_arena.shape[0]))

        # make the map of distance from path endpoint
        dist[:, :, 0] = x - self.head_location[0]
        dist[:, :, 1] = y - self.head_location[1]
        dist_map = np.linalg.norm(dist, axis=2)


        # give it the appropriate amount of blurring
        circle_arena = (255* (1 - .7 * np.exp(-(dist_map ** 2 / (2.0 * 60 ** 2))))).astype(np.uint8)
        circle_arena = cv2.cvtColor(circle_arena, cv2.COLOR_GRAY2BGR)

        # shade in circle
        shade_in_zone = ((self.color_arena == 255) * (circle_arena < 254)).astype(bool)
        self.rep_arena[shade_in_zone] = circle_arena[shade_in_zone]


    # def vector_repetition(self):
    #     '''        UPDATE ANGLE AND POSITION ACCORDING TO THE PATH REPETITION STRATEGY      '''
    #
    #     # use the state-action table to determine a which previous paths are worthy of being mimicked
    #     self.get_previous_homing_offsets()
    #
    #     # show the results and compute the output
    #     self.show_vector_repetition()
    #

    def target_repetition(self):
        '''        UPDATE ANGLE AND POSITION ACCORDING TO THE TARGET REPETITION STRATEGY      '''

        # use the state-action table to determine a which previous paths are worthy of being mimicked
        self.get_previous_homing_offsets()

        # show the results and compute the output
        self.show_target_repetition()


    def homing_vector(self):
        '''        UPDATE ANGLE AND POSITION ACCORDING TO THE HOMING VECTOR STRATEGY        '''

        # check if there is an obstacle in the way and adjust accordingly
        self.obstacle_guidance()

        self.plot_height = .1

        # if not, find the homing vector to the shelter
        if not self.obstacle_in_the_way:

            # find the angle with respect to the shelter
            self.intended_angle = self.angle_to_shelter()
            distance_from_shelter = np.sqrt( (self.body_location[0]-self.shelter_location[0])**2 + (self.body_location[1]-self.shelter_location[1])**2)
            distance_mod = np.max((1, 600 / distance_from_shelter))

            # get the likelihood of the strategy
            mu = (self.intended_angle + 180) * np.pi / 180

            # compute the von mises features (normalized so range is 0 to 1)
            predicted_angle = np.exp(self.kappa / distance_mod * np.cos(self.degrees - mu)) / np.exp(self.kappa / distance_mod)
            predicted_angle = predicted_angle / np.sum(predicted_angle)

            # get the likelihood
            self.likelihood = predicted_angle[self.current_path_idx][0]
            # self.plot_height = 1.1 * np.max(predicted_angle)
            self.ax.fill_between(self.degrees[:,0]*180/np.pi, predicted_angle[:,0], color='darkred', linewidth=4, alpha = .55)

            self.innate_strategy = 'homing vector'





    def get_trial_info(self):
        '''         GET THE DEETS FOR THE CURRENT TRIAL         '''

        # get the starting and ending indices
        self.get_current_stim_indices()
        this_stimulus_idx = np.array([(i in self.current_stim_idx) for i in np.where(self.start_idx)[0]])
        self.TS_starting_indices = np.where(self.start_idx)[0][this_stimulus_idx]
        self.TS_end_indices = self.start_idx[self.TS_starting_indices].astype(int)

        # get the start and end position for each one
        # self.TS_start_position = self.coordinates['center_body_location'][0][self.TS_starting_indices], self.coordinates['center_body_location'][1][self.TS_starting_indices]
        # self.TS_end_position = self.coordinates['center_body_location'][0][self.TS_end_indices], self.coordinates['center_body_location'][1][self.TS_end_indices]

        self.TS_start_position = self.coordinates['head_location'][0][self.TS_starting_indices], self.coordinates['head_location'][1][self.TS_starting_indices]
        self.TS_end_position = self.coordinates['head_location'][0][self.TS_end_indices], self.coordinates['head_location'][1][self.TS_end_indices]


        # get the vector direction of the previous path
        self.TS_path_direction = np.angle((self.TS_end_position[0] - self.TS_start_position[0]) + (-self.TS_end_position[1] + self.TS_start_position[1]) * 1j, deg=True)
        self.TS_head_direction = self.coordinates['body_angle'][self.TS_starting_indices]

    def initialize_bout(self):
        '''         SET THE CURRENT POSITION AND ANGLE AS THAT OF THE BEGINNING OF THE CURRENT BOUT         '''

        self.body_location = int(self.TS_start_position[0][self.bout]), int(self.TS_start_position[1][self.bout])
        self.body_angle = self.TS_head_direction[self.bout]

        self.current_path_idx = int((self.TS_path_direction[self.bout] + 180) * len(self.degrees) / 360)

        self.strategies = ['homing vector', 'spatial planning', 'repetition', 'not escape']  # 'vector_repetition',
        self.bout = bout
        self.likelihoods = []
        self.calculating = True
        self.frozen_weights = False
        self.color_arena_initialize = (
                (1 * cv2.cvtColor(self.arena.copy(), cv2.COLOR_GRAY2RGB).astype(float) + 3 * self.color_arena.astype(float)) / 4).astype(np.uint8)

        plt.figure(figsize=(18, 8))
        self.ax = plt.subplot(1, 1, 1)


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





    def geodesic_model(self):
        '''     CALCULATE THE GRADIENT OF THE GEODESIC MAP AT THE CURRENT LOCATION      '''

        if np.sum(self.distance_from_shelter_left):
            # get the angle of the gradient of the geodesic map
            dP_dx_right = -self.geodesic_gradient_right[1][self.body_location[1], self.body_location[0]]
            dP_dy_right = self.geodesic_gradient_right[0][self.body_location[1], self.body_location[0]]

            # the angle to turn to is the angle of the gradient
            self.intended_angle = np.angle(dP_dx_right + dP_dy_right * 1j, deg=True)

            # get the angle to turn by
            self.get_turn_angle()

            # get the output from this scenario
            intended_angle_right = self.intended_angle.copy()
            turn_angle_right = self.current_turn_angle.copy()

            # get the total distance from this scenario
            distance_right = self.distance_from_shelter_right[self.body_location[1], self.body_location[0]]
            # get the sub-goal distance from this scenario
            distance_right = np.sqrt( (self.body_location[0] - self.subgoal_locations[1][0])**2 + (self.body_location[1] - self.subgoal_locations[0][0])**2)
            if not distance_right: distance_right = np.inf

            # get the angle of the gradient of the geodesic map
            dP_dx_left = -self.geodesic_gradient_left[1][self.body_location[1], self.body_location[0]]
            dP_dy_left = self.geodesic_gradient_left[0][self.body_location[1], self.body_location[0]]

            # the angle to turn to is the angle of the gradient
            self.intended_angle = np.angle(dP_dx_left + dP_dy_left * 1j, deg=True)

            # get the angle to turn by
            self.get_turn_angle()

            # get the output from this scenario
            intended_angle_left = self.intended_angle.copy()
            turn_angle_left = self.current_turn_angle.copy()

            # get the total distance from this scenario
            distance_left = self.distance_from_shelter_left[self.body_location[1], self.body_location[0]]
            # get the sub-goal distance from this scenario
            distance_left = np.sqrt((self.body_location[0] - self.subgoal_locations[1][1]) ** 2 + (self.body_location[1] - self.subgoal_locations[0][1]) ** 2)
            if not distance_left: distance_left = np.inf

            # decide which to use: left or right - currently set 2 pixels per degree
            path_directions = np.array([intended_angle_left, intended_angle_right])
            turn_angles = [turn_angle_left, turn_angle_right]
            total_costs = [(distance_left + 2 * abs(turn_angle_left))**4, (distance_right + 2 * abs(turn_angle_right))**4]
            self.intended_angle = path_directions[np.argmin(total_costs)]
            self.current_turn_angle = turn_angles[np.argmin(total_costs)]

            # for now, just set costs and probability weights as equal

            # get the angle to the landmark
            current_angle_from_obstacle = np.ones(len(path_directions)) * self.angle_arena[self.body_location[1], self.body_location[0]]

            # get the angle to the shelter
            angle_to_shelter = self.angle_to_shelter()
            distances_from_obstacle = self.distance_arena[self.body_location[1], self.body_location[0]]
            far_from_obstacle = distances_from_obstacle > 75


            # get the deviation from the intended angle from the homing vector
            intended_minus_shelter = abs(path_directions - angle_to_shelter)
            intended_minus_shelter[intended_minus_shelter > 180] = abs(360 - intended_minus_shelter[intended_minus_shelter > 180])

            # get the deviation from the obstacle angle from the homing vector
            obstacle_minus_shelter = abs(current_angle_from_obstacle[0] - angle_to_shelter)
            if obstacle_minus_shelter > 180: obstacle_minus_shelter = abs(360 - obstacle_minus_shelter)

            # adjust the turn angle accordingly, if it helps getting to the shelter
            shift_sign = (np.sign(path_directions) != np.sign(current_angle_from_obstacle) )
            current_angle_from_obstacle[shift_sign] = \
                current_angle_from_obstacle[shift_sign] - 360 * np.sign(current_angle_from_obstacle[shift_sign])
            shift_direction = (obstacle_minus_shelter < intended_minus_shelter) * far_from_obstacle
            # path_directions[shift_direction] = \
            #     (path_directions[shift_direction] * 7 + current_angle_from_obstacle[shift_direction] * 1) / 8

            # if almost at the edge, just do the homing vector -- switch to endpoint of bout?
            if np.min((distance_left, distance_right)) < 50: path_directions[:] = self.angle_to_shelter(); total_cost = [1,1]
        else:
            path_directions = np.array([self.angle_to_shelter()])
            total_costs = 1

        # get their angles
        mu = (path_directions + 180) * np.pi / 180

        # compute the von mises features (normalized so range is 0 to 1)
        degree_features = np.tile(self.degrees, (1, len(mu)))
        vm_features = np.exp(self.kappa/ 4 * np.cos(degree_features - mu)) / np.exp(self.kappa/4) / total_costs
        predicted_angle = np.sum(vm_features, 1)

        # use the predicted weights to add them together
        predicted_angle = predicted_angle / np.sum(predicted_angle)

        # get the likelihood and max L intended angle
        self.likelihood = predicted_angle[self.current_path_idx]
        self.intended_angle = self.degrees[np.argmax(predicted_angle)] * 180 / np.pi - 180

        if not int(self.intended_angle - self.angle_to_shelter()): self.same_as_HV = 1
        else: self.same_as_HV = 0

        self.ax.fill_between(self.degrees[:, 0] * 180 / np.pi, predicted_angle, color='mediumspringgreen', linewidth=4 - 3*self.same_as_HV, alpha=.3-.15*self.same_as_HV)



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
        if intended_minus_shelter > 180: intended_minus_shelter = abs(360 - intended_minus_shelter)

        # get the deviation from the obstacle angle from the homing vector
        obstacle_minus_shelter = abs(current_angle_from_obstacle - angle_to_shelter)
        if obstacle_minus_shelter > 180: obstacle_minus_shelter = abs(360 - obstacle_minus_shelter)

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

            # do edge detection
            left_edge_radar = front_left_circle * (self.arena < 255)
            left_edge = np.sum(left_edge_radar[self.subgoal_locations])

            right_edge_radar = front_right_circle * (self.arena < 255)
            right_edge = np.sum(right_edge_radar[self.subgoal_locations])

            # get the angle to turn by based on the obstacle sensing:
            # if front left is highest, turn right
            if np.argmax(all_whiskers) == 0:
                if abs(self.intended_angle) < 60 or (left_edge and self.body_angle < -45) :  # unless goal or edge is on the left
                    self.current_turn_angle = 90 * front_left_contact / front_right_contact
                else:
                    self.current_turn_angle = -90 * front_right_contact / front_left_contact
            # if front right is highest, turn left
            elif np.argmax(all_whiskers) == 1:
                if abs(self.intended_angle) > 120 or (right_edge and self.body_angle > -135 and self.body_angle <= 0):
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

            if (left_edge and self.body_angle < -45) or (right_edge and self.body_angle > -135 and self.body_angle <= 0) and self.angle_arena[int(self.TS_end_position[1][self.bout]), int(self.TS_end_position[0][self.bout])] > 0:
                self.obstacle_in_the_way = False
            else:
                # get likelihood of angle
                # target direction
                path_directions = np.angle((np.array(self.subgoal_locations[1]) - self.body_location[0]) + \
                                           (-np.array(self.subgoal_locations[0]) + self.body_location[1]) * 1j, deg=True)

                # # get the angle to the landmark
                # current_angle_from_obstacle = np.ones(len(path_directions)) * self.angle_arena[self.body_location[1], self.body_location[0]]
                #
                # # adjust the turn angle accordingly, if it helps getting to the shelter
                # shift_sign = (np.sign(path_directions) != np.sign(current_angle_from_obstacle))
                # current_angle_from_obstacle[shift_sign] = \
                #     current_angle_from_obstacle[shift_sign] - 360 * np.sign(current_angle_from_obstacle[shift_sign])
                #
                # # go to landmark
                # path_directions = (path_directions * 6 + current_angle_from_obstacle * 1) / 7

                # angle from obstacle
                current_angle_from_obstacle = self.angle_arena[self.body_location[1], self.body_location[0]]
                collision_angle = self.body_angle - current_angle_from_obstacle
                if collision_angle > 180: collision_angle = 360 - collision_angle
                if collision_angle < -180: collision_angle = 360 + collision_angle

                # for now just do weights corresponding to angle of indicence with arbitrary exp function
                if self.obstacle_in_the_way:
                    if collision_angle > 0: obstacle_weights = np.array([ np.exp(-abs(collision_angle)/3), 1])
                    else: obstacle_weights = np.array([1, np.exp(-abs(collision_angle) / 3)])
                else:
                    obstacle_weights = np.zeros(len(self.subgoal_locations[0]))

                # get their angles
                mu = (path_directions + 180) * np.pi / 180

                # compute the von mises features (normalized so range is 0 to 1)
                degree_features = np.tile(self.degrees, (1, len(mu)))
                vm_features = np.exp(self.kappa/1 * np.cos(degree_features - mu)) / np.exp(self.kappa/1) * obstacle_weights
                predicted_angle = np.sum(vm_features, 1)

                # use the predicted weights to add them together
                if np.sum(predicted_angle): predicted_angle = predicted_angle / np.sum(predicted_angle)
                self.ax.fill_between(self.degrees[:,0]*180/np.pi, predicted_angle, color='darkred', linewidth=4, alpha = .6)

                # get the likelihood and max L intended angle
                self.likelihood = predicted_angle[self.current_path_idx]

                self.intended_angle = self.degrees[np.argmax(predicted_angle)] * 180 / np.pi - 180

                self.innate_strategy = 'obstacle guidance'

        '''     TO DO: GET DISTRIBUTION OF TURNS BASED ON ANGLE OF INDICDENCE, OR JUST ADD A BIT OF NOISE TO THE ANGLE       '''






    def get_current_escape_bout_lengths(self):
        '''         GET THE DISTANCE BETWEEN EACH ORIENTING MOVEMENT OF THE CURRENT ESCAPE      '''
        # get the current amount of distance between orienting movements
        current_escape_epoch = self.start_idx[self.stim_frame: self.stim_frame + 600]
        current_escape_start_indices = np.where(current_escape_epoch)[0] + self.stim_frame
        current_escape_end_indices = current_escape_epoch[current_escape_start_indices - self.stim_frame]

        # get the distances in between bouts
        # self.current_escape_bout_lengths = [ np.sqrt( (self.coordinates['center_body_location'][0][e] - self.coordinates['center_body_location'][0][s])**2 + \
        #                                          (self.coordinates['center_body_location'][1][e] - self.coordinates['head_location'][1][s])**2 ) \
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
            # self.current_escape_bout_lengths[0] = 100 # all subsequent bouts must be at least 30 mm long
            # self.current_escape_bout += 1
            self.starting_body_location = self.body_location.copy().astype(float)



    def get_previous_homing_offsets(self):
        '''         DETERMINE THE PATH TO FOLLOW BASED ON THE STATE ACTION TABLE             '''
        # extract start-point and end-point indices
        self.starting_indices = np.where(self.start_idx[:self.stim_frame])[0]
        repetition_end_indices = self.start_idx[self.starting_indices].astype(int)

        # get the start and end position for each one
        self.start_position = self.coordinates['head_location'][0][self.starting_indices], self.coordinates['head_location'][1][self.starting_indices]
        self.end_position = self.coordinates['head_location'][0][repetition_end_indices], self.coordinates['head_location'][1][repetition_end_indices]

        # get the start and end position for each one
        self.start_position_body = self.coordinates['center_body_location'][0][self.starting_indices], self.coordinates['center_body_location'][1][self.starting_indices]
        self.end_position_body = self.coordinates['center_body_location'][0][repetition_end_indices], self.coordinates['center_body_location'][1][repetition_end_indices]

        prior_length = np.sqrt((self.start_position[0] - self.end_position[0])**2 + (self.start_position[1] - self.end_position[1])**2)

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
        distances_from_obstacle = self.distance_arena[self.start_position_body[1].astype(int), self.start_position_body[0].astype(int)]
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

        across_boundary = (angles_from_obstacle == -current_angle_from_obstacle) + \
                          (abs(angles_from_obstacle) < 90) * (abs(current_angle_from_obstacle) > 90) + \
                          (abs(angles_from_obstacle) > 90) * (abs(current_angle_from_obstacle) < 90)

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
        too_far_short = position_offset > (np.sqrt(np.sum(self.arena > 0)) / 4) # was 8
        too_far_long = position_offset > (np.sqrt(np.sum(self.arena > 0)) / 3)  # was 8

        # do differently if there's no wall -- just too_far
        if self.trial_type <= 0:
            too_far_short = too_far

        # pairs to exclude #in_same_quadrant ~too_far+stimulus_evoked
        pairs_to_keep = ( ( (~in_same_quadrant * ~too_far_short * ~across_boundary) + (in_same_quadrant * ~too_far) + \
                            (~in_same_quadrant * ~too_far_long * ~across_boundary * stimulus_evoked) + (in_same_quadrant * stimulus_evoked)  ) * \
                    (prior_length > 40) * ((decrease_in_euclidean_distance > 10) + (decrease_in_geodesic_distance > 10)) ).astype(bool) # * (turn_angle > 20)
        # was > 40

        position_offset = position_offset / np.mean(position_offset[pairs_to_keep])

        # calculate the overall offset
        self.input = np.zeros((len(self.starting_indices),9))
        self.input[:, 0] = position_offset
        # self.input[:, 1] = HD_offset
        # self.input[:, 2] = position_offset * HD_offset
        self.input[:, 3] = distance_from_obstacle_offset
        # self.input[:, 4] = stimulus_evoked
        self.input[:, 5] = stimulus_evoked * position_offset
        # self.input[:, 6] = stimulus_evoked * HD_offset
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
        except:
            self.predicted_weights = np.zeros(len(position_offset))















    #
    # def show_vector_repetition(self):
    #     '''         SHOW ALL POTENTIAL REPETITION VECTORS        '''
    #     self.rep_arena = self.color_arena.copy()
    #     eligible_paths = np.where(self.predicted_weights)[0]
    #     arrow_color = np.array([230, 230, 230])
    #
    #     for path_num in eligible_paths:
    #         # get the path locations
    #         point1 = int(self.start_position[0][path_num]), int(self.start_position[1][path_num])
    #         point2 = int(self.end_position[0][path_num]), int(self.end_position[1][path_num])
    #         confidence = 1
    #         if self.predicted_weights[path_num] > .07: confidence = 1.4
    #         if self.predicted_weights[path_num] > .09: confidence = 1.6
    #         if self.predicted_weights[path_num] > .11: confidence = 2.4
    #         cv2.arrowedLine(self.rep_arena, point1, point2, arrow_color/confidence, thickness=int(np.round(confidence)), tipLength=.1)
    #
    #     if eligible_paths.size:
    #         # target direction
    #         path_directions = np.angle((self.end_position[0][eligible_paths] - self.start_position[0][eligible_paths]) + \
    #                                      (-self.end_position[1][eligible_paths] + self.start_position[1][eligible_paths]) * 1j, deg=True)
    #
    #         # get their angles
    #         mu = (path_directions + 180) * np.pi / 180
    #
    #         # compute the von mises features (normalized so range is 0 to 1)
    #         degree_features = np.tile(self.degrees, (1, len(mu)))
    #         vm_features = np.exp(self.kappa * np.cos(degree_features - mu)) / np.exp(self.kappa) * self.predicted_weights[eligible_paths]
    #         predicted_angle = np.sum(vm_features, 1)
    #
    #         # use the predicted weights to add them together
    #         predicted_angle = predicted_angle / np.sum(predicted_angle)
    #
    #         # show a circle indicating predicted intended angles
    #         desired_angle_square = (predicted_angle[self.angle_from_center] / np.max(predicted_angle) * 255).astype(np.uint8)
    #         desired_angle_square[self.no_ring] = 0
    #         desired_angle_square = cv2.applyColorMap(desired_angle_square, cv2.COLORMAP_JET)
    #
    #         point_array_roi = self.rep_arena[self.body_location[1]-44:self.body_location[1]+45,
    #                                            self.body_location[0]-44:self.body_location[0]+45]
    #
    #         point_array_roi[self.ring] = desired_angle_square[self.ring]
    #
    #         self.rep_arena[self.body_location[1] - 44:self.body_location[1] + 45,
    #                         self.body_location[0] - 44:self.body_location[0] + 45] = point_array_roi
    #     else:
    #         predicted_angle = np.zeros(len(self.degrees))
    #
    #
    #     cv2.imshow('VR', self.rep_arena)
    #
    #     if self.calculating:
    #         # get the likelihood and max L intended angle
    #         self.likelihood = predicted_angle[self.current_path_idx]
    #
    #         self.ax.fill_between(self.degrees[:, 0] * 180 / np.pi, predicted_angle, color='royalblue', linewidth=4, alpha=.5)
    #         self.intended_angle = self.degrees[np.argmax(predicted_angle)] * 180 / np.pi - 180



    def show_target_repetition(self):
        '''         SHOW ALL POTENTIAL REPETITION TARGETS        '''

        self.rep_arena = self.color_arena.copy()

        if not self.frozen_weights:
            eligible_paths = np.where(self.predicted_weights)[0]

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
                spatial_std = 2 * target_distance * np.tan( np.deg2rad(2*self.angle_std/2) )
                heat_map = cp.exp(-( dist_map**2 / ( 2.0 * spatial_std **2 ) ) )

                # add to the overall map
                self.heat_map += heat_map / cp.max(heat_map) * self.predicted_weights[path_num]

        # make the blurred heat map for this path
        heat_map_zero = cp.asnumpy(self.heat_map == 0)
        heat_map = cp.asnumpy(255 - self.heat_map * 185 / np.max(self.heat_map)).astype(np.uint8)
        heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_OCEAN)

        # show it
        alpha = .1
        cv2.addWeighted(self.rep_arena, alpha, heat_map, 1 - alpha, 0, self.rep_arena)
        self.rep_arena[heat_map_zero] = self.color_arena[heat_map_zero]
        self.rep_arena[self.color_arena[:,:,0] < 255] = self.color_arena[self.color_arena[:,:,0] < 255]

        cv2.imshow('TR', self.rep_arena)

        # target direction
        if self.calculating and eligible_paths.size:
            path_directions = np.angle((self.end_position[0][eligible_paths] - self.body_location[0]) + \
                                       (-self.end_position[1][eligible_paths] + self.body_location[1]) * 1j, deg=True)
    
    
            # get the angle to the landmark
            current_angle_from_obstacle = np.ones(len(path_directions)) * self.angle_arena[self.body_location[1], self.body_location[0]]
    
            # get the angle to the shelter
            angle_to_shelter = self.angle_to_shelter()
    
            # get the distance to the object
            # distances_from_obstacle = self.distance_arena[self.end_position[1][eligible_paths].astype(int), self.end_position[0][eligible_paths].astype(int)]
            distances_from_obstacle = self.distance_arena[self.body_location[1], self.body_location[0]]
            far_from_obstacle = distances_from_obstacle > 75
    
            # get the deviation from the intended angle from the homing vector
            intended_minus_shelter = abs(path_directions - angle_to_shelter)
            intended_minus_shelter[intended_minus_shelter > 180] = abs(360 - intended_minus_shelter[intended_minus_shelter > 180])
    
            # get the deviation from the obstacle angle from the homing vector
            obstacle_minus_shelter = abs(current_angle_from_obstacle[0] - angle_to_shelter)
            if obstacle_minus_shelter > 180: obstacle_minus_shelter = abs(360 - obstacle_minus_shelter)
            if obstacle_minus_shelter < 10: obstacle_minus_shelter = 0
    
            # adjust the turn angle accordingly, if it helps getting to the shelter
            shift_sign = (np.sign(path_directions) != np.sign(current_angle_from_obstacle) )
            current_angle_from_obstacle[shift_sign] = \
                current_angle_from_obstacle[shift_sign] - 360 * np.sign(current_angle_from_obstacle[shift_sign])
            shift_direction = (obstacle_minus_shelter < intended_minus_shelter) * far_from_obstacle
            path_directions[shift_direction] = \
                (path_directions[shift_direction] * 3 + current_angle_from_obstacle[shift_direction] * 1) / 4
    
            # get their angles
            mu = (path_directions + 180) * np.pi / 180
    
            # compute the von mises features (normalized so range is 0 to 1)
            degree_features = np.tile(self.degrees, (1, len(mu)))
            vm_features = np.exp(self.kappa/2 * np.cos(degree_features - mu)) / np.exp(self.kappa/2) * self.predicted_weights[eligible_paths]
            predicted_angle = np.sum(vm_features, 1)
    
            # use the predicted weights to add them together
            predicted_angle = predicted_angle / np.sum(predicted_angle)
        else:
            predicted_angle = np.zeros(len(self.degrees))

        self.ax.fill_between(self.degrees[:,0]*180/np.pi, predicted_angle , color='royalblue', linewidth=4, alpha = .5) #lightcoral [.94, .71, .25]

        # get the likelihood and max L intended angle
        self.likelihood = predicted_angle[self.current_path_idx]

        self.intended_angle = self.degrees[np.argmax(predicted_angle)] * 180 / np.pi - 180

        




















    def get_previous_homing_utility(self):
        '''         DETERMINE THE PATH TO FOLLOW BASED ON THE ACTION UTILITY TABLE             '''

        # extract start-point indices
        self.starting_indices = np.where(self.start_idx[:self.stim_frame])[0]

        # get the position and HD for each one
        self.start_position = np.zeros((2, len(self.starting_indices)))
        self.start_position[0] = self.coordinates['head_location'][0][self.starting_indices]
        self.start_position[1] = self.coordinates['head_location'][1][self.starting_indices]

        # get the final position of each path
        repetition_end_indices = self.start_idx[self.starting_indices].astype(int)
        repetition_end_points = self.coordinates['head_location'][0][repetition_end_indices], self.coordinates['head_location'][1][repetition_end_indices]

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

        # get the angle to turn by
        self.get_turn_angle()


    def determine_target_angle(self, type = 'end point'):
        '''         DETERMINE THE ANGLE TO TURN BASED ON THE ACTION ENDPOINT TABLE             '''

        # sort start indices by overall offset
        target_point = np.flip(np.unravel_index(int(cp.argmax(self.heat_map)), self.heat_map.shape))

        # get the distance to the end point
        distance_to_subgoal = np.sqrt((target_point[0] - self.body_location[0]) ** 2 + (target_point[1] - self.body_location[1]) ** 2)

        # get the vector direction and distance to travel
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
        self.rep_arena = self.color_arena.copy()

        # draw an arrowed line
        if point1 == point2:
            cv2.circle(self.rep_arena, tuple(int(x) for x in point1), 40, (255, 150, 150), 5)
        else:
            cv2.arrowedLine(self.rep_arena, tuple(int(x) for x in point1), tuple(int(x) for x in point2), (255, 150, 150), thickness=3, tipLength=.1)

        # show it
        cv2.imshow('coordinate', self.rep_arena)

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
        if self.frame_num < self.TS_starting_indices[0]:
            speed_color_light, speed_color_dark = (.9, .9, .9), (.3, .3, .3)
        else:
            speed_color_light, speed_color_dark = set_up_speed_colors(self.current_speed, simulation=True)

        # add dark mouse if applicable
        if (np.sum(self.large_mouse_mask * self.model_mouse_mask_previous) == 0) or not self.frozen_weights:
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

        # initialize arena, pre-stim
        if not self.frozen_weights or self.frame_num < self.stim_frame: self.color_arena = self.color_arena_initialize.copy()

        # extract DLC coordinates from the saved coordinates dictionary]
        body_angle = self.coordinates['body_angle'][self.frame_num - 1]
        shoulder_angle = self.coordinates['shoulder_angle'][self.frame_num - 1]
        head_angle = self.coordinates['head_angle'][self.frame_num - 1]
        neck_angle = self.coordinates['neck_angle'][self.frame_num - 1]
        nack_angle = self.coordinates['nack_angle'][self.frame_num - 1]
        head_location = tuple(self.coordinates['head_location'][:, self.frame_num - 1].astype(np.uint16))
        nack_location = tuple(self.coordinates['nack_location'][:, self.frame_num - 1].astype(np.uint16))
        front_location = tuple(self.coordinates['front_location'][:, self.frame_num - 1].astype(np.uint16))
        shoulder_location = tuple(self.coordinates['shoulder_location'][:, self.frame_num - 1].astype(np.uint16))
        body_location = tuple(self.coordinates['center_body_location'][:, self.frame_num - 1].astype(np.uint16))
        self.current_speed = self.coordinates['speed'][self.frame_num - 1]

        # make a single large ellipse used to determine when do use the flight_color_dark
        self.body_location = body_location
        if not self.frozen_weights: self.head_location = head_location
        self.large_mouse_mask = cv2.ellipse(self.arena*0, body_location, (int(self.large_body_length), int(self.large_body_length * 3 / 5)), 180 - self.body_angle,0, 360, 100, thickness=-1)

        # set scale for size of model mouse
        self.body_length = 16

        # when turning, adjust relative sizes
        if abs(body_angle - shoulder_angle) > 35:
            shoulder = False
        else:
            shoulder = True

        # draw ellipses representing model mouse
        model_mouse_mask = cv2.ellipse(self.arena*0, body_location, (int(self.body_length * .9), int(self.body_length * .5)), 180 - body_angle,0, 360, 100, thickness=-1)
        model_mouse_mask = cv2.ellipse(model_mouse_mask, nack_location, (int(self.body_length * .7), int(self.body_length * .35)), 180 - nack_angle, 0, 360, 100,thickness=-1)
        model_mouse_mask = cv2.ellipse(model_mouse_mask, head_location, (int(self.body_length * .6), int(self.body_length * .3)), 180 - head_angle, 0, 360, 100,thickness=-1)
        if shoulder:
            model_mouse_mask = cv2.ellipse(model_mouse_mask, shoulder_location, (int(self.body_length), int(self.body_length * .4)), 180 - shoulder_angle, 0, 360,100, thickness=-1)
        self.model_mouse_mask = cv2.ellipse(model_mouse_mask, front_location, (int(self.body_length * .5), int(self.body_length * .33)), 180 - neck_angle, 0, 360, 100,thickness=-1)


    def get_stim_indices(self, stims_video):
        '''         GET INDICES WHEN A STIMULUS WAS JUST PLAYED         '''
        # initialize list
        self.stim_idx = np.array([])

        # add each stim time and the following 10 seconds to the list
        for stim_frame in stims_video:
            self.stim_idx = np.append(self.stim_idx, np.arange( stim_frame - 100, stim_frame + 400) )



    def get_current_stim_indices(self):
        '''         GET INDICES WHEN A STIMULUS WAS JUST PLAYED         '''

        # add each stim time and the following 10 seconds to the list
        self.current_stim_idx = np.arange(self.stim_frame - 100, self.stim_frame + 400)
