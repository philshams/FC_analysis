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
from tqdm import tqdm
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
    def __init__(self, coordinates, stim_frame, shelter_location, arena, obstacle_type,
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
        self.large_mouse_mask = arena * 0
        self.model_mouse_mask = arena * 0
        self.model_mouse_mask_previous = arena * 0
        self.coordinates = coordinates
        self.distance_arena = np.load('C:\\Drive\\DLC\\transforms\\distance_arena_' + obstacle_type + '.npy')
        self.angle_arena = np.load('C:\\Drive\\DLC\\transforms\\angle_arena_' + obstacle_type + '.npy')
        self.stim_frame = stim_frame
        self.start_idx = start_idx
        self.obstacle_in_the_way = 0
        self.get_stim_indices(stims_video)
        self.video_name = videoname
        self.save_folder = save_folder

        self.start_position = []
        self.starting_indices = []

        self.trial_type = trial_type

        self.previous_strategy = ' '
        self.innate_strategy = ' '
        self.subgoal_locations = [ [int(y[1]*self.arena.shape[0]/1000) for y in subgoal_location['sub-goals']],
                                     [int(x[0]*self.arena.shape[1]/1000) for x in subgoal_location['sub-goals']] ]
        self.distance_to_shelter = np.inf

        self.degrees = np.linspace(0, 2 * np.pi, 360)

        self.strategy_colors = {}
        self.strategy_colors['homing vector'] = (0, 0, 250) #(70, 70, 150)
        self.strategy_colors['obstacle guidance'] = (0, 0, 250) #(70, 70, 150)
        self.strategy_colors['repetition'] = (250, 0, 0)
        self.strategy_colors['spatial planning'] = (0, 250, 0)
        self.strategy_colors['exploration'] = (120, 120, 120)

        self.strategies = [strategy]

        # set parameters
        self.trial_duration = 300
        self.dist_to_whiskers = 50 #20
        self.whisker_radius = 150 #50
        self.body_length = 16
        self.large_body_length = 40
        self.max_shelter_proximity = 190
        self.arrived_at_shelter_proximity = 55

        # SR parameters
        self.bins_per_side = 20
        self.bin_size = int(self.arena.shape[0] / self.bins_per_side)
        self.spatial_blurring = 2
        self.landmark_bias = 100

        # commence simulation
        self.main()





    def main(self):
        '''        RUN THE SIMULATION        '''

        # initialize trial data
        self.initialize_trial()

        # loop across post-stimulus frames
        for self.bout in range(len(self.TS_starting_indices)):

            if self.bout == len(self.TS_starting_indices) - 1:
                self.last_bout = True
            else:
                self.last_bout = False

            # initialize bout data
            self.initialize_bout()

            # loop across strategies, getting the likelihood
            for i, self.strategy in enumerate(self.strategies):

                # execute strategy
                if self.strategy == 'homing vector':        self.homing_vector()
                elif self.strategy == 'repetition':         self.repetition()
                elif self.strategy == 'spatial planning':   self.spatial_planning()
                elif self.strategy == 'exploration':        self.not_escape()
                
                self.likelihoods.append(self.likelihood_across_angles[self.current_path_idx])

            # format the strategy likelihood plot
            self.format_plot()

            # initialize the movie data
            self.initialize_movie()

            self.background = np.zeros((760,720,3),np.uint8)

            # loop across frames to show
            for self.frame_num in range(self.start_frame, self.TS_end_indices[self.bout]):

                # make the mouse masks
                self.make_mouse_masks()

                # shade in the mouse
                self.shade_in_mouse()

                # show the internal representation
                if self.strategy == 'homing vector':        self.show_homing_vector_representation()
                if self.strategy == 'obstacle guidance':    self.show_obstacle_guidance_representation()
                if self.strategy == 'repetition':           self.show_repetition_representation()
                if self.strategy == 'spatial planning':     self.show_planning_representation()
                if self.strategy == 'exploration':          self.show_random_representation()

                # add text
                if self.strategy == 'obstacle guidance': text = 'Instinctive obstacle avoidance'
                else: text = self.strategy


                if self.frame_num == self.TS_starting_indices[self.bout] or \
                    self.frame_num == self.TS_end_indices[self.bout]-1:

                    rep_arena_flash = self.rep_arena.copy()

                    if self.frame_num == self.TS_end_indices[self.bout] - 1:
                        _, newcontours, _ = cv2.findContours(self.model_mouse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(rep_arena_flash, newcontours, 0, self.strategy_colors[self.strategy], thickness=2)
                        cv2.drawContours(rep_arena_flash, contours, 0, self.strategy_colors[self.strategy], thickness=2)
                    else:
                        _, contours, _ = cv2.findContours(self.model_mouse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(rep_arena_flash, contours, 0, self.strategy_colors[self.strategy], thickness=2)



                    if self.frame_num == self.TS_starting_indices[self.bout]:
                        flash_on = np.tile(np.concatenate((np.zeros(10), np.ones(10), )), 3).astype(int)
                    else: flash_on = np.ones(30)

                    # pause here
                    for flash in flash_on:
                        # fade out the old trajectory

                        # but keep the current position / representation

                        if flash: rep_arena_show = rep_arena_flash
                        else: rep_arena_show = self.rep_arena

                        # show the previous frame
                        self.background[40:, :, :] = rep_arena_show
                        cv2.putText(self.background, text, (20, 40), 0, 1.25, self.strategy_colors[self.strategy], thickness=2)
                        cv2.imshow('strategy cam', self.background)
                        self.vid.write(self.background)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                # show the previous frame
                self.background[40:,:,:] = self.rep_arena
                cv2.putText(self.background, text, (20, 40), 0, 1.25, self.strategy_colors[self.strategy], thickness=2)
                cv2.imshow('strategy cam', self.background)
                self.vid.write(self.background)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # draw a line to show the linear piece escape
            dotted_line_arena = self.rep_arena.copy()
            cv2.line(dotted_line_arena, (int(self.TS_start_position[0][self.bout]), int(self.TS_start_position[1][self.bout])),
                     (int(self.TS_end_position[0][self.bout]), int(self.TS_end_position[1][self.bout])), (5, 5, 5), 6)
            cv2.line(dotted_line_arena, (int(self.TS_start_position[0][self.bout]), int(self.TS_start_position[1][self.bout])),
                     (int(self.TS_end_position[0][self.bout]), int(self.TS_end_position[1][self.bout])), self.strategy_colors[self.strategy], 2)
            alpha = .5
            cv2.addWeighted(self.rep_arena, alpha, dotted_line_arena, 1 - alpha, 0, dotted_line_arena)
            imageio.imwrite(os.path.join(self.new_save_folder, self.video_name + '_' + str(self.bout+1) + '.tif'), cv2.cvtColor(dotted_line_arena, cv2.COLOR_BGR2RGB))

        self.vid.release()




    def homing_vector(self):
        '''        SIMULATE THE homing vector STRATEGY        '''

        self.spatial_blurring_x = 0
        self.spatial_blurring_y = 0
        self.gaussian_blur = 19 #18

        # check if there is an obstacle in the way and adjust accordingly
        self.obstacle_guidance()

        # if not, find the homing vector to the shelter
        if not self.obstacle_in_the_way:

            # note that there was NO OBSTACLE
            self.innate_strategy = 'homing vector'

            # get the transition matrix for random exploration
            self.transition = self.transition_HV  #self.get_random_transition()

            # get the resulting functional successor representation
            self.time_discount = .99 # .99 ~ 1 sec is .74 // 5 sec is .2 // 10 sec is .05
            self.get_successor_arena()

            # go from SR to likelihood across angles
            self.get_likelihood()

            # plot the HOMING VECTOR likelihood
            self.ax.fill_between(self.degrees*180/np.pi, self.likelihood_across_angles, color='darkred', linewidth=4, alpha = .55)

        # store this representation
        self.innate_representation = self.color_successor.copy()

        # get the max likelihood response
        self.innate_response = self.degrees[np.argmax(self.likelihood_across_angles)] * 180 / np.pi



    def repetition(self):
        '''        SIMULATE THE repetition STRATEGY        '''

        self.spatial_blurring_x = np.max((10 - self.y_bin_body, 2)) #SUPER TEMPORARY
        self.spatial_blurring_y = 2
        self.gaussian_blur = 25 # 20

        # get the transition matrix for random exploration
        self.get_repetition_transition()

        # get the resulting functional successor representation
        self.time_discount = .995  # .99 ~ 1 sec is .74 // 5 sec is .2 // 10 sec is .05
        self.get_successor_arena()
        self.repetition_representation = self.color_successor.copy()

        # go from SR to likelihood across angles
        self.get_likelihood()

        # plot the REPETITION likelihood
        self.ax.fill_between(self.degrees * 180 / np.pi, self.likelihood_across_angles, color='royalblue', linewidth=4, alpha = .5)




    def spatial_planning(self):
        '''        SIMULATE THE spatial planning STRATEGY        '''

        self.spatial_blurring_x = 1
        self.spatial_blurring_y = 1
        self.gaussian_blur = 19 # 18

        # get the transition matrix for random exploration
        angle_from_obstacle = self.angle_arena[int(self.TS_start_position[1][self.bout]), int(self.TS_start_position[0][self.bout])]
        if angle_from_obstacle == -90 and self.trial_type:
            self.transition = self.transition_planning
            self.planning_strategy = 'geodesic'
        else:
            self.transition = self.transition_HV
            self.planning_strategy = 'homing vector'

        # get the resulting functional successor representation
        self.time_discount = .95  # .99 ~ 1 sec is .74 // 5 sec is .2 // 10 sec is .05
        self.get_successor_arena()

        # store this representation
        self.planning_representation = self.color_successor.copy()

        # go from SR to likelihood across angles
        self.get_likelihood()

        # (plot differently if it's the same as the HOMING VECTOR)
        self.intended_angle = self.degrees[np.argmax(self.likelihood_across_angles)] * 180 / np.pi
        if not int(self.intended_angle - self.innate_response): self.same_as_HV = 1
        else: self.same_as_HV = 0

        #TEMPORARY
        if self.last_bout:
            self.same_as_HV = 1

        # plot the SPATIAL PLANNING likelihood
        self.ax.fill_between(self.degrees * 180 / np.pi, self.likelihood_across_angles, color='mediumspringgreen', linewidth=4 - 3*self.same_as_HV, alpha=.3-.15*self.same_as_HV)


    def not_escape(self):
        '''        SIMULATE THE not escape STRATEGY        '''

        self.spatial_blurring_x = 2
        self.spatial_blurring_y = 2
        self.gaussian_blur = 25

        # get the resulting functional successor representation
        self.get_successor_arena()
        self.not_escape_representation = self.color_successor.copy()

        # go from SR to likelihood across angles
        self.get_likelihood()
        self.likelihood_across_angles[self.current_path_idx] = 0 #TEMPORARY FOR TIAGO

        # plot the REPETITION likelihood
        self.ax.fill_between(self.degrees * 180 / np.pi, self.likelihood_across_angles, color='grey', linewidth=2, linestyle='--', alpha=.2)


        




    def get_likelihood(self):

        # get the angle output likelihood
        origin = (int(self.bin_size * (self.x_bin_body + .5)), int(self.bin_size * (self.y_bin_body + .5)))

        # initialize likelihood list
        likelihood_across_angles = np.zeros_like(self.degrees)

        # initialize arena copy
        successor_copy = self.arena.copy()

        # loop across angles
        for i, degree in enumerate(self.degrees):

            # get the end point for the line, for this angle
            end_point = (int(origin[0] - self.arena.shape[1] * np.cos(degree)), int(origin[1] + self.arena.shape[0] * np.sin(degree)))

            # make a mask corresponding to this angle
            angle_mask = np.zeros_like(self.arena)
            cv2.line(angle_mask, origin, end_point, 1, 1)

            # get indices
            angle_indices = np.where(angle_mask)
            if degree > np.pi: angle_indices = tuple(np.flip(angle_indices, axis=1))

            # just use the second half of the vector
            if (self.strategy == 'homing vector' and self.innate_strategy == 'homing vector') or (self.planning_strategy == 'homing vector'):
                circle_mask = np.zeros_like(self.arena)
                distance_to_shelter = self.distance_from_shelter_euclid[int(self.TS_start_position[1][self.bout]), int(self.TS_start_position[0][self.bout])]
                cv2.circle(circle_mask, origin, int(distance_to_shelter / 2), 1, thickness=2)

                mask_circle_overlap = np.where(angle_mask * circle_mask)
                if mask_circle_overlap[0].size:
                    intersection_point = np.min(mask_circle_overlap[0])
                    if degree < np.pi: angle_indices = angle_indices[0][angle_indices[0] > intersection_point], angle_indices[1][angle_indices[0] > intersection_point]
                    else: angle_indices = angle_indices[0][angle_indices[0] < intersection_point], angle_indices[1][angle_indices[0] < intersection_point]
                else: angle_indices = []

            # remove those going past the wall (unless HOMING VECTOR)
            else:
                mask_obstacle_overlap = np.where(angle_mask * self.mask)
                if mask_obstacle_overlap[0].size:
                    intersection_point = np.min(mask_obstacle_overlap[0]) + self.landmark_bias #* np.sin(degree)
                    angle_indices = angle_indices[0][angle_indices[0] < intersection_point], angle_indices[1][angle_indices[0] < intersection_point]

            # fill in the likelihood for the current angle
            likelihood_across_angles[i] = np.sum(self.successor_arena[angle_indices])

            # show which values are used
            successor_copy[angle_indices] = 0

        # normalize the likelihood
        self.likelihood_across_angles = likelihood_across_angles / np.sum(likelihood_across_angles)

        # show the values use
        # cv2.imshow('values used', successor_copy)




    def get_repetition_transition(self):

        # extract start-point and end-point frame number
        self.start_indices = np.where(self.coordinates['start_index'])[0]
        self.end_indices = self.coordinates['start_index'][self.start_indices].astype(int)

        # get all indices in which there's a homing (or not)
        homing_idx = np.ones(len(self.coordinates['center_body_location'][0]), dtype = bool)
        homing_idx[self.stim_frame:] = False

        # bin the position data
        bin_array = np.ones(len(self.coordinates['front_location'][0][homing_idx]))*np.nan
        bin_ID = 0
        bin_sum = 0
        if self.coordinates['front_location'][ self.coordinates['front_location'] >= self.arena.shape[0] ].size:
            self.coordinates['front_location'][ self.coordinates['front_location'] >= self.arena.shape[0] ] = self.arena.shape[0] - 1

        for x_bin in range(self.bins_per_side):
            in_x_bin = (self.coordinates['front_location'][0][homing_idx] >= (self.bin_size*x_bin) ) * (self.coordinates['front_location'][0][homing_idx] < (self.bin_size*(x_bin+1)) )

            for y_bin in range(self.bins_per_side):
                in_y_bin = (self.coordinates['front_location'][1][homing_idx] >= (self.bin_size * y_bin)) * (self.coordinates['front_location'][1][homing_idx] < (self.bin_size * (y_bin + 1)))

                bin_array[in_x_bin * in_y_bin] = bin_ID
                bin_sum += np.sum(bin_array)
                bin_ID += 1

        bin_array = bin_array.astype(int)

        # Get counts of (bin_preceding --> bin_following) to (i,j) of bin_counts
        bin_counts = np.zeros((self.bins_per_side**2, self.bins_per_side**2), dtype=int)
        np.add.at(bin_counts, (bin_array[:-1], bin_array[1:]), 1)

        # Get transition probs (% in column 1 going to row 1,2,3,etc.)
        self.transition = (bin_counts.T / np.sum(bin_counts, axis=1)).T
        np.nan_to_num(self.transition, copy=False)



    def get_successor_arena(self):

        # get the SR
        successor_representation = np.linalg.inv(np.identity(self.bins_per_side ** 2) - self.time_discount * self.transition)

        # define shelter location
        shelter_indices = [(x, y) for x in range(8, 12) for y in range(15, 20)]
        shelter_bin_idx = [(SI[0] * self.bins_per_side) + SI[1] for SI in shelter_indices]

        # initialize SR arenaq
        successor_arena = np.zeros(self.arena.shape)

        # get obstacle angle
        obstacle_angle_body = self.angle_arena[int(self.bin_size * (self.y_bin_body + .5)), int(self.bin_size * (self.x_bin_body + .5))]

        # also use nearby points
        for x_shift in tqdm(range(-self.spatial_blurring_x, self.spatial_blurring_x + 1 )): #TEMPORARY
            for y_shift in range(-self.spatial_blurring_y, self.spatial_blurring_y + 1 ):

                # initialize SR arenaq
                curr_successor_arena = np.zeros(self.arena.shape)

                # get the SR for the clicked bin
                start_bin_idx = ((self.x_bin_body + x_shift) * self.bins_per_side) + (self.y_bin_body + y_shift)

                # get the geodesic distance from the clicked bin
                try:
                    obstacle_angle_shift = self.angle_arena[int(self.bin_size * (self.y_bin_body + y_shift + .5)), int(self.bin_size * (self.x_bin_body + x_shift + .5))]
                    if np.isnan(obstacle_angle_shift): continue
                except:
                    continue
                distance_body_euclid = self.distance_from_shelter_euclid[int(self.bin_size * (self.y_bin_body + .5)), int(self.bin_size * (self.x_bin_body + .5))]
                distance_body = self.distance_from_shelter[int(self.bin_size * (self.y_bin_body + .5)), int(self.bin_size * (self.x_bin_body + .5))]

                # get the obstacle angle offset
                obstacle_angle_offset = abs(obstacle_angle_shift - obstacle_angle_body)
                if obstacle_angle_offset > 180: obstacle_angle_offset = abs(360 - obstacle_angle_offset)
                if obstacle_angle_offset > 90:
                    continue

                # fill in each bin with the successor-ness
                for x_bin in range(self.bins_per_side):
                    for y_bin in range(self.bins_per_side):
                        # get the bin index
                        end_bin_idx = (x_bin * self.bins_per_side) + y_bin

                        if end_bin_idx != start_bin_idx:
                            # get the decrease in geodesic distance and euclidean distance
                            distance_bin = self.distance_from_shelter[int(self.bin_size * (y_bin + .5)), int(self.bin_size * (x_bin + .5))]
                            distance_bin_euclid = self.distance_from_shelter_euclid[int(self.bin_size * (y_bin + .5)), int(self.bin_size * (x_bin + .5))]
                            distance_change = np.max(((distance_body - distance_bin), (distance_body_euclid - distance_bin_euclid)))

                            if np.mean(successor_representation[end_bin_idx, shelter_bin_idx]) > np.mean(successor_representation[start_bin_idx, shelter_bin_idx]) \
                                    and distance_change > 60 and not self.strategy == 'exploration' \
                                    and y_bin > 8 and (y_bin < 11 or self.y_bin_body > 6) \
                                    and (y_bin > 10 or self.y_bin_body < 8) \
                                    and (y_bin > 15 or not self.last_bout) and (abs(x_bin - 9.5)<2 or not self.last_bout): #>= 9 TEMPORARY
                                curr_successor_arena[int(self.bin_size * (y_bin + .5)), int(self.bin_size * (x_bin + .5))] = \
                                    np.mean(successor_representation[end_bin_idx, shelter_bin_idx]) * successor_representation[start_bin_idx, end_bin_idx]
                            elif self.strategy == 'exploration':
                                curr_successor_arena[int(self.bin_size * (y_bin + .5)), int(self.bin_size * (x_bin + .5))] = \
                                    successor_representation[start_bin_idx, end_bin_idx]

                successor_arena = successor_arena + (curr_successor_arena)# / np.max((1, 4 * np.sqrt(x_shift ** 2 + y_shift ** 2))))

        # gaussian blur
        self.successor_arena = cv2.GaussianBlur(successor_arena, ksize=(181, 181), sigmaX=self.gaussian_blur, sigmaY=self.gaussian_blur) # = 20
        # if (self.innate_strategy == 'obstacle guidance' and self.strategy == 'homing vector'):
        #     self.successor_arena[int(self.arena.shape[0]/2):, :] = 0

        successor_arena = (self.successor_arena / np.max(self.successor_arena) * 255)  # / np.max(successor_arena) #.003
        successor_arena[successor_arena > 255] = 255
        successor_arena[self.arena == 0] = 0
        successor_arena = successor_arena.astype(np.uint8)

        if self.strategy == 'exploration': self.color_successor = cv2.cvtColor(cv2.resize(successor_arena, self.arena.shape), cv2.COLOR_GRAY2BGR)
        else:
            self.color_successor = cv2.applyColorMap(cv2.resize(successor_arena, self.arena.shape), cv2.COLORMAP_OCEAN)
            if self.strategy == 'homing vector': self.color_successor[:,:,[0,1,2]] = self.color_successor[:,:,[2,1,0]]
            elif self.strategy == 'spatial planning': self.color_successor[:,:,[0,1,2]] = self.color_successor[:,:,[1,0,2]]

            # self.color_successor[(np.mean(self.color_successor,2)==0) * ~(self.arena==0)] = 120

        # merge the arena and the SR
        alpha = .1
        cv2.addWeighted(self.color_arena, alpha, self.color_successor, 1 - alpha, 0, self.color_successor)
        cv2.imshow('SR', self.color_successor)
        cv2.waitKey(1)



    def format_plot(self):

        self.ax.set_title('Likelihood of this Orientation Movement for Each Strategy');
        self.ax.set_xlabel('allocentric head direction (degrees)');
        self.ax.set_ylabel('likelihood of HD given strategy')
        self.plot_height = .03
        # self.ax.set_xlim([0, 180]);
        self.ax.set_ylim([0, self.plot_height])
        self.ax.plot([self.degrees[self.current_path_idx] * 180 / np.pi, self.degrees[self.current_path_idx] * 180 / np.pi], [0, self.plot_height], color='white',
                     linestyle='--')
        leg = self.ax.legend(("escape path", self.innate_strategy, 'spatial planning', 'path learning', 'exploration'))
        leg.draggable()
        plt.savefig(os.path.join(self.new_save_folder, self.video_name + '_like' + str(self.bout + 1) + '.tif'))
        # plt.show()



    def initialize_trial(self):
        '''        GENERATE MASK CORRESPONDING TO THE MOUSE'S POSITION        '''
        # extract the location and angle from the coordinates
        # self.body_location = self.coordinates['center_body_location'][:, self.stim_frame].astype(np.uint16)
        # self.body_angle = self.coordinates['body_angle'][self.stim_frame]
        self.current_speed = 0

        self.distance_to_shelter = np.inf
        # self.feature_values = np.load('C:\\Drive\\DLC\\transforms\\feature_values_' + self.obstacle_type + '.npy')
        self.feature_values = np.load('C:\\Drive\\DLC\\transforms\\feature_values_wall.npy')
        # self.LR = joblib.load('C:\\Drive\\DLC\\transforms\\regression_' + self.obstacle_type)
        self.LR = joblib.load('C:\\Drive\\DLC\\transforms\\regression_wall')

        # np.save('C:\\Drive\\DLC\\transforms\\transition_HV_wall.npy', self.transition)
        # np.save('C:\\Drive\\DLC\\transforms\\transition_planning_wall.npy', self.transition)
        # np.save('C:\\Drive\\DLC\\transforms\\transition_planning_left_block_wall.npy', self.transition)
        # np.save('C:\\Drive\\DLC\\transforms\\transition_planning_right_block_wall.npy', self.transition)

        self.transition_HV = np.load('C:\\Drive\\DLC\\transforms\\transition_HV_wall.npy')
        self.transition_planning = np.load('C:\\Drive\\DLC\\transforms\\transition_planning_wall.npy')
        self.transition_planning_left_block = np.load('C:\\Drive\\DLC\\transforms\\transition_planning_left_block_wall.npy')
        self.transition_planning_right_block = np.load('C:\\Drive\\DLC\\transforms\\transition_planning_right_block_wall.npy')


        self.initialize_model()

        self.get_trial_info()

        # initialize mouse position and arena array
        self.new_save_folder = os.path.join(self.save_folder + '_simulate')
        if not os.path.isdir(self.new_save_folder):
            os.makedirs(self.new_save_folder)

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.vid = cv2.VideoWriter(os.path.join(self.new_save_folder, self.video_name + '_classifier.mp4'), fourcc, 30, (self.width, self.height+40), True)
        plt.style.use('dark_background');
        plt.rcParams.update({'font.size': 22})


    def initialize_movie(self):

        # select the winning strategy
        self.strategies[0] = self.innate_strategy
        self.strategy = self.strategies[np.argmax(self.likelihoods)]
        # if self.strategy == 'spatial planning' and self.same_as_HV:
        #     self.strategy = 'homing vector'

        #temporary
        if self.last_bout: #and (self.strategy == 'spatial planning' or self.strategy == 'obstacle guidance')
            self.strategy = 'homing vector'


        # display the mouse running around with this representation in mind
        if not self.bout:
            self.start_frame = self.TS_starting_indices[0] - 30
        else:
            self.start_frame = self.TS_starting_indices[self.bout] - 10 * (self.previous_strategy != self.strategy)

        self.previous_strategy = self.strategy
        self.calculating = False
















    def initialize_model(self):
        '''         GERNERATE THE GEODESIC MAP USED FOR THE INTERNAL MODEL      '''

        # initialize the map
        phi = np.ones_like(self.arena)

        # mask the map wherever there's an obstacle
        self.mask = (self.arena == 90)
        self.phi_masked = np.ma.MaskedArray(phi, self.mask)

        # get the geodesic map of distance from the shelter
        phi_from_shelter = self.phi_masked.copy()
        phi_from_shelter[self.shelter_location[1], self.shelter_location[0]] = 0
        self.distance_from_shelter = np.array(skfmm.distance(phi_from_shelter))

        # get the euclidean map of distance from the shelter
        phi[self.shelter_location[1], self.shelter_location[0]] = 0
        self.distance_from_shelter_euclid = np.array(skfmm.distance(phi))

        # calculate the gradient along this map
        self.geodesic_gradient = np.gradient(self.distance_from_shelter)
        self.geodesic_gradient[0][abs(self.geodesic_gradient[0]) > 1.1] = 0
        self.geodesic_gradient[1][abs(self.geodesic_gradient[1]) > 1.1] = 0

        # get the geodesic map of distance from shelter, given one option blocked
        if np.sum(self.mask):
            # generate a new, hypothetical mask RIGHT
            self.mask_right = self.mask.copy()
            self.mask_right[np.min(np.where(self.mask)[0]), np.max(np.where(self.mask)[1]):] = 1
            phi_masked_right = np.ma.MaskedArray(phi, self.mask_right)

            # get the geodesic map of distance from the shelter
            phi_masked_right[self.shelter_location[1], self.shelter_location[0]] = 0
            self.distance_from_shelter_right = np.array(skfmm.distance(phi_masked_right))

            # calculate the gradient along this map
            self.geodesic_gradient_right = np.gradient(self.distance_from_shelter_right)
            self.geodesic_gradient_right[0][abs(self.geodesic_gradient_right[0]) > 1.1] = 0
            self.geodesic_gradient_right[1][abs(self.geodesic_gradient_right[1]) > 1.1] = 0

            # generate a new, hypothetical mask LEFT
            self.mask_left = self.mask.copy()
            self.mask_left[np.min(np.where(self.mask)[0]), :np.min(np.where(self.mask)[1])] = 1
            phi_masked_left = np.ma.MaskedArray(phi, self.mask_left)

            # get the geodesic map of distance from the shelter
            phi_masked_left[self.shelter_location[1], self.shelter_location[0]] = 0
            self.distance_from_shelter_left = np.array(skfmm.distance(phi_masked_left))

            # calculate the gradient along this map
            self.geodesic_gradient_left = np.gradient(self.distance_from_shelter_left)
            self.geodesic_gradient_left[0][abs(self.geodesic_gradient_left[0]) > 1.1] = 0
            self.geodesic_gradient_left[1][abs(self.geodesic_gradient_left[1]) > 1.1] = 0
        else:
            self.distance_from_shelter_left, self.distance_from_shelter_right = False, False

    def show_homing_vector_representation(self):
        '''         DISPLAY THE HOMING VECTOR           '''

        # initialize arrays
        self.rep_arena = self.color_arena.copy()

        # shade into rep map
        alpha = .4
        cv2.addWeighted(self.rep_arena, alpha, self.innate_representation, 1 - alpha, 0, self.rep_arena)

        # shade into rep map
        # shade_in_zone = ((self.color_arena == 255)).astype(bool)
        # self.rep_arena[shade_in_zone] = self.innate_representation[shade_in_zone]



    def show_obstacle_guidance_representation(self):
        '''         DISPLAY THE OBSTACLE                '''

        # initialize arrays
        self.rep_arena = self.color_arena.copy()

        # shade into rep map
        alpha = .4
        cv2.addWeighted(self.rep_arena, alpha, self.innate_representation, 1 - alpha, 0, self.rep_arena)

        # shade into rep map
        # shade_in_zone = ((self.color_arena == 255)).astype(bool)
        # self.rep_arena[shade_in_zone] = self.innate_representation[shade_in_zone]

        #
        # # initialize arrays
        # obstacle_arena = self.arena*0
        # self.rep_arena = self.color_arena.copy()
        #
        # # shade in obstacle
        # self.rep_arena[self.arena==90] = 255*np.array([.1,.1,1])


    def show_repetition_representation(self):
        '''         DISPLAY THE PRIOR TARGETS           '''

        # initialize arrays
        self.rep_arena = self.color_arena.copy()

        # shade into rep map
        alpha = .4
        cv2.addWeighted(self.rep_arena, alpha, self.repetition_representation, 1 - alpha, 0, self.rep_arena)

        # shade into rep map
        # shade_in_zone = ((self.color_arena == 255)).astype(bool)
        # self.rep_arena[shade_in_zone] = self.repetition_representation[shade_in_zone]


    def show_planning_representation(self):
        '''         DISPLAY THE SPATIAL PLANNING        '''

        # initialize arrays
        self.rep_arena = self.color_arena.copy()

        # shade into rep map
        alpha = .4
        cv2.addWeighted(self.rep_arena, alpha, self.planning_representation, 1 - alpha, 0, self.rep_arena)

        # shade_in_zone = ((self.color_arena == 255)).astype(bool)
        # self.rep_arena[shade_in_zone] = self.planning_representation[shade_in_zone]



    def show_random_representation(self):
        '''         DISPLAY THE SPATIAL PLANNING        '''

        # initialize arrays
        self.rep_arena = self.color_arena.copy()

        # shade into rep map
        alpha = .4
        cv2.addWeighted(self.rep_arena, alpha, self.not_escape_representation, 1 - alpha, 0, self.rep_arena)

        # shade into rep map
        # shade_in_zone = ((self.color_arena == 255)).astype(bool)
        # self.rep_arena[shade_in_zone] = self.not_escape_representation[shade_in_zone]



    def get_trial_info(self):
        '''         GET THE DEETS FOR THE CURRENT TRIAL         '''

        # get the starting and ending indices
        self.get_current_stim_indices()
        this_stimulus_idx = np.array([(i in self.current_stim_idx) for i in np.where(self.start_idx)[0]])
        self.TS_starting_indices = np.where(self.start_idx)[0][this_stimulus_idx]
        self.TS_end_indices = self.start_idx[self.TS_starting_indices].astype(int)

        # get the start and end position for each one
        self.TS_start_position = self.coordinates['head_location'][0][self.TS_starting_indices], self.coordinates['head_location'][1][self.TS_starting_indices]
        self.TS_end_position = self.coordinates['head_location'][0][self.TS_end_indices], self.coordinates['head_location'][1][self.TS_end_indices]


        # get the vector direction of the previous path
        self.TS_path_direction = np.angle((self.TS_end_position[0] - self.TS_start_position[0]) + (-self.TS_end_position[1] + self.TS_start_position[1]) * 1j, deg=True)
        self.TS_head_direction = self.coordinates['body_angle'][self.TS_starting_indices + 10] #TEMPORARY

    def initialize_bout(self):
        '''         SET THE CURRENT POSITION AND ANGLE AS THAT OF THE BEGINNING OF THE CURRENT BOUT         '''

        # self.body_location = int(self.TS_start_position[0][self.bout]), int(self.TS_start_position[1][self.bout])
        # self.body_angle = self.TS_head_direction[self.bout]

        self.current_path_idx = int((self.TS_path_direction[self.bout] + 180) * len(self.degrees) / 360)

        self.x_bin_body = int(self.TS_start_position[0][self.bout] / self.bin_size)
        self.y_bin_body = int(self.TS_start_position[1][self.bout] / self.bin_size)

        self.strategies = ['homing vector', 'spatial planning', 'repetition', 'exploration']  # 'vector_repetition',

        self.likelihoods = []
        self.calculating = True

        self.color_arena_initialize = (
                (1 * cv2.cvtColor(self.arena.copy(), cv2.COLOR_GRAY2RGB).astype(float) + 3 * self.color_arena.astype(float)) / 4).astype(np.uint8)

        plt.figure(figsize=(18, 8))
        self.ax = plt.subplot(1, 1, 1)





    def obstacle_guidance(self):
        '''     SEE IF THERE IS AN OBSTACLE IN THE WAY        '''

        # first, make a circle to indicate whisker sensing
        whisking_center = (int(self.TS_start_position[0][self.bout] + self.dist_to_whiskers * np.cos(np.radians(self.body_angle))),
                           int(self.TS_start_position[1][self.bout] - self.dist_to_whiskers * np.sin(np.radians(self.body_angle))))
        whisking_circle = cv2.ellipse(self.arena * 0, whisking_center, (self.whisker_radius, self.whisker_radius), 180 - self.body_angle, 90, 270, 1, -1)

        # check if it is touching an obstacle
        obstacle_contact = whisking_circle * (self.arena < 255) * (self.arena > 0)
        current_obstacle_in_the_way = np.sum(obstacle_contact).astype(float)

        # get the increase in obstacle touching for decceleration
        self.obstacle_increase = current_obstacle_in_the_way - self.obstacle_in_the_way

        # get the direction to the obstacle -- if it's above, then dont do obstacle guidance?
        current_angle_from_obstacle = self.angle_arena[int(self.TS_start_position[1][self.bout]), int(self.TS_start_position[0][self.bout])]

        # is the obstacle in the way?
        self.obstacle_in_the_way = current_obstacle_in_the_way * (current_angle_from_obstacle != 90)

        # also get the distance to the shelter
        self.distance_to_shelter = np.sqrt(
            (self.TS_start_position[0][self.bout] - self.shelter_location[0]) ** 2 + (self.TS_start_position[1][self.bout] - self.shelter_location[1]) ** 2)

        # if it is touching an obstacle, see whether there is more obstacle sensed on the left vs right, front vs back
        if self.obstacle_in_the_way:

            # set the innate strategy
            self.innate_strategy = 'obstacle guidance'

            # get the transition matrix for random exploration
            self.transition = self.transition_planning  # self.get_geodesic_transition()

            # loop over left and right options
            for i, self.transition in enumerate([self.transition_planning_left_block, self.transition_planning_right_block]):

                # get the resulting functional successor representation
                self.get_successor_arena()

                # put them together in a weighted sum
                if i == 0:
                    successor_arena_1 = self.successor_arena.copy()
                    color_successor_1 = self.color_successor.copy()

            # angle from obstacle
            current_angle_from_obstacle = self.angle_arena[int(self.TS_start_position[1][self.bout]), int(self.TS_start_position[0][self.bout])]
            collision_angle = self.TS_head_direction[self.bout] - current_angle_from_obstacle
            if collision_angle > 180: collision_angle = 360 - collision_angle
            if collision_angle < -180: collision_angle = 360 + collision_angle

            # for now just do weights corresponding to angle of indicence with arbitrary exp function
            if collision_angle > 0:
                obstacle_weights = np.array([np.exp(-abs(collision_angle)/3), 1])
            else:
                obstacle_weights = np.array([1, np.exp(-abs(collision_angle) / 3)])
            obstacle_weights = obstacle_weights / np.sum(obstacle_weights)

            cv2.addWeighted(self.successor_arena, obstacle_weights[0], successor_arena_1, obstacle_weights[1], 0, self.successor_arena)
            cv2.addWeighted(self.color_successor, obstacle_weights[0], color_successor_1, obstacle_weights[1], 0, self.color_successor)
            cv2.imshow('SR', self.color_successor)


            # get the resulting functional successor representation
            # self.get_successor_arena()
            self.innate_representation = self.color_successor

            # go from SR to likelihood across angles
            self.get_likelihood()

            # plot the SPATIAL PLANNING likelihood
            self.ax.fill_between(self.degrees * 180 / np.pi, self.likelihood_across_angles, color='darkred', linewidth=4, alpha = .6)




    def shade_in_mouse(self):
        '''        SHADE IN THE MOUSE ON THE COLOR ARENA PLOT        '''

        speed_color_light, speed_color_dark = set_up_speed_colors(self.current_speed, simulation=True)

        # add dark mouse if applicable
        if (np.sum(self.large_mouse_mask * self.model_mouse_mask_previous) == 0) or self.frame_num < self.TS_starting_indices[0]:
            self.color_arena[self.model_mouse_mask.astype(bool)] = self.color_arena[self.model_mouse_mask.astype(bool)] * speed_color_dark**1.4
            self.model_mouse_mask_previous = self.model_mouse_mask
        # otherwise, shade in the current mouse position
        else:
            self.color_arena[self.model_mouse_mask.astype(bool)] = self.color_arena[self.model_mouse_mask.astype(bool)] * speed_color_light**.4




    def make_mouse_masks(self):
        '''        MAKE MASKS REPRESENTING THE MOUSE POSITIONS        '''

        # initialize arena, pre-stim
        if self.frame_num < self.TS_starting_indices[0]: self.color_arena = self.color_arena_initialize.copy()

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





















    def get_random_transition(self):

        # get RANDOM transition matrix
        transition = np.zeros((self.bins_per_side ** 2, self.bins_per_side ** 2))
        # loop over starting bins
        for x_from in range(self.bins_per_side):
            for y_from in range(self.bins_per_side):
                from_idx = (x_from * self.bins_per_side) + y_from
                # loop over terminal bins
                for x_to in range(self.bins_per_side):
                    for y_to in range(self.bins_per_side):
                        to_idx = (x_to * self.bins_per_side) + y_to
                        # self transition
                        if from_idx == to_idx:
                            transition[from_idx, to_idx] = .8
                        # allo transition
                        elif (x_from == x_to and abs((y_from - y_to)) == 1) or (y_from == y_to and abs((x_from - x_to)) == 1):
                            transition[from_idx, to_idx] = .05

        # normalize the edges
        self.transition = transition / np.sum(transition, axis=0)



    def get_geodesic_transition(self):

        # # get RANDOM transition matrix given environmental constraints
        transition = np.zeros((self.bins_per_side ** 2, self.bins_per_side ** 2))

        # loop over starting bins
        for x_from in tqdm(range(self.bins_per_side)):
            for y_from in range(self.bins_per_side):
                from_idx = (x_from * self.bins_per_side) + y_from

                # check if there's any obstacle in this square
                obstacle_in_the_way = np.sum(self.arena[self.bin_size * y_from:self.bin_size * (y_from + 1), self.bin_size * x_from:self.bin_size * (x_from + 1)] == 90)

                if obstacle_in_the_way:
                    # get the geodesic map of distance from the shelter
                    phi_from = self.phi_masked.copy()
                    phi_from[int(self.bin_size * (y_from + .5)), int(self.bin_size * (x_from + .5))] = 0
                    distance_from = np.array(skfmm.distance(phi_from))

                # loop over terminal bins
                for x_to in range(self.bins_per_side):
                    for y_to in range(self.bins_per_side):
                        to_idx = (x_to * self.bins_per_side) + y_to

                        # self transition
                        if from_idx == to_idx:
                            transition[from_idx, to_idx] = .8
                        # allo transition
                        elif (x_from == x_to and abs((y_from - y_to)) == 1) or (y_from == y_to and abs((x_from - x_to)) == 1):

                            # less likely or impossible transition, if obstacle is in the way
                            if obstacle_in_the_way:
                                distance_from_here = distance_from[int(self.bin_size * (y_to + .5)), int(self.bin_size * (x_to + .5))]
                                # accessible
                                if distance_from_here <= (self.bin_size + 1):
                                    transition[from_idx, to_idx] = .05
                                # no accessible
                                elif distance_from_here > (self.bin_size * np.sqrt(2)):
                                    transition[from_idx, to_idx] = 0
                                # somewhat accessible
                                elif distance_from_here > (self.bin_size + 1):
                                    # get proportion of blockage
                                    angle_to_edge = np.arcsin(self.bin_size / distance_from_here)
                                    p = 1 / np.tan(angle_to_edge)
                                    print(str(p) + ' should be between 0 and 1')
                                    # modify transition probability
                                    transition[from_idx, to_idx] = .05 * (1 - p)

                            else:
                                transition[from_idx, to_idx] = .05

        # normalize the edges
        self.transition = transition / np.sum(transition, axis=0)



    def get_lateral_geodesic_transitions(self):

        # make arena where one side is blocked
        side_blocked_arena = self.arena.copy()
        side_blocked_arena[int(self.arena.shape[0] / 2)-3:int(self.arena.shape[0] / 2)+3, :int(self.arena.shape[1] / 2)] = 90    # block left side
        # side_blocked_arena[int(self.arena.shape[0] / 2)-3:int(self.arena.shape[0] / 2)+3, int(self.arena.shape[1] / 2):] = 90    # block right side

        # # get RANDOM transition matrix given environmental constraints
        transition = np.zeros((self.bins_per_side ** 2, self.bins_per_side ** 2))

        # loop over starting bins
        for x_from in tqdm(range(self.bins_per_side)):
            for y_from in range(self.bins_per_side):
                from_idx = (x_from * self.bins_per_side) + y_from

                # check if there's any obstacle in this square
                obstacle_in_the_way = np.sum(side_blocked_arena[self.bin_size * y_from:self.bin_size * (y_from + 1), self.bin_size * x_from:self.bin_size * (x_from + 1)] == 90)

                if obstacle_in_the_way:
                    # get the geodesic map of distance from current point
                    phi = np.ones_like(self.arena)
                    phi_from = np.ma.MaskedArray(phi, self.mask_left) # SWITCH !!!

                    phi_from[int(self.bin_size * (y_from + .5)), int(self.bin_size * (x_from + .5))] = 0
                    distance_from = np.array(skfmm.distance(phi_from))

                # loop over terminal bins
                for x_to in range(self.bins_per_side):
                    for y_to in range(self.bins_per_side):
                        to_idx = (x_to * self.bins_per_side) + y_to

                        # self transition
                        if from_idx == to_idx:
                            transition[from_idx, to_idx] = .8
                        # allo transition
                        elif (x_from == x_to and abs((y_from - y_to)) == 1) or (y_from == y_to and abs((x_from - x_to)) == 1):

                            # less likely or impossible transition, if obstacle is in the way
                            if obstacle_in_the_way:
                                distance_from_here = distance_from[int(self.bin_size * (y_to + .5)), int(self.bin_size * (x_to + .5))]
                                # accessible
                                if distance_from_here <= (self.bin_size + 1):
                                    transition[from_idx, to_idx] = .05
                                # no accessible
                                elif distance_from_here > (self.bin_size * np.sqrt(2)):
                                    transition[from_idx, to_idx] = 0
                                # somewhat accessible
                                elif distance_from_here > (self.bin_size + 1):
                                    # get proportion of blockage
                                    angle_to_edge = np.arcsin(self.bin_size / distance_from_here)
                                    p = 1 / np.tan(angle_to_edge)
                                    print(str(p) + ' should be between 0 and 1')
                                    # modify transition probability
                                    transition[from_idx, to_idx] = .05 * (1 - p)

                            else:
                                transition[from_idx, to_idx] = .05

        # normalize the edges
        self.transition = transition / np.sum(transition, axis=0)
