import numpy as np
import os
import pickle
import scipy.stats
import scipy.signal
import skfmm
import random
import cv2
import matplotlib.pyplot as plt
plt.style.use('dark_background'); plt.rcParams.update({'font.size': 22})
from Utils.Data_rearrange_funcs import check_session_selected
from Utils.registration_funcs import get_arena_details, model_arena
from tqdm import tqdm
import cupy as cp
cp.cuda.Device(0).use()
from Config import setup
_, _, _, _, _, _, _, selector_type, selector = setup()

'''
---
----
Calculate the parameters to be used in the strategy simulations
---
----
'''

class SR():
    '''     Calculate the parameters to be used in the strategy simulations     '''

    def __init__(self, db):

        # loop across sessions, adding to the input and likelihood
        for session_name in db.index:

            # check if the session is selected
            self.session = db.loc[session_name]
            selected = check_session_selected(self.session.Metadata, selector_type, selector)

            # create the feature array and likelihood output for each selected session
            if selected:

                print(session_name)

                self.extract_coordinates()

                self.compute_quantities()






    def extract_coordinates(self):
        '''         EXTRACT THE SAVED COORDINATES FILE FOR THE CURRENT SESSION      '''

        # find the coordinates file
        video_path = self.session['Metadata'].video_file_paths[0][0]
        processed_coordinates_file = os.path.join(os.path.dirname(video_path), 'coordinates')

        # load the coordinates file
        with open(processed_coordinates_file, "rb") as dill_file:
            self.coordinates = pickle.load(dill_file)

        # get the arena details
        self.x_offset, self.y_offset, self.obstacle_type, self.shelter_location, self.subgoal_location, self.obstacle_changes = \
            get_arena_details(self.session['Metadata'].experiment)

        # load the distance and angle to obstacle data
        self.distance_arena = np.load('C:\\Drive\\DLC\\transforms\\distance_arena_' + self.obstacle_type + '.npy')
        self.angle_arena = np.load('C:\\Drive\\DLC\\transforms\\angle_arena_' + self.obstacle_type + '.npy')

        # get the arena
        self.arena, _, _ = model_arena((self.distance_arena.shape[0], self.distance_arena.shape[1]), True, False, self.obstacle_type, simulate=True)
        self.color_arena = cv2.cvtColor(self.arena, cv2.COLOR_GRAY2BGR)
        self.shelter_location = [int(a * b / 1000) for a, b in zip(self.shelter_location, self.arena.shape)]

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


    def compute_quantities(self):
        '''         COMPUTE THE QUANTITIES NEEDED TO CALCULATE THE DIFFERENCE IN INITIAL CONDITIONS FUNCTION        '''

        # extract start-point and end-point frame number
        self.start_indices = np.where(self.coordinates['start_index'])[0]
        self.end_indices = self.coordinates['start_index'][self.start_indices].astype(int)

        # get all indices in which there's a homing (or not)
        homing_idx = np.ones(len(self.coordinates['center_body_location'][0]), dtype = bool)
        # for s, e in zip(self.start_indices, self.end_indices):
        #     homing_idx[s:e+1] = True

        homing_idx[:30*60*25] = False #!!!

        # bin the position data
        bins_per_side = 20
        bin_size = int(self.arena.shape[0] / bins_per_side)
        bin_array = np.ones(len(self.coordinates['front_location'][0][homing_idx]))*np.nan
        bin_ID = 0
        bin_sum = 0
        if self.coordinates['front_location'][ self.coordinates['front_location'] >= self.arena.shape[0] ].size:
            self.coordinates['front_location'][ self.coordinates['front_location'] >= self.arena.shape[0] ] = self.arena.shape[0] - 1

        for x_bin in range(bins_per_side):
            in_x_bin = (self.coordinates['front_location'][0][homing_idx] >= (bin_size*x_bin) ) * (self.coordinates['front_location'][0][homing_idx] < (bin_size*(x_bin+1)) )

            for y_bin in range(bins_per_side):
                in_y_bin = (self.coordinates['front_location'][1][homing_idx] >= (bin_size * y_bin)) * (self.coordinates['front_location'][1][homing_idx] < (bin_size * (y_bin + 1)))

                bin_array[in_x_bin * in_y_bin] = bin_ID
                bin_sum += np.sum(bin_array)

                bin_ID += 1

        print(np.sum(bin_array))
        bin_array = bin_array.astype(int)

        # get the transition matrix

        # Get counts of (bin_preceding --> bin_following) to (i,j) of bin_counts
        bin_counts = np.zeros((bins_per_side**2, bins_per_side**2), dtype=int)
        np.add.at(bin_counts, (bin_array[:-1], bin_array[1:]), 1)

        # Get transition probs (% in column 1 going to row 1,2,3,etc.)
        transition = (bin_counts.T / np.sum(bin_counts, axis=1)).T
        np.nan_to_num(transition, copy=False)


        # # get RANDOM transition matrix
        # transition = np.zeros((bins_per_side**2, bins_per_side**2))
        #
        # # loop over starting bins
        # for x_from in range(bins_per_side):
        #     for y_from in range(bins_per_side):
        #         from_idx = (x_from * bins_per_side) + y_from
        #
        #         # loop over terminal bins
        #         for x_to in range(bins_per_side):
        #             for y_to in range(bins_per_side):
        #                 to_idx = (x_to * bins_per_side) + y_to
        #
        #                 # self transition
        #                 if from_idx == to_idx:
        #                     transition[from_idx, to_idx] = .8
        #                 # allo transition
        #                 elif (x_from == x_to and abs((y_from - y_to)) == 1) or (y_from == y_to and abs((x_from - x_to)) == 1): # or \
        #                     transition[from_idx, to_idx] = .05
        #
        # # normalize the edges
        # transition = transition / np.sum(transition, axis = 0)



        # # # get RANDOM transition matrix given environmental constraints
        # transition = np.zeros((bins_per_side**2, bins_per_side**2))
        #
        # # loop over starting bins
        # for x_from in tqdm(range(bins_per_side)):
        #     for y_from in range(bins_per_side):
        #         from_idx = (x_from * bins_per_side) + y_from
        #
        #         # check if there's any obstacle in this square
        #         obstacle_in_the_way = np.sum(self.arena[bin_size * y_from:bin_size * (y_from + 1), bin_size * x_from:bin_size * (x_from + 1)] == 90)
        #
        #         if obstacle_in_the_way:
        #             # get the geodesic map of distance from the shelter
        #             phi_from = self.phi_masked.copy()
        #             phi_from[int(bin_size * (y_from + .5)), int(bin_size * (x_from + .5))] = 0
        #             distance_from = np.array(skfmm.distance(phi_from))
        #
        #         # loop over terminal bins
        #         for x_to in range(bins_per_side):
        #             for y_to in range(bins_per_side):
        #                 to_idx = (x_to * bins_per_side) + y_to
        #
        #                 # self transition
        #                 if from_idx == to_idx:
        #                     transition[from_idx, to_idx] = .8
        #                 # allo transition
        #                 elif (x_from == x_to and abs((y_from - y_to)) == 1) or (y_from == y_to and abs((x_from - x_to)) == 1): # or \
        #                      #(abs((y_from - y_to)) == 1 and abs((x_from - x_to)) == 1):
        #
        #                     if obstacle_in_the_way:
        #                         distance_from_here = distance_from[int(bin_size * (y_to + .5)), int(bin_size * (x_to + .5))]
        #                         # accessible
        #                         if distance_from_here <= (bin_size + 1):
        #                             transition[from_idx, to_idx] = .05
        #                         # no accessible
        #                         elif distance_from_here > (bin_size * np.sqrt(2)):
        #                             transition[from_idx, to_idx] = 0
        #                         # somewhat accessible
        #                         elif distance_from_here > (bin_size + 1):
        #                             # get proportion of blockage
        #                             angle_to_edge = np.arcsin(bin_size / distance_from_here)
        #                             p = 1 / np.tan(angle_to_edge)
        #                             print(str(p) + ' should be between 0 and 1')
        #                             # modify transition probability
        #                             transition[from_idx, to_idx] = .05 * (1 - p)
        #
        #                     else:
        #                         transition[from_idx, to_idx] = .05
        #
        #
        #
        # # normalize the edges
        # transition = transition / np.sum(transition, axis = 0)




        # get the SR
        time_discount = .99 # .99 ~ 1 sec is .74 // 5 sec is .2 // 10 sec is .05
        successor_representation = np.linalg.inv(np.identity(bins_per_side**2) - time_discount * transition)

        # display the SR from a selected bin
        click_data = [0,0]

        # initialize GUI
        cv2.startWindowThread()
        cv2.namedWindow('SR')

        # create functions to react to clicked points
        cv2.setMouseCallback('SR', select_SR_points, click_data)  # Mouse callback


        # shelter_indices = [(x, y) for x in range(4, 6) for y in range(8, 10)]
        shelter_indices = [(x, y) for x in range(8, 12) for y in range(15, 20)]
        shelter_indices = [(x, y) for x in range(8, 12) for y in range(5)]
        # shelter_indices = [(x, y) for x in range(12, 18) for y in range(24, 30)]
        # shelter_indices = [(x, y) for x in range(15, 25) for y in range(30, 40)]
        shelter_bin_idx = [(SI[0] * bins_per_side) + SI[1] for SI in shelter_indices]

        x_bin_click = 1
        y_bin_click = 1

        while True:  # take in clicked points and show the SR

            if int(click_data[0] / bin_size) != x_bin_click or int(click_data[1] / bin_size) != y_bin_click:
                x_bin_click = int(click_data[0] / bin_size)
                y_bin_click = int(click_data[1] / bin_size)


                # initialize SR arenaq
                successor_arena = np.zeros(self.arena.shape)

                # get obstacle angle
                obstacle_angle_click = self.angle_arena[int(bin_size * (y_bin_click + .5)), int(bin_size * (x_bin_click + .5))]

                # also use nearby points
                # for x_shift in tqdm(range(-2, 3)):
                #     for y_shift in range(-2, 3):
                for x_shift in tqdm(range(-3, 4)):
                    for y_shift in range(-3, 4):

                        # initialize SR arenaq
                        curr_successor_arena = np.zeros(self.arena.shape)

                        # get the SR for the clicked bin
                        start_bin_idx = ( (x_bin_click + x_shift) * bins_per_side) + (y_bin_click + y_shift)

                        # get the geodesic distance from the clicked bin
                        try:
                            obstacle_angle_shift = self.angle_arena[int(bin_size * (y_bin_click + y_shift + .5)), int(bin_size * (x_bin_click + x_shift + .5))]
                            if np.isnan(obstacle_angle_shift): continue
                        except: continue
                        distance_click_euclid = self.distance_from_shelter_euclid[int(bin_size * (y_bin_click + .5)), int(bin_size * (x_bin_click + .5))]
                        distance_click = self.distance_from_shelter[int(bin_size * (y_bin_click + .5)), int(bin_size * (x_bin_click + .5))]

                        # get the obstacle angle offset
                        obstacle_angle_offset = abs(obstacle_angle_shift-obstacle_angle_click)
                        if obstacle_angle_offset > 180: obstacle_angle_offset = abs(360 - obstacle_angle_offset)

                        if obstacle_angle_offset > 90:
                            print(obstacle_angle_offset)
                            continue

                        # fill in each bin with the successor-ness
                        for x_bin in range(bins_per_side):
                            for y_bin in range(bins_per_side):
                                # get the bin index
                                end_bin_idx = (x_bin * bins_per_side) + y_bin

                                if end_bin_idx != start_bin_idx:
                                    # get the decrease in geodesic distance and euclidean distance
                                    distance_bin = self.distance_from_shelter[int(bin_size * (y_bin + .5)), int(bin_size * (x_bin + .5))]
                                    distance_bin_euclid = self.distance_from_shelter_euclid[int(bin_size * (y_bin + .5)), int(bin_size * (x_bin + .5))]
                                    distance_change = np.max( ( (distance_click - distance_bin), (distance_click_euclid - distance_bin_euclid) ) )

                                    # if np.mean(successor_representation[end_bin_idx, shelter_bin_idx]) > np.mean(successor_representation[start_bin_idx, shelter_bin_idx]) \
                                    #         and distance_change > 12:

                                    curr_successor_arena[int(bin_size * (y_bin + .5)), int(bin_size * (x_bin + .5))] = \
                                        np.mean(successor_representation[end_bin_idx, shelter_bin_idx]) * successor_representation[start_bin_idx, end_bin_idx]

                        successor_arena = successor_arena + (curr_successor_arena / np.max( (1, 4 * np.sqrt(x_shift**2 + y_shift**2)) ) )

                # gaussian blur
                successor_arena = cv2.GaussianBlur(successor_arena, ksize=(181, 181), sigmaX=19, sigmaY=19)

                successor_arena = (successor_arena / np.max(successor_arena) * 255) # / np.max(successor_arena) #.003
                successor_arena[successor_arena > 255] = 255
                successor_arena[self.arena==0] = 0
                successor_arena = successor_arena.astype(np.uint8)
                color_successor = cv2.applyColorMap(cv2.resize(successor_arena, self.arena.shape), cv2.COLORMAP_JET)

                # fill in the origin square black & white
                color_successor[bin_size * y_bin_click:bin_size * (y_bin_click + 1), bin_size * x_bin_click:bin_size * (x_bin_click + 1)] = 0
                cv2.circle(color_successor, (int(bin_size * (x_bin_click + .5)), int(bin_size * (y_bin_click + .5)) ), 5, 255, -1)

                # merge the arena and the SR
                alpha = .2
                cv2.addWeighted(self.color_arena, alpha, color_successor, 1 - alpha, 0, color_successor)
                cv2.imshow('SR', color_successor)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break


        # get the angle output likelihood
        origin = (int(bin_size * (x_bin_click + .5)), int(bin_size * (y_bin_click + .5)))
        degrees = np.linspace(0, 2 * np.pi, 360)
        likelihood = np.zeros_like(degrees)
        landmark_bias = 100
        # successor_arena_copy = successor_arena.copy()

        for i, degree in enumerate(degrees):
            # get the end point for the line, for this angle
            end_point = (int(origin[0] + self.arena.shape[0] * np.cos(degree)), int(origin[1] + self.arena.shape[0] * np.sin(degree)) )

            # make a mask corresponding to this angle
            angle_mask = np.zeros_like(successor_arena)
            cv2.line(angle_mask, origin, end_point, 1, 1)

            # get indices
            angle_indices = np.where(angle_mask)
            if degree > np.pi: angle_indices = tuple(np.flip(angle_indices, axis=1))

            # remove those going past the wall
            mask_obstacle_overlap = np.where(angle_mask * self.mask)
            if mask_obstacle_overlap[0].size:
                intersection_point = np.min(mask_obstacle_overlap[0]) + landmark_bias * np.sin(degree)
                angle_indices = angle_indices[0][angle_indices[0] < intersection_point], angle_indices[1][angle_indices[0] < intersection_point]

            # successor_arena_copy[angle_indices] = 255

            likelihood[i] = np.sum(successor_arena[angle_indices])

        # cv2.imshow('sup', successor_arena_copy)


        # show the likelihood
        likelihood = likelihood / np.sum(likelihood)
        # open figure
        plt.figure(figsize=(18, 8))
        self.ax = plt.subplot(1, 1, 1)
        self.ax.set_title('Likelihood of this Orientation Movement for Repetition Strategy');
        self.ax.set_xlabel('allocentric head direction (degrees)');
        self.ax.set_ylabel('likelihood of HD given strategy')
        self.ax.fill_between(degrees * 180 / np.pi, likelihood, color='darkred', linewidth=4, alpha=.55)



        print('hi')



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


# mouse callback function I
def select_SR_points(event,x,y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:

        data[0] = x
        data[1] = y




# do eigen stuff
# w, v = np.linalg.eig(successor_representation)
# eig_arena = np.zeros(successor_arena.shape)
# for x_bin in range(bins_per_side):
#     for y_bin in range(bins_per_side):
#         eig_idx = (x_bin * bins_per_side) + y_bin
#         eig_arena[bin_size * y_bin:bin_size * (y_bin + 1), bin_size * x_bin:bin_size * (x_bin + 1)] = np.real(v[eig_idx, 1])
#         print(np.real(v[eig_idx, 0]))
#
# eig_arena = ( (eig_arena+np.min(eig_arena)) / (-np.min(eig_arena) + np.max(eig_arena)) * 255).astype(np.uint8)
# cv2.imshow('SR', eig_arena)


# stop after one hump
# line_data = successor_arena[angle_indices]
# local_min = scipy.signal.find_peaks(-line_data, distance=100)[0]
# if local_min.size:
#     intersection_point = angle_indices[0][local_min[0]]
#     if degree > np.pi:
#         angle_indices = angle_indices[0][angle_indices[0] >= intersection_point], angle_indices[1][angle_indices[0] >= intersection_point]
#     else:
#         angle_indices = angle_indices[0][angle_indices[0] <= intersection_point], angle_indices[1][angle_indices[0] <= intersection_point]
#     print(degree)
#     print(local_min)
#     print('')
