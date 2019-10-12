import cv2
import numpy as np
import os
import pickle
import time
import skfmm
from scipy.ndimage import gaussian_filter1d
from Utils.registration_funcs import model_arena

# analysis_dictionary = {}


def proper_analysis(self, trial_types, stims_video, trial_colors, height, width, save_folder, obstacle_type, shelter_location,
                    stims_video_control = None, control = False):
    '''
    GET SPEED TRACES AND THE LIKE
    '''

    '''
    INITIALIZE VARIABLES
    '''
    #initialize variables
    coordinates = self.coordinates
    experiment = self.session['Metadata'].experiment
    mouse = self.session['Metadata'].mouse_id
    save_file = os.path.join(save_folder, 'proper_analysis')

    # for control analysis
    if control:
        experiment = experiment + '_control'
        stims_video = stims_video_control

    # load and initialize dictionary
    while True:
        try:
            with open(save_file, 'rb') as dill_file:
                analysis_dictionary = pickle.load(dill_file)
            break
        except:
            print('file in use...')
            time.sleep(5)


    if not experiment in analysis_dictionary:
        analysis_dictionary[experiment] = {}
        analysis_dictionary[experiment]['obstacle'] = {}
        analysis_dictionary[experiment]['no obstacle'] = {}

    # get postion, speed, speed toward shelter (or subgoal)
    # position = coordinates['butty_location']
    position = coordinates['center_location']
    front_position = coordinates['front_location']
    # butt_position = coordinates['butty_location']
    angles = coordinates['body_angle']
    shelter_angles = coordinates['shelter_angle']
    # speed = coordinates['speed']
    # escape_speed = coordinates['speed_toward_subgoal']
    delta_position = np.concatenate((np.zeros((2, 1)), np.diff(position)), axis=1)
    speed = np.sqrt(delta_position[0, :] ** 2 + delta_position[1, :] ** 2)

    distance_from_shelter = coordinates['distance_from_shelter']

    # start_idx = np.where(coordinates['start_index'])[0]


    # initialize the map
    arena, _, _ = model_arena((height, width), 2, False, obstacle_type, simulate=True)
    shelter_location = [int(a*b/1000) for a, b in zip(shelter_location, arena.shape)]
    phi = np.ones_like(arena)
    mask = (arena == 90)
    phi_masked = np.ma.MaskedArray(phi, mask)
    distance_map = {}

    # get the geodesic map of distance from the shelter
    phi_from_shelter = phi_masked.copy()
    phi_from_shelter[shelter_location[1], shelter_location[0]] = 0
    distance_map['obstacle'] = np.array(skfmm.distance(phi_from_shelter))

    # get the euclidean map of distance from the shelter
    phi[shelter_location[1], shelter_location[0]] = 0
    distance_map['no obstacle'] = np.array(skfmm.distance(phi))


    analysis_dictionary[experiment]['obstacle']['shape'] = (height, width)
    analysis_dictionary[experiment]['obstacle']['type'] = obstacle_type
    analysis_dictionary[experiment]['obstacle']['scale'] = 1
    analysis_dictionary[experiment]['obstacle']['direction scale'] = 8

    # separate by epochs of trial type
    if -1 in trial_types:
        wall_down_idx = np.where(np.array(trial_types)==-1)[0][0]
        wall_up_epoch = list(range(0,stims_video[wall_down_idx]))
        wall_down_epoch = list(range(stims_video[wall_down_idx] + 300, len(speed)))

    elif 1 in trial_types:
        wall_up_idx = np.where(np.array(trial_types)==1)[0][0]
        wall_down_epoch = list(range(0,stims_video[wall_up_idx]))
        wall_up_epoch = list(range(stims_video[wall_up_idx] + 300, len(speed)))

    else: #if all(x==2 for x in trial_types):
        wall_up_epoch = list(range(0, len(speed) ))
        wall_down_epoch = []


    # TEMPORARY
    if True:
        '''
        EXPLORATION HEAT MAP
        '''
        if not 'exploration' in analysis_dictionary[experiment]['obstacle']:
            analysis_dictionary[experiment]['obstacle']['exploration'] = {}
            analysis_dictionary[experiment]['no obstacle']['exploration'] = {}

        # get exploration heat map for each epoch
        for i, epoch in enumerate([wall_up_epoch, wall_down_epoch]):
            if not epoch:
                continue

            if 'no shelter' in experiment and 'down' in experiment and i == 0:
                epoch = list(range(0,stims_video[wall_down_idx] - 6*30*60))

            # Histogram of positions
            scale = analysis_dictionary[experiment]['obstacle']['scale']
            H, x_bins, y_bins = np.histogram2d(position[0, epoch], position[1, epoch], [np.arange(0, width + 1, scale),
                                                                            np.arange(0, height + 1, scale)], normed=True)
            H = H.T
            # H[H > 0] = 1

            # make into uint8 image
            H_image = (H * 255 / np.max(H)).astype(np.uint8)
            # H_image = cv2.resize(H_image, (width, height), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('heat map', H_image)

            # put into dictionary
            if i==0: analysis_dictionary[experiment]['obstacle']['exploration'][mouse] = H_image
            elif i==1: analysis_dictionary[experiment]['no obstacle']['exploration'][mouse] = H_image


        '''
        EXPLORATION DIRECTIONALITY
        '''
        if not 'direction' in analysis_dictionary[experiment]['obstacle']:
            analysis_dictionary[experiment]['obstacle']['direction'] = {}
            analysis_dictionary[experiment]['no obstacle']['direction'] = {}

        # get exploration heat map for each epoch
        for i, epoch in enumerate([wall_up_epoch, wall_down_epoch]):
            if not epoch:
                continue

            cur_angles = angles[epoch]

            # Histogram of positions
            scale = int(height / analysis_dictionary[experiment]['obstacle']['direction scale'])
            H, x_bins, y_bins = np.histogram2d(position[0, epoch], position[1, epoch], [np.arange(0, width + 1, scale),
                                                                            np.arange(0, height + 1, scale)], normed=True)
            x_bin_num = np.digitize(position[0, epoch], x_bins)
            y_bin_num = np.digitize(position[1, epoch], y_bins)
            bin_array = (x_bin_num-1) + (y_bin_num-1) * H.shape[1]

            # Get counts of (bin_preceding --> bin_following) to (i,j) of bin_counts
            bin_counts = np.zeros((H.shape[0] ** 2, H.shape[1] ** 2), dtype=int)
            np.add.at(bin_counts, (bin_array[:-1], bin_array[1:]), 1)

            # Get transition probs (% in column 1 going to row 1,2,3,etc.)
            transition = (bin_counts.T / np.sum(bin_counts, axis=1)).T
            np.nan_to_num(transition, copy=False)

            # get the SR
            time_discount = .96  # .99 ~ 1 sec is .74 // 5 sec is .2 // 10 sec is .05
            successor_representation = np.linalg.inv(np.identity(H.shape[0] ** 2) - time_discount * transition)

            # visualize the directionality
            direction_image = np.zeros((height, width), np.uint8)
            for x in range(H.shape[1]):
                for y in range(H.shape[0]):
                    from_idx = x + y * H.shape[1]

                    origin = int((x + .5) * scale), int((y + .5) * scale)
                    cv2.circle(direction_image, origin, 5, 60, -1)

                    for x_move in range(-1,2):
                        for y_move in range(-1,2):
                            to_idx = (x + x_move) + (y + y_move) * H.shape[1]

                            try:
                                endpoint = int(origin[0] + .05*scale * x_move * successor_representation[from_idx, to_idx]), \
                                           int(origin[1] + .05*scale * y_move * successor_representation[from_idx, to_idx])
                                cv2.arrowedLine(direction_image, origin, endpoint, 255, 2, tipLength = .15)
                            except: pass

                    cv2.imshow('angles', direction_image)

            # put into dictionary
            if i==0: analysis_dictionary[experiment]['obstacle']['direction'][mouse] = successor_representation
            elif i==1: analysis_dictionary[experiment]['no obstacle']['direction'][mouse] = successor_representation


    '''
    ABSOLUTE AND GEODESIC SPEED TRACES
    '''
    for condition in ['obstacle', 'no obstacle']:
        if not 'lunge' in analysis_dictionary[experiment][condition]: # or True:
            analysis_dictionary[experiment][condition]['speed'] = {}
            analysis_dictionary[experiment][condition]['geo speed'] = {}
            analysis_dictionary[experiment][condition]['HD'] = {}
            analysis_dictionary[experiment][condition]['escape'] = {}
            analysis_dictionary[experiment][condition]['time'] = {}
            analysis_dictionary[experiment][condition]['optimal path length'] = {}
            analysis_dictionary[experiment][condition]['actual path length'] = {}
            analysis_dictionary[experiment][condition]['path'] = {}
            analysis_dictionary[experiment][condition]['RT'] = {}
            analysis_dictionary[experiment][condition]['SR'] = {}
            analysis_dictionary[experiment][condition]['IOM'] = {}
            analysis_dictionary[experiment][condition]['lunge'] = {}



        analysis_dictionary[experiment][condition]['speed'][mouse] = []
        analysis_dictionary[experiment][condition]['geo speed'][mouse] = []
        analysis_dictionary[experiment][condition]['HD'][mouse] = []
        analysis_dictionary[experiment][condition]['escape'][mouse] = []
        analysis_dictionary[experiment][condition]['time'][mouse] = [[],[]]
        analysis_dictionary[experiment][condition]['optimal path length'][mouse] = []
        analysis_dictionary[experiment][condition]['actual path length'][mouse] = []
        analysis_dictionary[experiment][condition]['path'][mouse] = []
        analysis_dictionary[experiment][condition]['RT'][mouse] = []
        analysis_dictionary[experiment][condition]['SR'][mouse] = []
        analysis_dictionary[experiment][condition]['IOM'][mouse] = []
        analysis_dictionary[experiment][condition]['lunge'][mouse] = []


    # loop across stimuli
    for i, stim in enumerate(stims_video):

        # check if obstacle or no obstacle
        if stim in wall_up_epoch: condition = 'obstacle'
        elif stim in wall_down_epoch: condition = 'no obstacle'
        else: continue

        # get indices corresponding to the next 12 seconds
        threat_idx = np.arange(stim - self.fps*4, stim + self.fps*10).astype(int)
        stim_idx = np.arange(stim, stim + self.fps * 9).astype(int)

        # get the start time and end time
        analysis_dictionary[experiment][condition]['time'][mouse][0].append(stim / self.fps / 60)

        # get the speed
        analysis_dictionary[experiment][condition]['speed'][mouse].append(list(speed[threat_idx]))

        # get the geodesic speed
        threat_idx_mod = np.concatenate((np.ones(1, int) * threat_idx[0] - 1, threat_idx))
        threat_position = position[0][threat_idx_mod].astype(int), position[1][threat_idx_mod].astype(int)
        geo_location = distance_map[condition][threat_position[1], threat_position[0]]
        geo_speed = np.diff(geo_location)
        analysis_dictionary[experiment][condition]['geo speed'][mouse].append(list(geo_speed))

        # get the HD rel to the HV
        analysis_dictionary[experiment][condition]['HD'][mouse].append(list(shelter_angles[threat_idx]))

        # get the idx when at shelter, and trim threat idx if applicable, and label as completed escape for nah
        arrived_at_shelter = np.where(distance_from_shelter[stim_idx] < 60)[0]


        '''
        get the SR for the trial
        '''
        # just use homings!
        start_idx = np.where(coordinates['start_index'][:stim])[0]
        end_idx = coordinates['start_index'][start_idx]

        # join them together!
        # start_idx_caut = np.array([]); end_idx_caut = np.array([])
        # remove_this_bout = False
        #
        # it = enumerate(zip(start_idx, end_idx))
        # for j, (s, e) in it: #enumerate(zip(start_idx, end_idx)):
        #
        #     # if it's the last one, we're done
        #     if j == (len(start_idx) - 1):
        #         break
        #
        #     # if the end of this beginning is another beginning's end, change the end index
        #     next_bout = 1
        #     while True:
        #         try:
        #             if end_idx[j+next_bout-1] == start_idx[j+next_bout] - 1:
        #                 next_bout +=1
        #             else: break
        #         except: break
        #
        #     start_idx_caut = np.append(start_idx_caut, s)
        #     end_idx_caut = np.append(end_idx_caut, end_idx[j + next_bout - 1])
        #
        #     # skip the sub-bouts
        #     for _ in range(next_bout-1):
        #         next(it)

        # get the x values at center
        x_SH = []; y_SH = []; thru_center = []; SH_time = []
        scaling_factor = 100 / arena.shape[0]
        center_y = 45
        for j, (s, e) in enumerate(zip(start_idx, end_idx)):
            homing_idx = np.arange(s, e).astype(int)
            path = coordinates['center_location'][0][homing_idx] * scaling_factor, coordinates['center_location'][1][homing_idx] * scaling_factor

            '''
            EXCLUDE IF STARTS TOO LOW, OR NEVER GOES INSIDE X OF OBSTACLE, OR NEVER GETS CLOSE TO Y=45
            '''
            # if to low or too lateral don't use
            # SHOULD BE path[1][0] > 40
            if path[1][0] > 40 or ( not np.sum(abs(path[1]-45) < 5) ) or not ( np.sum( (abs(path[0]-50) < 24.5) * (path[1] < 50)) ):
                continue
            if j:
                if path[1][0] > 27 and (not s == end_idx[j-1]+1):
                    print('block')
                    continue

            # print(path[1][0])
            center_idx = np.argmin(abs(path[1] - 45))
            x_SH.append(path[0][center_idx])

            edge_idx = np.argmin(abs(path[1] - 50))
            y_SH.append(path[1][edge_idx])

            SH_time.append(s / 30 / 60)

            # find true endpoint and location past wall
            if path[1][0] < 25 and np.max(path[1]) > 70 and abs(path[0][np.argmin(abs(path[1] - 45))] - 50) < 15 and \
              abs(path[0][np.argmin(abs(path[1] - 55))] - 50) < 15:
                thru_center.append(True)
                cv2.circle(arena, (int(path[0][0] / scaling_factor), int(path[1][0] / scaling_factor)), 6, 100, -1)
            else: thru_center.append(False)

            cv2.circle(arena, (int(path[0][0] / scaling_factor), int(path[1][0] / scaling_factor)), 4, 0, 2)
            cv2.imshow('SH starts', arena)
            cv2.waitKey(1)

        analysis_dictionary[experiment][condition]['SR'][mouse].append( [x_SH, y_SH, thru_center, SH_time] )


        '''
        get the reaction time
        '''
        scaling_factor = 100 / height
        trial_subgoal_speed = [s * scaling_factor * 30 for s in geo_speed]
        subgoal_speed_trace = gaussian_filter1d(trial_subgoal_speed, 2)
        initial_speed = np.where(-subgoal_speed_trace[4*30:] > 15)[0]

        if arrived_at_shelter.size and initial_speed.size:
            analysis_dictionary[experiment][condition]['escape'][mouse].append(True)
            analysis_dictionary[experiment][condition]['time'][mouse][1].append(arrived_at_shelter[0])
            analysis_dictionary[experiment][condition]['path'][mouse].append(
                    (position[0][stim:stim+arrived_at_shelter[0]], position[1][stim:stim+arrived_at_shelter[0]]))

            RT = initial_speed[0] / 30
            analysis_dictionary[experiment][condition]['RT'][mouse].append(RT)

            # get the start position
            start_position = int(position[0][stim+int(RT*30)]), int(position[1][stim+int(RT*30)])

            # get the optimal path length
            optimal_path_length = distance_map[condition][start_position[1], start_position[0]]
            analysis_dictionary[experiment][condition]['optimal path length'][mouse].append(optimal_path_length)

            # get the actual path length
            # actual_path_length = np.sum(speed[stim+int(RT*30):stim+arrived_at_shelter[0]])
            actual_path_length = np.sum(speed[stim:stim + arrived_at_shelter[0]])
            analysis_dictionary[experiment][condition]['actual path length'][mouse].append(actual_path_length+60)

            '''
            get the start and end of the first escape lunge
            '''
            # just use current escape
            start_idx = np.where(coordinates['start_index'][stim:stim + 30 * 9])[0] + stim
            end_idx = coordinates['start_index'][start_idx]

            for bout_idx in start_idx:
                end_idx = int(coordinates['start_index'][bout_idx])
                bout_start_position = int(front_position[0][bout_idx]), int(front_position[1][bout_idx])
                bout_end_position = int(front_position[0][end_idx]), int(front_position[1][end_idx])

                # get euclidean distance change
                euclid_dist_change = distance_from_shelter[bout_idx] - distance_from_shelter[end_idx]

                # get geodesic distance change
                geodesic_dist_change = distance_map[condition][bout_start_position[1], bout_start_position[0]] - \
                                       distance_map[condition][bout_end_position[1], bout_end_position[0]]

                if euclid_dist_change > 40 or geodesic_dist_change > 40:
                    break


            try: analysis_dictionary[experiment][condition]['lunge'][mouse].append([bout_start_position, bout_end_position])
            except: pass
            # print([bout_start_position, bout_end_position])

            bout_used = False
            for bout_idx in start_idx:
                end_idx = int(coordinates['start_index'][bout_idx])
                bout_start_position = int(front_position[0][bout_idx]), int(front_position[1][bout_idx])
                bout_end_position = int(front_position[0][end_idx]), int(front_position[1][end_idx])

                # get euclidean distance change
                euclid_dist_change = distance_from_shelter[bout_idx] - distance_from_shelter[end_idx]

                # get geodesic distance change
                geodesic_dist_change = distance_map[condition][bout_start_position[1], bout_start_position[0]] - \
                                       distance_map[condition][bout_end_position[1], bout_end_position[0]]

                if euclid_dist_change > 10 or geodesic_dist_change > 10:
                    break

            try: analysis_dictionary[experiment][condition]['IOM'][mouse].append([bout_start_position, bout_end_position])
            except: pass
            # print([bout_start_position, bout_end_position])


        else:
            analysis_dictionary[experiment][condition]['escape'][mouse].append(False)
            analysis_dictionary[experiment][condition]['time'][mouse][1].append(np.nan)
            analysis_dictionary[experiment][condition]['path'][mouse].append(np.nan)
            analysis_dictionary[experiment][condition]['optimal path length'][mouse].append(np.nan)
            analysis_dictionary[experiment][condition]['actual path length'][mouse].append(np.nan)
            analysis_dictionary[experiment][condition]['RT'][mouse].append(np.nan)
            analysis_dictionary[experiment][condition]['IOM'][mouse].append(np.nan)
            analysis_dictionary[experiment][condition]['lunge'][mouse].append(np.nan)

        if 'control' in experiment:
            analysis_dictionary[experiment][condition]['escape'][mouse][-1] = None
            analysis_dictionary[experiment][condition]['path'][mouse][-1] = \
                (position[0][stim:stim+300], position[1][stim:stim+300])
            analysis_dictionary[experiment][condition]['speed'][mouse][-1] = speed[stim:stim + 300]






    '''
    SAVE RESULTS
    '''
    # save the dictionary
    while True:
        try:
            with open(save_file, "wb") as dill_file:
                pickle.dump(analysis_dictionary, dill_file)
            break
        except:
            print('file in use...')
            time.sleep(5)




  # # get a bins x bins x 100 array ~ a histogram of body angle for each spaital bin
  #       angle_bins = np.linspace(-180, 180, 101)
  #       angle_hist = np.zeros((len(y_bins)-1, len(x_bins)-1, len(angle_bins)-1))
  #
  #       for x in range(H.shape[1]):
  #           for y in range(H.shape[1]):
  #               x_bin_idx = np.where(x_bin_num==(x+1))
  #               y_bin_idx = np.where(y_bin_num == (y + 1))
  #               bin_idx = np.intersect1d(x_bin_idx, y_bin_idx)
  #
  #               if bin_idx.size:
  #                   angles_in_bin = cur_angles[bin_idx]
  #                   H_angle, _ = np.histogram(angles_in_bin, angle_bins, normed=True)
  #                   angle_hist[y,x,:] = H_angle
  #
  #       # visualize the directionality
  #       direction_image = np.zeros((height, width), np.uint8)
  #       for x in range(angle_hist.shape[1]):
  #           for y in range(angle_hist.shape[0]):
  #               origin = int((x + .5) * scale), int((y + .5) * scale)
  #               cv2.circle(direction_image, origin, 5, 60, -1)
  #
  #               for i, angle in enumerate(angle_bins[1:]):
  #                   endpoint = int(origin[0] + 3*scale * angle_hist[y,x,i] * np.cos(np.deg2rad(angle))), \
  #                              int(origin[1] - 3*scale * angle_hist[y,x,i] * np.sin(np.deg2rad(angle)))
  #                   cv2.line(direction_image, origin, endpoint, 255, 1)
  #
  #               cv2.imshow('angles', direction_image)
