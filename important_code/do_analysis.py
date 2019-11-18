import os
import dill as pickle
import numpy as np
import skfmm
import itertools
from scipy.ndimage import gaussian_filter1d
from helper_code.registration_funcs import get_arena_details, model_arena


class analyze_data():
    def __init__(self, analysis_object, dataframe, analysis_type):
        '''     initialize quantities for analyzing data         '''
        # list of quantities to analyze
        if analysis_type == 'exploration': self.quantities_to_analyze =['exploration']
        if analysis_type == 'traversals': self.quantities_to_analyze =['back traversal', 'front traversal']
        if analysis_type == 'escape paths': self.quantities_to_analyze = ['speed', 'start time', 'end time', 'path', 'edginess']
        if analysis_type == 'edginess': self.quantities_to_analyze = ['speed', 'geo speed', 'HD', 'escape', \
            'start time', 'end time','optimal path length', 'optimal RT path length','full path length','RT path length', \
             'path', 'edginess','RT', 'SR', 'in center'] # 'IOM', 'lunge']
        if analysis_type == 'efficiency': self.quantities_to_analyze = ['speed', 'geo speed', 'HD', 'escape', \
            'start time', 'end time','optimal path length', 'optimal RT path length','full path length','RT path length', \
             'path', 'edginess','RT', 'SR', 'in center'] # 'IOM', 'lunge']
        if analysis_type == 'speed traces': self.quantities_to_analyze = ['speed', 'geo speed', 'HD', 'escape', \
            'start time', 'end time','optimal path length', 'optimal RT path length','full path length','RT path length', \
             'path', 'edginess','RT', 'SR', 'in center'] # 'IOM', 'lunge']
        self.conditions = ['obstacle', 'no obstacle', 'probe']
        self.analysis = analysis_object.analysis
        self.control = analysis_object.analysis_options['control']
        # run analysis
        self.main(analysis_object, dataframe, analysis_type)


    def main(self, analysis_object, dataframe, analysis_type):
        '''    analyze data from each selected session      '''
        # loop over all selected sessions
        for i, session_name in enumerate(analysis_object.selected_sessions):
            # get the metadata
            self.session = dataframe.db.loc[session_name]
            self.experiment = self.session['Metadata']['experiment']
            self.mouse = self.session['Metadata']['mouse_id']
            if self.control: self.mouse= 'control'
            self.scaling_factor = 100 / self.session['Registration'][4][0]
            self.fps = self.session['Metadata']['videodata'][0]['Frame rate'][0]
            # format the analysis dictionary
            self.format_analysis_dict(i)
            # get indices when the stimulus is or was just on
            self.get_stim_idx()
            # loop over all videos
            for vid_num in range(len(self.stims_all)):
                self.vid_num = vid_num
                # load the saved coordinates
                coordinates = self.load_coordinates(dataframe, session_name)
                # get control frames
                if self.control: self.get_control_stim_frames(coordinates)
                # get the time periods when the wall is up or down
                total_frames = len(coordinates['speed'])
                self.get_obstacle_epochs(total_frames)
                # loop over both epochs
                for i, epoch in enumerate([self.wall_up_epoch, self.wall_down_epoch, self.probe_epoch]):
                    if not epoch: continue
                    else: self.epoch = epoch; self.condition = self.conditions[i]; self.start_point = self.start_points[i]
                    # Plot all traversals across the arena
                    if analysis_type == 'traversals': self.analyze_traversals(coordinates)
                    # Make an exploration heat map
                    if analysis_type == 'exploration': self.analyze_exploration(coordinates)
                    # Get speed traces
                    if analysis_type == 'escapes': self.analyze_escapes(coordinates)
                    # Compare various quantities across conditions
                    if analysis_type == 'comparisons': self.analyze_comparisons(coordinates)
        # save analysis
        for experiment in analysis_object.experiments:
            save_folder = os.path.join( analysis_object.dlc_settings['clips_folder'], experiment, analysis_type)
            with open(save_folder, "wb") as dill_file: pickle.dump(self.analysis[experiment], dill_file)



    def analyze_traversals(self, coordinates):
        '''     analyze traversals across the arena         '''
        # take the position data
        x_location = coordinates['center_location'][0][self.epoch] * self.scaling_factor
        y_location = coordinates['center_location'][1][self.epoch] * self.scaling_factor
        # calculate when the mouse is in each section (# - 5*('void' in experiment))
        back = 25; front = 75;  middle = 50
        back_idx = y_location < back; front_idx = y_location > front
        back_half_idx = y_location < middle; front_half_idx = y_location > middle
        center_back_idx = back_half_idx * ~back_idx; center_front_idx = front_half_idx * ~front_idx
        # initialize lists
        traversals_from_back = []; traversals_from_front = []
        traversals_time_back = []; traversals_time_front = []
        escapes_from_back = []; escape_times = []
        # loop over front and back
        for j, location_idx in enumerate([center_back_idx, center_front_idx]):
            idx = 0;
            group_length = 0
            for k, g in itertools.groupby(location_idx):
                idx += group_length
                frames_grouped = list(g)
                group_length = len(frames_grouped)
                # must not start the video
                if not idx: continue
                # must be in the right section
                if not k: continue
                # must start at front or back
                if abs(y_location[idx] - 50) < 10: continue
                # must end in middle
                if abs(y_location[idx + group_length - 1] - 50) > 10: continue
                # a stimulus-evoked escape
                if (idx + self.start_point in self.stim_idx[self.vid_num] or idx + self.start_point + group_length - 1 in self.stim_idx[self.vid_num]):
                    if j == 0:
                        escapes_from_back.append((x_location[idx:idx + group_length - 1] / self.scaling_factor,
                                                       y_location[idx:idx + group_length - 1] / self.scaling_factor))
                        escape_times.append(idx)
                        continue
                # for back traversals
                if j == 0:
                    traversals_from_back.append((x_location[idx:idx + group_length - 1] / self.scaling_factor,
                                                   y_location[idx:idx + group_length - 1] / self.scaling_factor))
                    traversals_time_back.append(idx)
                # for front traversals
                elif j == 1:
                    traversals_from_front.append((x_location[idx:idx + group_length - 1] / self.scaling_factor,
                                                    y_location[idx:idx + group_length - 1] / self.scaling_factor))
                    traversals_time_front.append(idx)

        # add to the analysis dictionary
        self.analysis[self.experiment][self.condition]['back traversal'][self.mouse] = [traversals_from_back, traversals_time_back, escapes_from_back, escape_times]
        self.analysis[self.experiment][self.condition]['front traversal'][self.mouse] = [traversals_from_front, traversals_time_front, [], []]



    def analyze_exploration(self, coordinates):
        '''       analyze explorations         '''
        # Histogram of positions
        position = coordinates['center_location']
        height, width = self.analysis[self.experiment][self.condition]['shape'][0], self.analysis[self.experiment][self.condition]['shape'][1]
        H, x_bins, y_bins = \
            np.histogram2d(position[0, self.epoch], position[1, self.epoch], [np.arange(0, width + 1), np.arange(0, height + 1)], normed=True)
        H = H.T

        # put into dictionary
        self.analysis[self.experiment][self.condition]['exploration'][self.mouse] = H



    def analyze_escapes(self, coordinates):
        '''       analyze speed, geodesic speed, trial time, reaction time        '''
        # initialize data
        distance_from_shelter = coordinates['distance_from_shelter']
        position = coordinates['center_location']
        # position = coordinates['front_location']
        angular_speed = coordinates['angular_speed']
        delta_position = np.concatenate((np.zeros((2, 1)), np.diff(position)), axis=1)
        speed = np.sqrt(delta_position[0, :] ** 2 + delta_position[1, :] ** 2)
        distance_map = self.get_distance_map()
        print(self.mouse); print(self.condition)
        # loop over each trial
        for stim in self.stims_all[self.vid_num]:
            # make sure it's the right epoch
            if not stim in self.epoch: continue
            if len(position[0]) - stim < 600: continue
            # get the amount of time spent in the central region
            frames_in_center = np.sum((position[0][self.epoch[0]:stim] * self.scaling_factor > 25) * (position[0][self.epoch[0]:stim] * self.scaling_factor < 75) * \
                                 (position[1][self.epoch[0]:stim] * self.scaling_factor > 45) * (position[1][self.epoch[0]:stim] * self.scaling_factor < 55))
            print(frames_in_center)
            self.analysis[self.experiment][self.condition]['in center'][self.mouse].append(frames_in_center)
            # get peri-stimulus indices
            threat_idx = np.arange(stim - self.fps * 10, stim + self.fps * 18).astype(int)
            stim_idx = np.arange(stim, stim + self.fps * 18).astype(int)
            # get the path
            self.analysis[self.experiment][self.condition]['path'][self.mouse].append([position[0][stim_idx], position[1][stim_idx]])
            # get the start time
            self.analysis[self.experiment][self.condition]['start time'][self.mouse].append(stim / self.fps / 60)
            # get the speed
            self.analysis[self.experiment][self.condition]['speed'][self.mouse].append(list(speed[threat_idx]))
            # get the geodesic speed
            threat_idx_mod = np.concatenate((np.ones(1, int) * threat_idx[0] - 1, threat_idx))
            threat_position = position[0][threat_idx_mod].astype(int), position[1][threat_idx_mod].astype(int)
            geo_location = distance_map[self.condition][threat_position[1], threat_position[0]]
            geo_speed = np.diff(geo_location)
            self.analysis[self.experiment][self.condition]['geo speed'][self.mouse].append(list(geo_speed))
            # get the idx when at shelter, and trim threat idx if applicable, and label as completed escape or nah
            arrived_at_shelter = np.where(distance_from_shelter[stim_idx] < 60)[0]
            # get the reaction time
            trial_subgoal_speed = [s * self.scaling_factor * 30 for s in geo_speed]
            subgoal_speed_trace = gaussian_filter1d(trial_subgoal_speed, 4)  # 2
            RT_speed = 15
            initial_speed = np.where((-subgoal_speed_trace[10 * 30:] > RT_speed))[0]
            # results for successful escapes
            if arrived_at_shelter.size and initial_speed.size and not ('no shelter' in self.experiment and not 'down' in self.experiment) \
                    and arrived_at_shelter[0] > 10:
                self.analysis[self.experiment][self.condition]['end time'][self.mouse].append(arrived_at_shelter[0])
                self.analysis[self.experiment][self.condition]['RT'][self.mouse].append(initial_speed[0] / 30)
                # get the optimal path length
                optimal_path_length = distance_map[self.condition][int(position[1][stim]), int(position[0][stim])]
                optimal_path_length_RT = distance_map[self.condition][int(position[1][stim+initial_speed[0]]), int(position[0][stim+initial_speed[0]])]
                self.analysis[self.experiment][self.condition]['optimal path length'][self.mouse].append(optimal_path_length)
                self.analysis[self.experiment][self.condition]['optimal RT path length'][self.mouse].append(optimal_path_length_RT)
                # get the actual path length
                full_path_length = np.sum(speed[stim:stim + arrived_at_shelter[0]])
                RT_path_length = np.sum(speed[stim+initial_speed[0]:stim+arrived_at_shelter[0]])
                self.analysis[self.experiment][self.condition]['full path length'][self.mouse].append(full_path_length+60)
                self.analysis[self.experiment][self.condition]['RT path length'][self.mouse].append(RT_path_length+60)

                # print(optimal_path_length / (full_path_length + 60) )
                # print(optimal_path_length_RT / (RT_path_length + 60) )


                '''     get edginess        '''
                # check for pauses in escape
                initial_speed = np.where((-subgoal_speed_trace[10 * 30:] > RT_speed) * (angular_speed[threat_idx[10*30:]] < 5))[0]
                wrong_way = subgoal_speed_trace[10*30:10*30+arrived_at_shelter[0]] > 5
                up_top =  (position[1][stim_idx[:arrived_at_shelter[0]]] * self.scaling_factor) < 25
                # if goes the wrong way while in the back, doesn't count as RT
                speed_wrong_way = np.where(wrong_way * up_top)[0]  # 15
                if speed_wrong_way.size:
                    initial_speed = initial_speed[initial_speed > speed_wrong_way[0]]
                # if passes the back, counts as the start of trajectory
                if up_top.size:
                    if not up_top[initial_speed[0]]:
                        up_top_at_start = np.where(up_top[:initial_speed[0]])[0]
                        if up_top_at_start.size:
                            initial_speed[0] = up_top_at_start[-1]
                x_pos = position[0][stim_idx[initial_speed[0]:arrived_at_shelter[0]]] * self.scaling_factor
                y_pos = position[1][stim_idx[initial_speed[0]:arrived_at_shelter[0]]] * self.scaling_factor
                # switch around if shelter-on-side
                if 'side' in self.experiment:
                    cache = y_pos.copy(); y_pos = x_pos; x_pos = cache
                    if y_pos[0] > 50:
                        y_pos = 100 - y_pos
                # get where to evaluate trajectory
                # distance_eval_point = 30 #15
                # distance_travelled = np.sqrt((x_pos - x_pos[0])**2+(y_pos - y_pos[0])**2)
                # eval_idx = np.where( (distance_travelled > distance_eval_point) )[0][0]
                # y_eval_point = 35
                y_eval_point = 40
                # y_eval_point = y_pos[eval_idx]
                y_wall_point = 45
                # x_pos = position[0][stim_idx[initial_speed[0]:]] * self.scaling_factor
                # y_pos = position[1][stim_idx[initial_speed[0]:]] * self.scaling_factor
                x_pos_start = x_pos[0]
                y_pos_start = y_pos[0]
                # do line from starting position to shelter
                y_pos_shelter = self.shelter_location[1] / 10
                x_pos_shelter = self.shelter_location[0] / 10
                slope = (y_pos_shelter - y_pos_start) / (x_pos_shelter - x_pos_start)
                intercept = y_pos_start - x_pos_start * slope
                distance_to_line = abs(y_pos - slope * x_pos - intercept) / np.sqrt((-slope) ** 2 + (1) ** 2)
                # # get index at center point (wall location)
                mouse_at_eval_point = np.argmin(abs(y_pos - y_eval_point))
                homing_vector_at_center = (y_eval_point - intercept) / slope
                # get offset from homing vector
                linear_offset = distance_to_line[mouse_at_eval_point]
                # get line to the closest edge
                mouse_at_wall = np.argmin(abs(y_pos - y_wall_point))
                y_edge = 50
                if x_pos[mouse_at_wall] > 50: x_edge = 75# + 5
                else: x_edge = 25# - 5
                # do line from starting position to edge position
                slope = (y_edge - y_pos_start) / (x_edge - x_pos_start)
                intercept = y_pos_start - x_pos_start * slope
                distance_to_line = abs(y_pos - slope * x_pos - intercept) / np.sqrt((-slope) ** 2 + (1) ** 2)
                edge_offset = np.mean(distance_to_line[mouse_at_eval_point])
                # compute the max possible deviation
                edge_vector_at_center = (y_eval_point - intercept) / slope
                line_to_edge_offset = abs(homing_vector_at_center - edge_vector_at_center)  # + 5
                # get edginess
                # edginess = np.min((1, (linear_offset - edge_offset + line_to_edge_offset) / (2 * line_to_edge_offset)))
                edginess = (linear_offset - edge_offset + line_to_edge_offset) / (2 * line_to_edge_offset) #* np.sign(x_edge-50)
                self.analysis[self.experiment][self.condition]['edginess'][self.mouse].append(edginess)
                # print(edginess)

                '''     analyze previous homings!       '''
                # get the x values at center
                x_SH = []; y_SH = []; thru_center = []; SH_time = []
                center_y = 45
                start_idx = np.where(coordinates['start_index'][:stim])[0]
                end_idx = coordinates['start_index'][start_idx]

                for j, (s, e) in enumerate(zip(start_idx, end_idx)):
                    homing_idx = np.arange(s, e).astype(int)
                    path = coordinates['center_location'][0][homing_idx] * self.scaling_factor, \
                           coordinates['center_location'][1][homing_idx] * self.scaling_factor
                    '''     EXCLUDE IF STARTS TOO LOW, OR NEVER GOES INSIDE X OF OBSTACLE, OR NEVER GETS CLOSE TO Y=45      '''
                    # if to low or too lateral don't use
                    if path[1][0] > 40 or (not np.sum(abs(path[1] - 45) < 5)) or not (np.sum((abs(path[0] - 50) < 24.5) * (path[1] < 50))):
                        continue
                    if j:
                        if path[1][0] > 27 and (not s == end_idx[j - 1] + 1):
                            # print('block')
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
                    else: thru_center.append(False)

                self.analysis[self.experiment][self.condition]['SR'][self.mouse].append([x_SH, y_SH, thru_center, SH_time, x_edge])

            else:
                self.analysis[self.experiment][self.condition]['end time'][self.mouse].append(np.nan)
                self.analysis[self.experiment][self.condition]['RT'][self.mouse].append(np.nan)
                self.analysis[self.experiment][self.condition]['edginess'][self.mouse].append(np.nan)
                self.analysis[self.experiment][self.condition]['optimal path length'][self.mouse].append(np.nan)
                self.analysis[self.experiment][self.condition]['optimal RT path length'][self.mouse].append(np.nan)
                self.analysis[self.experiment][self.condition]['full path length'][self.mouse].append(np.nan)
                self.analysis[self.experiment][self.condition]['RT path length'][self.mouse].append(np.nan)
                self.analysis[self.experiment][self.condition]['SR'][self.mouse].append([np.nan, np.nan, np.nan, np.nan, np.nan])





        pass
    def analyze_comparisons(self, coordinates):
        '''     analyze other quantities         '''
        pass

    def load_coordinates(self, dataframe, session_name):
        '''     load the coordinates file       '''
        # get and report the session and metadata
        session = dataframe.db.loc[session_name];
        metadata = session.Metadata
        print('Analyzing: {} - {} (#{})'.format(metadata['experiment'], metadata['mouse_id'], metadata['number']))
        # find file path for video and saved coordinates
        video_path = self.session['Metadata']['video_file_paths'][self.vid_num][0]
        processed_coordinates_file = os.path.join(os.path.dirname(video_path), 'coordinates')
        # open saved coordinates
        with open(processed_coordinates_file, "rb") as dill_file: coordinates = pickle.load(dill_file)
        return coordinates

    def format_analysis_dict(self, i):
        '''     make sure all the quantities and conditions are listed in the dictionary        '''
        # initialize arena data
        get_arena_details(self)
        # experiment is there
        if not self.experiment in self.analysis:
            self.analysis[self.experiment] = {}
        # conditions are there
        for condition in self.conditions:
            if not condition in self.analysis[self.experiment]:
                self.analysis[self.experiment][condition] = {}
                # add the shape and obstacle type of the arena
                shape = tuple(self.session['Registration'][4])
                self.analysis[self.experiment][condition]['shape'] = shape
                self.analysis[self.experiment][condition]['type'] = self.obstacle_type
            # quantities are there
            for q in self.quantities_to_analyze:
                if not q in self.analysis[self.experiment][condition]:
                    self.analysis[self.experiment][condition][q] = {}
                if not i or not self.mouse == 'control' or not 'control' in self.analysis[self.experiment][condition][q]:
                    self.analysis[self.experiment][condition][q][self.mouse] = []

    def get_stim_idx(self):
        '''     get indices when the stimulus is or was just on     '''
        stims_video = []
        # loop over each possible stimulus type
        for stim_type, stims_all in self.session['Stimuli']['stimuli'].items():
            # Only use stimulus modalities that were used during the session
            if not stims_all[0]: continue
            # Add to object
            self.stims_all = stims_all
        # initialize list of stim indices
        self.stim_idx = [[] for x in range(len(self.stims_all))]
        # add each stim time and the following 10 seconds to the list
        for vid_num in range(len(self.stims_all)):
            for stim_frame in self.stims_all[vid_num]:
                self.stim_idx[vid_num] = np.append(self.stim_idx[vid_num], np.arange(stim_frame, stim_frame + 240))

    def get_control_stim_frames(self, coordinates):
        '''     get frames when a stim didn't happen but could have         '''
        # initialize values
        stims_video = self.stims_all[self.vid_num]
        y_position = coordinates['center_location'][1].copy()
        stim_idx = np.array([], 'int')
        # loop over actual stims, making those times ineligible
        for stim in stims_video:
            stim_idx = np.append(stim_idx, np.arange(stim - int(self.fps*18), stim + int(self.fps*18)))
        y_position[stim_idx] = np.nan; y_position[:int(self.fps*18)] = np.nan; y_position[-int(self.fps*18):] = np.nan
        # must be in threat zone
        eligible_frames = y_position < (25 / self.scaling_factor)
        # create fake stim times
        stims_video_control = []
        for i, stim in enumerate(stims_video):
            control_stim = np.random.choice(np.where(eligible_frames)[0])
            stims_video_control.append(control_stim)
            eligible_frames[control_stim - int(self.fps*6): control_stim + int(self.fps*6)] = False
            if not np.sum(eligible_frames): break

        stims_video_control.sort()
        # replace real values with fake stims
        self.stims_all[self.vid_num] = stims_video_control

    def get_obstacle_epochs(self, total_frames):
        '''     get time periods when the obstacle is up or down respectively       '''
        # loop across videos
        trial_types = self.session['Tracking']['Trial Types'][self.vid_num]
        if not trial_types: trial_types = [np.nan]
        # wall down experiments
        if -1 in trial_types:
            # when did the wall fall
            wall_down_idx = np.where(np.array(trial_types)==-1)[0][0]
            # when the wall was up
            self.wall_up_epoch = list(range(0, self.stims_all[self.vid_num][wall_down_idx]))
            # when the wall was down
            self.wall_down_epoch = list(range(self.stims_all[self.vid_num][wall_down_idx] + 300, total_frames))
            # and no probe trials
            self.probe_epoch = list(range(self.stims_all[self.vid_num][wall_down_idx], self.stims_all[self.vid_num][wall_down_idx] + 300))
            # when the epoch starts
            self.start_points = [0, self.stims_all[self.vid_num][wall_down_idx] + 300, self.stims_all[self.vid_num][wall_down_idx]]
        # wall up experiments
        elif 1 in trial_types:
            # when did the wall rise
            wall_up_idx = np.where(np.array(trial_types)==1)[0][0]
            # when the wall was down
            self.wall_down_epoch = list(range(0,self.stims_all[self.vid_num][wall_up_idx]))
            # when the wall was up
            self.wall_up_epoch = list(range(self.stims_all[self.vid_num][wall_up_idx] + 300, total_frames))
            # and no probe trials
            self.probe_epoch = list(range(self.stims_all[self.vid_num][wall_up_idx], self.stims_all[self.vid_num][wall_up_idx] + 300))
            # when the epoch starts
            self.start_points = [self.stims_all[self.vid_num][wall_up_idx] + 300, 0, self.stims_all[self.vid_num][wall_up_idx]]
        # void up experiments
        elif trial_types[0]==2 and trial_types[-1] == 0:
            # when did the void rise
            last_wall_idx = np.where([t==2 for t in trial_types])[0][-1]
            first_no_wall_idx = np.where([t==0 for t in trial_types])[0][0]
            # when the void was there
            self.wall_up_epoch = list(range(0,self.stims_all[self.vid_num][last_wall_idx]+300))
            # when the void was filled
            self.wall_down_epoch = list(range(self.stims_all[self.vid_num][first_no_wall_idx] - 300, total_frames))
            # and no probe trials
            self.probe_epoch = []
            # when the epoch starts
            self.start_points = [0, self.stims_all[self.vid_num][first_no_wall_idx] - 300, np.nan]
        # lights on off baseline experiments
        elif 'lights on off (baseline)' in self.session['Metadata']['experiment']:
            if len(self.stims_all[self.vid_num]) > 4:
                # when the light was there
                self.wall_up_epoch = list(range(0,self.stims_all[self.vid_num][4]-300))
                # when the dark was there
                self.wall_down_epoch = list(range(self.stims_all[self.vid_num][4] - 10, total_frames))
                # and no probe trials
                self.probe_epoch = []
                # when the epoch starts
                self.start_points = [0, self.stims_all[self.vid_num][4] - 300, np.nan]
            else:
                # when the light was there
                self.wall_up_epoch = list(range(0,total_frames))
                # when the dark was there
                self.wall_down_epoch = []
                # and no probe trials
                self.probe_epoch = []
                # when the epoch starts
                self.start_points = [0, np.nan, np.nan]
        # obstacle static experiments
        else:
            # the wall was always up
            self.wall_up_epoch = list(range(0, total_frames))
            # and never down
            self.wall_down_epoch = []
            # and no probe trials
            self.probe_epoch = []
            # when the epoch starts
            self.start_points = [0, np.nan, np.nan]

    def get_distance_map(self):
        '''     get the map of geodesic distance to shelter, to compute geodesic speed      '''
        # initialize the arena data
        shape = self.analysis[self.experiment][self.condition]['shape']
        arena, _, _ = model_arena(shape, 2, False, self.obstacle_type, simulate=True)
        shelter_location = [int(a * b / 1000) for a, b in zip(self.shelter_location, arena.shape)]
        # initialize the geodesic function
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
        # get the distance map for the probe trial
        trial_types = self.session['Tracking']['Trial Types'][self.vid_num]
        if -1 in trial_types: distance_map['probe'] = distance_map['no obstacle']
        elif 1 in trial_types: distance_map['probe'] = distance_map['obstacle']
        else: distance_map['probe'] = np.nan

        return distance_map




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
                self.analysis = pickle.load(dill_file)
            break
        except:
            print('file in use...')
            time.sleep(5)
            # TEMPORARY
            self.analysis = {}
            break


    if not experiment in self.analysis:
        self.analysis[experiment] = {}
        self.analysis[experiment]['obstacle'] = {}
        self.analysis[experiment]['no obstacle'] = {}

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


    self.analysis[experiment]['obstacle']['shape'] = (height, width)
    self.analysis[experiment]['obstacle']['type'] = obstacle_type
    self.analysis[experiment]['obstacle']['scale'] = 1
    self.analysis[experiment]['obstacle']['direction scale'] = 8





    # TEMPORARY
    if True:
        '''
        EXPLORATION HEAT MAP
        '''
        if not 'exploration' in self.analysis[experiment]['obstacle']:
            self.analysis[experiment]['obstacle']['exploration'] = {}
            self.analysis[experiment]['no obstacle']['exploration'] = {}

        # get exploration heat map for each epoch
        for i, epoch in enumerate([wall_up_epoch, wall_down_epoch]):
            if not epoch:
                continue

            if 'no shelter' in experiment and 'down' in experiment and i == 0:
                epoch = list(range(0,stims_video[wall_down_idx] - 6*30*60))

            # Histogram of positions
            scale = self.analysis[experiment]['obstacle']['scale']
            H, x_bins, y_bins = np.histogram2d(position[0, epoch], position[1, epoch], [np.arange(0, width + 1, scale),
                                                                            np.arange(0, height + 1, scale)], normed=True)
            H = H.T
            # H[H > 0] = 1

            # make into uint8 image
            H_image = (H * 255 / np.max(H)).astype(np.uint8)
            # H_image = cv2.resize(H_image, (width, height), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('heat map', H_image)

            # put into dictionary
            if i==0: self.analysis[experiment]['obstacle']['exploration'][mouse] = H_image
            elif i==1: self.analysis[experiment]['no obstacle']['exploration'][mouse] = H_image


        '''
        EXPLORATION DIRECTIONALITY
        '''
        if not 'direction' in self.analysis[experiment]['obstacle']:
            self.analysis[experiment]['obstacle']['direction'] = {}
            self.analysis[experiment]['no obstacle']['direction'] = {}

        # get exploration heat map for each epoch
        for i, epoch in enumerate([wall_up_epoch, wall_down_epoch]):
            if not epoch:
                continue

            cur_angles = angles[epoch]

            # Histogram of positions
            scale = int(height / self.analysis[experiment]['obstacle']['direction scale'])
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
            if i==0: self.analysis[experiment]['obstacle']['direction'][mouse] = successor_representation
            elif i==1: self.analysis[experiment]['no obstacle']['direction'][mouse] = successor_representation

        '''
        get the movements toward the center
        '''



            # y_location = coordinates['center_location'][1][epoch] * scaling_factor
            # x_location = coordinates['center_location'][0][epoch] * scaling_factor
            # speed = coordinates['speed'][epoch] * scaling_factor * 30
            # 
            # # indices of being in different sections: modified due to systematic bias in registering void vs wall expts
            # # improve registration -> remove difference in values between experiments
            # middle_idx = (y_location > (40 - 5*('void' in experiment))) * (y_location < (57 + 3*('void' in experiment)))  # 40, 60
            # # middle_idx = (y_location > 40) * (y_location < 60)  # 40, 60
            # back = 25 - 5*('void' in experiment)
            # front = 75
            # back_idx = np.where(y_location < back)[0]
            # front_idx = np.where(y_location > front)[0]
            # time_chase = 25
            # 
            # idx = 0
            # crossing_from_front = []; start_in_front = []; front_speed = []
            # crossing_from_back = []; start_in_back = []; back_speed = []
            # 
            # for k, g in itertools.groupby(middle_idx):
            #     frames_grouped = list(g)
            #     # if idx in stim_idx: print(y_location[idx])
            #     if idx and k and len(frames_grouped) > 3:
            #         if y_location[idx] < 50 and np.sum(y_location[idx - time_chase:idx] < back) and (not idx in stim_idx):
            #             crossing_from_back.append(x_location[idx])
            #             leave_back_idx = np.max(back_idx[back_idx < idx])
            #             start_in_back.append(x_location[leave_back_idx])
            #             back_speed.append(np.mean(speed[leave_back_idx:idx]))
            #         elif y_location[idx] > 50 and np.sum(y_location[idx - time_chase:idx] > (front)):
            #             crossing_from_front.append(x_location[idx])
            #             leave_front_idx = np.max(front_idx[front_idx < idx])
            #             start_in_front.append(x_location[leave_front_idx])
            #             front_speed.append(np.mean(speed[leave_front_idx:idx]))
            # 
            #     idx += len(frames_grouped)


    '''
    ABSOLUTE AND GEODESIC SPEED TRACES
    '''
    if False:
        quantities = ['speed', 'geo speed', 'HD', 'escape', 'time', 'optimal path length',
                      'actual path length', 'path', 'RT', 'SR', 'IOM', 'lunge']

        for condition in ['obstacle', 'no obstacle']:
            for q in quantities:
                if not q in self.analysis[experiment][condition]:
                    self.analysis[experiment][condition][q] = {}

                if q == 'time': self.analysis[experiment][condition][q][mouse] = [[],[]]
                else: self.analysis[experiment][condition][q][mouse] = []


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
            self.analysis[experiment][condition]['time'][mouse][0].append(stim / self.fps / 60)

            # get the speed
            self.analysis[experiment][condition]['speed'][mouse].append(list(speed[threat_idx]))

            # get the geodesic speed
            threat_idx_mod = np.concatenate((np.ones(1, int) * threat_idx[0] - 1, threat_idx))
            threat_position = position[0][threat_idx_mod].astype(int), position[1][threat_idx_mod].astype(int)
            geo_location = distance_map[condition][threat_position[1], threat_position[0]]
            geo_speed = np.diff(geo_location)
            self.analysis[experiment][condition]['geo speed'][mouse].append(list(geo_speed))

            # get the HD rel to the HV
            self.analysis[experiment][condition]['HD'][mouse].append(list(shelter_angles[threat_idx]))

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

            self.analysis[experiment][condition]['SR'][mouse].append( [x_SH, y_SH, thru_center, SH_time] )


            '''
            get the reaction time
            '''
            scaling_factor = 100 / height
            trial_subgoal_speed = [s * scaling_factor * 30 for s in geo_speed]
            subgoal_speed_trace = gaussian_filter1d(trial_subgoal_speed, 4) #2
            initial_speed = np.where(-subgoal_speed_trace[4*30:] > 15)[0]

            if arrived_at_shelter.size and initial_speed.size:
                self.analysis[experiment][condition]['escape'][mouse].append(True)
                self.analysis[experiment][condition]['time'][mouse][1].append(arrived_at_shelter[0])
                self.analysis[experiment][condition]['path'][mouse].append(
                        (position[0][stim:stim+arrived_at_shelter[0]], position[1][stim:stim+arrived_at_shelter[0]]))

                RT = initial_speed[0] / 30
                self.analysis[experiment][condition]['RT'][mouse].append(RT)

                # get the start position
                start_position = int(position[0][stim+int(RT*30)]), int(position[1][stim+int(RT*30)])

                # get the optimal path length
                optimal_path_length = distance_map[condition][start_position[1], start_position[0]]
                self.analysis[experiment][condition]['optimal path length'][mouse].append(optimal_path_length)

                # get the actual path length
                full_path_length = np.sum(speed[stim:stim + arrived_at_shelter[0]])
                RT_path_length = np.sum(speed[stim+int(RT*30):stim+arrived_at_shelter[0]])
                self.analysis[experiment][condition]['full path length'][mouse].append(actual_path_length+60)
                self.analysis[experiment][condition]['RT path length'][mouse].append(actual_path_length+60)


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


                try: self.analysis[experiment][condition]['lunge'][mouse].append([bout_start_position, bout_end_position])
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

                try: self.analysis[experiment][condition]['IOM'][mouse].append([bout_start_position, bout_end_position])
                except: pass
                # print([bout_start_position, bout_end_position])


            else:
                self.analysis[experiment][condition]['escape'][mouse].append(False)
                self.analysis[experiment][condition]['time'][mouse][1].append(np.nan)
                self.analysis[experiment][condition]['path'][mouse].append(np.nan)
                self.analysis[experiment][condition]['optimal path length'][mouse].append(np.nan)
                self.analysis[experiment][condition]['actual path length'][mouse].append(np.nan)
                self.analysis[experiment][condition]['RT'][mouse].append(np.nan)
                self.analysis[experiment][condition]['IOM'][mouse].append(np.nan)
                self.analysis[experiment][condition]['lunge'][mouse].append(np.nan)

            if 'control' in experiment:
                self.analysis[experiment][condition]['escape'][mouse][-1] = None
                self.analysis[experiment][condition]['path'][mouse][-1] = \
                    (position[0][stim:stim+300], position[1][stim:stim+300])
                self.analysis[experiment][condition]['speed'][mouse][-1] = speed[stim:stim + 300]






    '''
    SAVE RESULTS
    '''
    # save the dictionary
    while True:
        try:
            with open(save_file, "wb") as dill_file:
                pickle.dump(self.analysis, dill_file)
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
