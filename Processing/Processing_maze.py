from Utils.imports import *

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('axes', edgecolor=[0.8, 0.8, 0.8])
params = {'legend.fontsize': 16,
          'legend.handlelength': 2}
col = [0, 0, 0]
# col = [.8, .8, .8]
matplotlib.rc('axes', edgecolor=col)
matplotlib.rcParams['text.color'] = col
matplotlib.rcParams['axes.labelcolor'] = col
matplotlib.rcParams['axes.labelcolor'] = col
matplotlib.rcParams['xtick.color'] = col
matplotlib.rcParams['ytick.color'] = col

plt.rcParams.update(params)

import math

arms_colors = dict(left=(255, 0, 0), central=(0, 255, 0), right=(0, 0, 255), shelter=(200, 180, 0),
                        threat=(0, 180, 200))

exp_specifics = {'FlipFlop': flipflop}


class mazeprocessor:
    # TODO session processor
    def __init__(self, session, settings=None, debugging=False):
        self.escape_duration_limit = 9  # number of seconds within which the S platf must be reached to consider the trial succesf.


        # Initialise variables
        print(colored('      ... maze specific processing session: {}'.format(session.name), 'green'))
        self.session = session
        self.settings = settings
        self.debug_on = debugging

        self.colors = arms_colors
        self.xyv_trace_tup = namedtuple('trace', 'x y velocity')

        # Get maze structure
        self.maze_rois = self.get_maze_components()

        # Analyse exploration and trials in parallel
        funcs = [self.exploration_processer, self.trial_processor]
        pool = ThreadPool(len(funcs))
        [pool.apply(func) for func in funcs]

        if settings is not None:
            if settings['apply exp-specific'] and session.Metadata['exp'] in exp_specifics.keys():
                cls = exp_specifics[session.Metadata['exp']]
                cls()

    # UTILS ############################################################################################################
    def get_templates(self):
        # Get the templates
        exp_name = self.session.Metadata.experiment
        base_fld = self.settings['templates folder']
        bg_folder = os.path.join(base_fld, 'Bgs')
        templates_fld = os.path.join(base_fld, exp_name)

        platf_templates = [os.path.join(templates_fld, f) for f in os.listdir(templates_fld) if 'platform' in f]
        bridge_templates = [os.path.join(templates_fld, f) for f in os.listdir(templates_fld) if 'bridge' in f]
        bridge_templates = [b for b in bridge_templates if 'closed' not in b]
        maze_templates = [os.path.join(templates_fld, f) for f in os.listdir(templates_fld) if 'maze_config' in f]
        return base_fld, bg_folder, platf_templates, bridge_templates, maze_templates

    def get_maze_configuration(self, frame):
        """ Uses templates to check in which configuration the maze in at a give time point during an exp  """
        base_fld, _, _, _, maze_templates = self.get_templates()
        maze_templates = [t for t in maze_templates if 'default' not in t]
        maze_templates_dict = {name: cv2.imread(os.path.join(base_fld, name)) for name in maze_templates}

        matches = []
        for name, template in maze_templates_dict.items():
            template = template[1:, 1:]  # the template needs to be smaller than the frame
            res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            matches.append(max_val)
        if not matches: return 'Static'
        best_match = maze_templates[matches.index(max(matches))]
        return os.path.split(best_match)[1].split('__')[0]

    def get_maze_components(self):
        """ Uses template matching to identify the different components of the maze and their location """

        def loop_over_templates(templates, img, bridge_mode=False):
            """ in bridge mode we use the info about the pre-supposed location of the bridge to increase accuracy """
            rois = {}
            point = namedtuple('point', 'topleft bottomright')

            # Set up open CV
            font = cv2.FONT_HERSHEY_SIMPLEX
            if len(img.shape) == 2:  colored_bg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else: colored_bg = img

            # Loop over the templates
            for n, template in enumerate(templates):
                id = os.path.split(template)[1].split('_')[0]
                col = self.colors[id.lower()]
                templ = cv2.imread(template)
                if len(templ.shape) == 3: templ = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)
                w, h = templ.shape[::-1]

                res = cv2.matchTemplate(gray, templ, cv2.TM_CCOEFF_NORMED)  # <-- template matchin here
                rheight, rwidth = res.shape
                if not bridge_mode:  # platforms
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  # location of best template match
                    top_left = max_loc
                else:  # take only the relevant quadrant of the frame based on bridge name
                    if id == 'Left':
                        res = res[:, 0:int(rwidth / 2)]
                        hor_sum = 0
                    elif id == 'Right':
                        res = res[:, int(rwidth / 2):]
                        hor_sum = int(rwidth / 2)
                    else:
                        hor_sum = 0

                    origin = os.path.split(template)[1].split('_')[1][0]
                    if origin == 'T':
                        res = res[int(rheight / 2):, :]
                        ver_sum = int(rheight / 2)
                    else:
                        res = res[:int(rheight / 2):, :]
                        ver_sum = 0

                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  # location of best template match
                    top_left = (max_loc[0] + hor_sum, max_loc[1] + ver_sum)

                # Get location and mark on the frame the position of the template
                bottom_right = (top_left[0] + w, top_left[1] + h)
                midpoint = point(top_left, bottom_right)
                rois[os.path.split(template)[1].split('.')[0]] = midpoint
                cv2.rectangle(colored_bg, top_left, bottom_right, col, 2)
                cv2.putText(colored_bg, os.path.split(template)[1].split('.')[0] + '  {}'.format(round(max_val, 2)),
                            (top_left[0] + 10, top_left[1] + 25),
                            font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            return colored_bg, rois

        # Get bg
        bg = self.session.Metadata.videodata[0]['Background']
        if len(bg.shape) == 3: gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        else: gray = bg

        # get templates
        base_fld, bg_folder, platf_templates, bridge_templates, maze_templates = self.get_templates()
        if maze_templates:
            img = [t for t in maze_templates if 'default' in t]
            bg = cv2.imread(img[0])
            if bg.shape[-1] > 2: bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

        # Store background
        f_name = '{}.png'.format(self.session.name)
        if not f_name in os.listdir(bg_folder):
            cv2.imwrite(os.path.join(bg_folder, f_name), gray)

        # Calculate the position of the templates and save resulting image
        display, platforms = loop_over_templates(platf_templates, bg)
        display, bridges = loop_over_templates(bridge_templates, display, bridge_mode=True)
        cv2.imwrite(os.path.join(base_fld, 'Matched\\{}.png'.format(self.session.name)), display)

        # Return locations of the templates
        dic = {**platforms, **bridges}
        return dic

    def get_roi_at_each_frame(self, data, datatype='namedtuple'):
        # TODO handle incoming data
        if datatype == 'namedtuple':
            data_length = len(data.x)
            pos = np.zeros((data_length, 2))
            pos[:, 0], pos[:, 1] = data.x, data.y

        # Get the center of each roi
        centers, roi_names = [], []
        for name, points in self.maze_rois.items():
            center_x = (points.topleft[0] + points.bottomright[0]) / 2
            center_y = (points.topleft[1] + points.bottomright[1]) / 2
            center = np.asarray([center_x, center_y])
            centers.append(center)
            roi_names.append(name)

        # Calc distance toe ach roi for each frame
        distances = np.zeros((data_length, len(centers)))
        for idx, center in enumerate(centers):
            cnt = np.tile(center, data_length).reshape((data_length, 2))
            dist = np.hypot(np.subtract(cnt[:, 0], pos[:, 0]), np.subtract(cnt[:, 1], pos[:, 1]))
            distances[:, idx] = dist

        # Get which roi the mouse is in at each frame
        sel_rois = np.argmin(distances, 1)
        roi_at_each_frame = tuple([roi_names[x] for x in sel_rois])
        return roi_at_each_frame

    def get_timeinrois_stats(self, data, datatype='namedtuple'):
        """
        Quantify the ammount of time in each maze roi and the avg stay in each roi
        :param datatype: built-in type of data
        :param data: tracking data
        :return: number of frames in each roi, number of seconds in each roi, number of enters in each roi, avg time
                in each roi in frames, avg time in each roi in secs and avg velocity on each roi
        """

        def get_indexes(lst, match):
            return np.asarray([i for i, x in enumerate(lst) if x == match])

        if datatype == 'namedtuple':
            if 'velocity' in data._fields: vel = data.velocity
            else: vel = False
        else:
            vel = False

        # get roi at each frame of data
        data_rois = self.get_roi_at_each_frame(data, datatype=datatype)
        data_time_inrois = {name: data_rois.count(name) for name in set(data_rois)}  # total time (frames) in each roi

        # number of enters in each roi
        transitions = [n for i, n in enumerate(list(data_rois)) if i == 0 or n != list(data_rois)[i - 1]]
        transitions_count = {name: transitions.count(name) for name in transitions}

        # avg time spend in each roi (frames)
        avg_time_in_roi = {transits[0]: time / transits[1]
                           for transits, time in zip(transitions_count.items(), data_time_inrois.values())}

        # convert times to frames
        fps = self.session.Metadata.videodata[0]['Frame rate'][0]
        data_time_inrois_sec = {name: t / fps for name, t in data_time_inrois.items()}
        avg_time_in_roi_sec = {name: t / fps for name, t in avg_time_in_roi.items()}

        # get avg velocity in each roi
        avg_vel_per_roi = {}
        if vel:
            for name in set(data_rois):
                indexes = get_indexes(data_rois, name)
                vels = [vel[x] for x in indexes]
                avg_vel_per_roi[name] = np.average(np.asarray(vels))

        results = dict(time_in_rois = data_time_inrois,
                       time_in_rois_sec = data_time_inrois_sec,
                       transitions_count = transitions_count,
                       avg_time_in_roi = avg_time_in_roi,
                       avg_tiime_in_roi_sec = avg_time_in_roi_sec,
                       avg_vel_per_roi=avg_vel_per_roi)

        return transitions, results

    # EXPLORATION PROCESSOR ############################################################################################
    def exploration_processer(self, expl=None):
        if expl is None:          # Define the exploration phase if none is given
            if 'Exploration' in self.session.Tracking.keys():
                expl = self.session.Tracking['Exploration']
            elif 'Whole Session' in self.session.Tracking.keys():
                whole = self.session.Tracking['Whole Session']

                # find the first stim of the session
                first_stim = 100000
                for stims in self.session.Metadata.stimuli.values():
                    for idx, vid_stims in enumerate(stims):
                        if not vid_stims: continue
                        else:
                            if first_stim < 100000 and idx > 0: break
                            first = vid_stims[0]
                            if first < first_stim: first_stim = first - 1

                # Extract the part of whole session that corresponds to the exploration
                len_first_vid = len(whole[list(whole.keys())[0]].x)
                if len_first_vid < first_stim:
                    if len(whole.keys())> 1:
                        fvideo = whole[list(whole.keys())[0]]
                        svideo = whole[list(whole.keys())[1]]
                        expl = self.xyv_trace_tup(np.concatenate((fvideo.x.values, svideo.x.values)),
                                                  np.concatenate((fvideo.y.values, svideo.y.values)),
                                                  None)
                        if first_stim < len(expl.x):
                            expl = self.xyv_trace_tup(expl.x[0:first_stim], expl.y[0:first_stim], None)
                    else:
                        expl = whole[list(whole.keys())[0]]
                else:
                    vid_names = sorted(list(whole.keys()))
                    if not 'velocity' in whole[list(whole.keys())[0]]._fields:
                        vel = calc_distance_2d((whole[vid_names[0]].x, whole[vid_names[0]].y),
                                               vectors=True)
                    else:
                        vel = whole[list(whole.keys())[0]].vel
                    expl = self.xyv_trace_tup(whole[vid_names[0]].x[0:first_stim],
                                              whole[vid_names[0]].y[0:first_stim],
                                              vel[0:first_stim])
                self.session.Tracking['Exploration'] = expl
            else:
                return False

        expl_roi_transitions, rois_results = self.get_timeinrois_stats(expl)
        rois_results['all rois'] = self.maze_rois
        cls_exp = Exploration()
        cls_exp.metadata = self.session.Metadata
        cls_exp.tracking = expl
        cls_exp.processing['ROI analysis'] = rois_results
        cls_exp.processing['ROI transitions'] = expl_roi_transitions
        self.session.Tracking['Exploration processed'] = cls_exp

    # TRIAL PROCESSOR ##################################################################################################
    def trial_processor(self):
        rois_for_hesitations = ['Threat', 'Left_med', 'Right_med']

        # Loop over each trial
        tracking_items = self.session.Tracking.keys()
        if tracking_items:
            for trial_name in sorted(tracking_items):
                    if 'whole' not in trial_name.lower() and 'exploration' not in trial_name.lower():
                        # Get tracking data and first frame of the trial
                        print('         ... maze specific processing: {}'.format(trial_name))
                        data = self.session.Tracking[trial_name]
                        startf_num = data.metadata['Start frame']
                        videonum = int(trial_name.split('_')[1].split('-')[0])
                        video = self.session.Metadata.video_file_paths[videonum][0]
                        grabber = cv2.VideoCapture(video)
                        ret, frame = grabber.read()

                        maze_configuration = self.get_maze_configuration(frame)

                        self.get_trial_outcome(data, maze_configuration)
                        self.get_status_at_timepoint(data)  # get status at stim

                        vtes = {r: self.get_vte_in_roi(data, roi=r) for r in rois_for_hesitations}
                        self.session.Tracking[trial_name].processing['vtes'] = vtes

    @staticmethod
    def get_trial_length(trial):
        if 'Posture' in trial.dlc_tracking.keys():
            return True, len(trial.dlc_tracking['Posture']['body']['x'])
        else:
            try:
                return True, len(trial.std_tracing.x.values)
            except:
                return False, False

    @staticmethod
    def get_arms_of_originandescape(rois, outward=True): # TODO refactor
        """ if outward get the last shelter roi and first threat otherwise first shelter and last threat in rois
            Also if outward it gets the first arm after shelter and the last arm before threat else vicevers"""
        if outward:
            shelt =[i for i, x in enumerate(rois) if x == "Shelter_platform"]
            if shelt: shelt = shelt[-1]
            else: shelt = 0
            if 'Threat_platform' in rois[shelt:]:
                thrt = rois[shelt:].index('Threat_platform') + shelt
            else:
                thrt = 0
            try:
                shelt_arm = rois[shelt+1]
            except:
                shelt_arm = rois[shelt]
            threat_arm = rois[thrt-1]
        else:
            if 'Shelter_platform' in rois:
                shelt = rois.index('Shelter_platform')
                shelt_arm = rois[shelt - 1]
            else:
                shelt = False
                shelt_arm = False
            thrt = [i for i, x in enumerate(rois[:shelt]) if x == "Threat_platform"]
            if thrt: thrt = thrt[-1]+1
            else: thrt = len(rois)-1
            threat_arm = rois[thrt]
        return shelt, thrt, shelt_arm, threat_arm

    def get_trial_outcome(self, data, maze_config):
        """  Look at what happens after the stim onset:
                 define some criteria (e.g. max duration of escape) and, given the criteria:
             - If the mouse never leaves T in time that a failure
             - If it leaves T but doesnt reach S while respecting the criteria it is an incomplete escape
             - If everything goes okay and it reaches T thats a succesful escape

         a trial is considered succesful if the mouse reaches the shelter platform within 30s from stim onset
         """

        results = dict(
            maze_rois = None,
            maze_configuration=None,
            trial_outcome = None,
            trial_rois_trajectory = None,
            last_at_shelter = None,
            first_at_threat = None,
            shelter_leave_arm = None,
            threat_origin_arm = None,
            first_at_shelter = None,
            last_at_threat = None,
            shelter_rach_arm = None,
            threat_escape_arm = None,
            stim_time = None
        )
        results['maze_configuration'] = maze_config
        results['maze_rois'] = self.maze_rois


        ret, trial_length = self.get_trial_length(data)
        if not ret:
            data.processing['Trial outcome'] = results
            return

        stim_time = math.floor(trial_length/2)  # the stim is thought to happen at the midpoint of the trial
        results['stim_time'] = stim_time

        trial_rois = self.get_roi_at_each_frame(data.dlc_tracking['Posture']['body'])
        results['trial_rois_trajectory'] = trial_rois
        escape_rois = trial_rois[stim_time:]
        origin_rois = trial_rois[:stim_time-1]

        res = self.get_arms_of_originandescape(origin_rois)
        results['last_at_shelter'] = res[0]
        results['first_at_threat'] = res[1]
        results['shelter_leave_arm'] = res[2]
        results['threat_origin_arm'] = res[3]

        # Find the first non Threat roi
        nonthreats = [(idx, name) for idx,name in enumerate(escape_rois) if 'Threat platform' not in name]
        if not nonthreats:
            # No escape
            results['trial_outcome'] = None  # None is a no escape
            return
        else:
            first_nonthreat = (nonthreats[0][0], escape_rois[nonthreats[0][0]])
            if not 'Shelter_platform' in [name for idx, name in nonthreats]:
                # Incomplete escape
                results['trial_outcome'] = False  # False is an incomplete escape
                res =self.get_arms_of_originandescape(escape_rois, outward=False)
            else:
                # Complete escape
                fps = self.session.Metadata.videodata[0]['Frame rate'][0]
                time_to_shelter = escape_rois.index('Shelter_platform')/fps
                if time_to_shelter > self.escape_duration_limit: results['trial_outcome'] = False  # Too slow
                else:  results['trial_outcome'] = True  # True is a complete escape
            res = self.get_arms_of_originandescape(escape_rois, outward=False)
            results['first_at_shelter'] = res[0] + results['stim_time']
            results['last_at_threat'] = res[1] + results['stim_time']
            results['shelter_rach_arm'] = res[2]
            results['threat_escape_arm'] = res[3]

        if not 'processing' in data.__dict__.keys():
            setattr(data, 'processing', {})
        data.processing['Trial outcome'] = results

        # Get hesitations
        """  
             - Get the frames from stim time to after it leaves T (for successful or incomplete trials
             - Calc the integral of the head velocity for those frames 
             - Calc the smoothness of the trajectory and velocity curves
             - Set thresholds for hesitations
        """

        # Get status at time point

    @staticmethod
    def get_status_at_timepoint(data, time: int = None, timename: str = 'stimulus'): # TODO add platform to status
        """
        Get the status of the mouse [location, orientation...] at a specific timepoint.
        If not time is give the midpoint of the tracking traces is taken as stim time
        """
        if not 'session' in data.name.lower() or 'exploration' in name.lower:
            if data.dlc_tracking.keys():
                if time is None:  # if a time is not give take the midpoint
                    time = data.processing['Trial outcome']['stim_time']

                # Create a named tuple with all the params from processing (e.g. head angle) and the selected time point
                status = data.dlc_tracking['Posture']['body'].loc[time]

                # Make named tuple with posture data at timepoint
                posture_names = namedtuple('posture', sorted(list(data.dlc_tracking['Posture'].keys())))
                bodypart = namedtuple('bp', 'x y')
                bodyparts = []
                for bp, vals in sorted(data.dlc_tracking['Posture'].items()):
                    pos = bodypart(vals['x'].values[time], vals['y'].values[time])
                    bodyparts.append(pos)
                posture = posture_names(*bodyparts)

                complete = namedtuple('status', 'posture status')
                complete = complete(posture, status)
                data.processing['status at {}'.format(timename)] = complete
            else:
                data.processing['status at {}'.format(timename)] = None

        # Get relevant timepoints
        """
            Get status at frames where:
            - max velocity
            - max ang vel
            - max head-body ang difference
            - shortest and longest bodylength
        
        """

    @staticmethod
    def get_vte_in_roi(data, roi='Threat', threshold=False):
        """  during an escape, look at the integral of the head's angular
        velocity and compare it to a threshold value """

        stim_time = data.processing['Trial outcome']['stim_time']
        escape_rois_trajectory = data.processing['Trial outcome']['trial_rois_trajectory'][stim_time:]

        roi_indexes = [i for i,r in enumerate(escape_rois_trajectory) if roi in r]
        df = np.diff(roi_indexes)
        if np.any(df > 1):   # only loook at the first time it enters the roi
            stop = np.where(df > 1)[0]
            roi_indexes = roi_indexes[:stop]

        ang_vel = data.tracking['Posture']['body']['Head ang vel'].loc[roi_indexes[0]:roi_indexes[-1]]
        dPhi = np.trapz(ang_vel)

        if threshold:
            if dPhi > threshold: outcome = True
            else: outcome = False
        else: outcome = True

        return dPhi, outcome









