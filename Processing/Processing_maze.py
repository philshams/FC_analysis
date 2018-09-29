import matplotlib.path as mplPath
import math

from Utils.imports import *

from Utils.Messaging import slack_chat_messenger
from Utils.decorators import clock

from Config import cohort_options


experiment_specific_classes = []

class mazeprocessor:
    """
    * analyse exploration: - define exploration
                           - quantify different aspects of explorations
    * Analyse individual trials: - get outcome
                                 - get stuff


    """

    def __init__(self, session, settings=None, debugging=False):
        self.session = session
        self.settings = settings
        self.debug_on = debugging
        self.exp_spec_classes = experiment_specific_classes

        self.xyv_trace_tup = namedtuple('trace', 'x y velocity')

        # Get maze structure
        self.maze_rois = self.get_maze_components()

        # Analyse exploration
        self.exploration_processer()

        # Analyse individual trials
        self.trial_processor()

        # Analyse the whole session
        self.session_processor()

        # Do experiment specfic analysis
        self.experiment_processor()

        pass

    # UTILS ############################################################################################################
    def get_templates(self):
        # Get the templates
        exp_name = self.session.Metadata.experiment
        base_fld = self.settings['templates folder']
        bg_folder = os.path.join(base_fld, 'Bgs')
        templates_fld = os.path.join(base_fld, exp_name)

        platf_templates = [os.path.join(templates_fld, f) for f in os.listdir(templates_fld) if 'platform' in f]
        bridge_templates = [os.path.join(templates_fld, f) for f in os.listdir(templates_fld) if 'bridge' in f]
        maze_templates = [os.path.join(templates_fld, f) for f in os.listdir(templates_fld) if 'maze_config' in f]
        return base_fld, bg_folder, platf_templates, bridge_templates, maze_templates

    def get_maze_configuration(self, frame):
        """ Uses templates to check in which configuration the maze in at a give time point during an exp  """
        base_fld, _, _, _, maze_templates = self.get_templates()
        maze_templates_dict = {name: cv2.imread(os.path.join(base_fld, name)) for name in maze_templates}


        matches = []
        for name, template in maze_templates_dict.items():
            template = template[1:, 1:]  # the template needs to be smaller than the frame
            res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            matches.append(max_val)
        if not matches: return 'Static'
        best_match = maze_templates[matches.index(max(matches))]
        return best_match.split('_')[0]

    def get_maze_components(self):
        """ Uses template matching to identify the different components of the maze and their location """

        def loop_over_templates(templates, img, bridge_mode=False):
            cols = dict(left=(255, 0, 0), central=(0, 255, 0), right=(0, 0, 255), shelter=(200, 180, 0),
                        threat=(0, 180, 200))
            rois = {}
            point = namedtuple('point', 'topleft bottomright')

            font = cv2.FONT_HERSHEY_SIMPLEX
            if len(img.shape) == 2:
                colored_bg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                colored_bg = img

            for n, template in enumerate(templates):
                id = os.path.split(template)[1].split('_')[0]
                col = cols[id.lower()]
                templ = cv2.imread(template)
                if len(templ.shape) == 3:
                    templ = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)
                w, h = templ.shape[::-1]

                res = cv2.matchTemplate(gray, templ, cv2.TM_CCOEFF_NORMED)
                rheight, rwidth = res.shape
                if not bridge_mode:
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    top_left = max_loc
                else:  # take only the relevant quadrant
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

                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    top_left = (max_loc[0] + hor_sum, max_loc[1] + ver_sum)

                bottom_right = (top_left[0] + w, top_left[1] + h)

                midpoint = point(top_left, bottom_right)
                rois[os.path.split(template)[1].split('.')[0]] = midpoint
                cv2.rectangle(colored_bg, top_left, bottom_right, col, 2)
                cv2.putText(colored_bg, os.path.split(template)[1].split('.')[0] + '  {}'.format(round(max_val, 2)),
                            (top_left[0] + 10, top_left[1] + 25),
                            font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            return colored_bg, rois

        bg = self.session.Metadata.videodata[0]['Background']
        if len(bg.shape) == 3:
            gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        else:
            gray = bg

            base_fld, bg_folder, platf_templates, bridge_templates, _ = self.get_templates()

        # Store background
        f_name = '{}.png'.format(self.session.name)
        if not f_name in os.listdir(bg_folder):
            cv2.imwrite(os.path.join(bg_folder, f_name), gray)

        # Calculate the position of the templates and save resulting image
        display, platforms = loop_over_templates(platf_templates, bg)
        display, bridges = loop_over_templates(bridge_templates, display, bridge_mode=True)
        cv2.imwrite(os.path.join(base_fld, 'Matched\\{}.png'.format(self.session.name)), display)

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
            vel = data.velocity
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

        return results

    # EXPLORATION PROCESSOR ############################################################################################
    def exploration_processer(self, expl=None):
        if expl is None:          # Define the exploration phase if none is given
            if 'Exploration' in self.session.Tracking.keys():
                expl = self.session.Tracking['Exploratin']
            elif 'Whole Session' in self.session.Tracking.keys():
                whole = self.session.Tracking['Whole Session']

                # find the first stim of the session
                first_stim = 100000
                for stims in self.session.Metadata.stimuli.values():
                    if isinstance(stims[0], list):
                        stims = stims[0]
                    if not stims:
                        continue
                    elif min(stims) < first_stim:
                        first_stim = min(stims) - 1

                # Extract the part of whole session that corresponds to the exploration
                len_first_vid = len(whole[list(whole.keys())[0]].x)
                if len_first_vid < first_stim:
                    expl = whole[list(whole.keys())[0]]  # TODO stitch together multiple vids
                else:
                    if not 'velocity' in whole[list(whole.keys())[0]]._fields:
                        vel = calc_distance_2d((whole[list(whole.keys())[0]].x, whole[list(whole.keys())[0]].y),
                                               vectors=True)
                    else:
                        vel = whole[list(whole.keys())[0]].vel
                    expl = self.xyv_trace_tup(whole[list(whole.keys())[0]].x[0:first_stim],
                                              whole[list(whole.keys())[0]].y[0:first_stim],
                                              vel[0:first_stim])
                self.session.Tracking['Exploration'] = expl
            else:
                return False

        rois_results = self.get_timeinrois_stats(expl)
        cls_exp = Exploration()
        cls_exp.tracking = expl
        cls_exp.processing['ROI analysis'] = rois_results
        self.session.Tracking['Exploration processed'] = cls_exp

    # TRIAL PROCESSOR ##################################################################################################
    def trial_processor(self):
        # Loop over each trial
        tracking_items = self.session.Tracking.keys()
        if tracking_items:
            for trial_name in tracking_items:
                    if 'whole' not in trial_name.lower() and 'exploration' not in trial_name.lower():
                        # Get tracking data and first frame of the trial
                        print('Processing: {}'.format(trial_name))
                        data = self.session.Tracking[trial_name]
                        startf_num = data.metadata['Start frame']
                        videonum = int(trial_name.split('_')[1].split('-')[0])
                        video = self.session.Metadata.video_file_paths[videonum][0]
                        grabber = cv2.VideoCapture(video)
                        ret, frame = grabber.read()

                        maze_configuration = self.get_maze_configuration(frame)

                        self.get_trial_outcome(data)

                        """ functions to write
                        get tracking trajectory between the two
                        get "maze" trajectory (rois)  +   get origin and escape(s) arms
                        get hesitations at T platform --> quantify head ang acc 
                        get status at stim onset (pose, vel...)
                        get status at relevant timepoints
                        
                        """
    @staticmethod
    def get_trial_length(trial):
        if 'Posture' in trial.dlc_tracking.keys():
            return len(trial.dlc_tracking['Posture']['body']['x'])
        else:
            try:
                return len(trial.std_tracing.x.values)
            except:
                raise ValueError

    @staticmethod
    def get_arms_of_originandescape(rois, outward=True):
        """ if outward get the last shelter roi and first threat otherwise first shelter and last threat in rois
            Also if outward it gets the first arm after shelter and the last arm before threat else vicevers"""
        # TODO arm identification
        if outward:
            shelt =[i for i, x in enumerate(rois) if x == "Shelter_platform"][-1]
            thrt = rois[shelt:].index('Threat_platform') + shelt
            shelt_arm = rois[shelt+1]
            threat_arm = rois[thrt-1]
        else:
            if 'Shelter_platform' in rois:
                shelt = rois.index('Shelter_platform')
                shelt_arm = rois[shelt - 1]
            else:
                shelt = False
                shelt_arm = False
            thrt = [i for i, x in enumerate(rois) if x == "Threat_platform"][-1]
            threat_arm = rois[thrt + 1]
        return shelt, thrt, shelt_arm, threat_arm

    def get_trial_outcome(self, data):
        """  Look at what happens after the stim onset:
                 define some criteria (e.g. max duration of escape) and, given the criteria:
             - If the mouse never leaves T in time that a failure
             - If it leaves T but doesnt reach S while respecting the criteria it is an incomplete escape
             - If everything goes okay and it reaches T thats a succesful escape

         a trial is considered succesful if the mouse reaches the shelter platform within 30s from stim onset
         """

        results = dict(
            trial_outcome = None,
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

        timelimit = 30  # number of seconds within which the S platf must be reached to consider the trial succesf.
        timelimit_frame = timelimit * self.session.Metadata.videodata[0]['Frame rate'][0]

        trial_length = self.get_trial_length(data)
        stim_time = math.floor(trial_length/2)  # the stim is thought to happen at the midpoint of the trial
        results['stim_time'] = stim_time

        trial_rois = self.get_roi_at_each_frame(data.dlc_tracking['Posture']['body'])  # TODO make this work
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
            if not 'Shelter platform' in [name for idx, name in nonthreats]:
                # Incomplete escape
                results['trial_outcome'] = False  # False is an incomplete escape
                res =self.get_arms_of_originandescape(escape_rois, outward=False)
            else:
                # Copmlete escape
                results['trial_outcome'] = True  # True is a complete escape
            res = self.get_arms_of_originandescape(escape_rois, outward=False)
            results['first_at_shelter'] = res[0]
            results['last_at_threat'] = res[1]
            results['shelter_rach_arm'] = res[2]
            results['threat_escape_arm'] = res[3]

        data.processing['Trial outcome'] = results

    def get_STS_trajectory(self, data):
        """
             - Get tracking data for frames between when it leaves the shelter and when it returns to it while also 
                 dividing them based on pre-stim or after stim
             - Do the same but for the maze rois trajectory (store them all together)
             - Get escape and origin arms 
        """
        pass

        # Get hesitations
        """  
             - Get the frames from stim time to after it leaves T (for successful or incomplete trials
             - Calc the integral of the head velocity for those frames 
             - Calc the smoothness of the trajectory and velocity curves
             - Set thresholds for hesitations
        """

        # Get status at time point
        """
            - Given a time point (e.g. stim onset) extract info about the mouse status at that point: location, 
                 velocity, platform...)
        """

        # Get relevant timepoints
        """
            Get status at frames where:
            - max velocity
            - max ang vel
            - max head-body ang difference
            - shortest and longest bodylength
        
        """

    # SESSION PROCESSOR ################################################################################################
    def session_processor(self):
        pass

        """ To do:
            - Get stats relative to arms of origin and arms of escapes
            - Get stats of maze rois for the whole session (when available) and just the exploration 
        
            - Get probabilities: 
                                * Escape probability (escape, vs incomplete escape, vs no escape)
                                * Escape arm as a function of: X pos, Y pos, arm of origin, prev trial
                                
        
        """

    # experiment PROCESSOR #############################################################################################
    def experiment_processor(self):
        exp_name = self.session.Metadata.experiment
        if not exp_name in self.exp_spec_classes:
            return
        else:
            exp = self.exp_spec_classes[self.exp_spec_classes.index(exp_name)]
            exp()
        pass

