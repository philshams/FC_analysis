from Utils.imports import *


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('axes', edgecolor=[0.8, 0.8, 0.8])
matplotlib.rcParams['text.color'] = [0.8, 0.8, 0.8]
matplotlib.rcParams['axes.labelcolor'] = [0.8, 0.8, 0.8]
matplotlib.rcParams['axes.labelcolor'] = [0.8, 0.8, 0.8]
matplotlib.rcParams['xtick.color'] = [0.8, 0.8, 0.8]
matplotlib.rcParams['ytick.color'] = [0.8, 0.8, 0.8]
params = {'legend.fontsize': 16,
          'legend.handlelength': 2}
plt.rcParams.update(params)


import matplotlib.path as mplPath
import math

from Plotting.Plotting_utils import make_legend

from Utils.Messaging import slack_chat_messenger
from Utils.decorators import clock

from Config import cohort_options


class FlipFlop:
    # TODO look at the two explorations, find a way to track the second exploration too
    def __init__(self, session):
        self.name = 'FlipFlop Maze'
        sess_outcomes = session.Metadata.processing['Session outcomes']
        perconfig = sess_outcomes['outcomes_perconfig']

        numr = perconfig['Right']['origins'].numorigins
        numr_left = perconfig['Right']['origins'].numleftorigins
        numrors, numrescs = perconfig['Right']['origins'].numorigins - numr_left, perconfig['Right']['escapes'].numescapes - numr_left
        if numr: probr = numrescs/ numr
        else: probr = 0

        numl = perconfig['Left']['origins'].numorigins
        numl_left = perconfig['Left']['origins'].numleftorigins
        numlors, numlescs = perconfig['Left']['origins'].numorigins - numl_left, perconfig['Left']['escapes'].numescapes - numl_left
        if numl: probl = numlescs/numl
        else: probl = 0

        print(""" Session {}
        For Right configuration:
            {} trials
            {} R origns
            {} R escapes
            {} R escape probability
        
        for Left configuration:
            {} trials
            {} R origins
            {} R escapes
            {} R escape probability
           
        """.format(session.name, numr, numrors, numrescs, probr, numl, numlors, numlescs, probl))

        def name(self):
            return 'FlipFlop Maze'

    def __repr__(self):
        return 'FlipFlop Maze'
    def __str__(self):
        return 'FlipFlop Maze'



experiment_specific_classes = [FlipFlop]

class mazeprocessor:
    """
    * analyse exploration: - define exploration
                           - quantify different aspects of explorations
    * Analyse individual trials: - get outcome
                                 - get stuff
    """

    def __init__(self, session, settings=None, debugging=False):
        print(colored('      ... maze specific processing session: {}'.format(session.name), 'green'))
        self.session = session
        self.settings = settings
        self.debug_on = debugging
        self.exp_spec_classes = experiment_specific_classes

        self.xyv_trace_tup = namedtuple('trace', 'x y velocity')

        # Get maze structure
        self.maze_rois = self.get_maze_components()

        # Analyse exploration  # TODO make these two functions happen in parallel to speed things up
        self.exploration_processer()

        # Analyse individual trials
        self.trial_processor()

        # Analyse the whole session
        # self.session_processor()

        # Do experiment specfic analysis
        # self.experiment_processor()

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

        return results

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

        rois_results = self.get_timeinrois_stats(expl)
        rois_results['all rois'] = self.maze_rois
        cls_exp = Exploration()
        cls_exp.metadata = self.session.Metadata
        cls_exp.tracking = expl
        cls_exp.processing['ROI analysis'] = rois_results
        self.session.Tracking['Exploration processed'] = cls_exp

    # TRIAL PROCESSOR ##################################################################################################
    def trial_processor(self):
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
                        # self.get_status_at_timepoint(data)  # get status at stim

                        """ functions to write
                        get hesitations at T platform --> quantify head ang acc 
                        get status at relevant timepoints
                        """

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

        timelimit = 30  # number of seconds within which the S platf must be reached to consider the trial succesf.
        timelimit_frame = timelimit * self.session.Metadata.videodata[0]['Frame rate'][0]

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
                # Copmlete escape
                results['trial_outcome'] = True  # True is a complete escape
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

    def get_status_at_timepoint(self, data, time: int = None, timename: str = 'stimulus'): # TODO add platform to status
        """
        Get the status of the mouse [location, orientation...] at a specific timepoint.
        If not time is give the midpoint of the tracking traces is taken as stim time
        """
        if not 'session' in data.name.lower() or 'exploration' in name.lower:
            if data.dlc_tracking.keys():
                if time is None:  # if a time is not give take the midpoint
                    time = data.processing['Trial outcome']['stim_time']

                # Create a named tuple with all the params from processing (e.g. head angle) and the selected time point
                params = data.dlc_tracking['Posture']['body'].keys()
                params = [x.replace(' ', '') for x in params]
                params = namedtuple('params', list(params))
                values = data.dlc_tracking['Posture']['body'].values[time]
                status = params(*values)

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

    # SESSION PROCESSOR ################################################################################################
    def session_processor(self):
        """ To do:
            - Get stats relative to arms of origin and arms of escapes
            - Get stats of maze rois for the whole session (when available) and just the exploration 
        
            - Get probabilities: 
                                * Escape probability (escape, vs incomplete escape, vs no escape)
                                * Escape arm as a function of: X pos, Y pos, arm of origin, prev trial             
        """
        tracking_items = self.session.Tracking.keys()
        maze_configs, origins, escapes = [], [], []
        if tracking_items:
            for trial_name in sorted(tracking_items):
                if 'whole' not in trial_name.lower() and 'exploration' not in trial_name.lower():
                    # extract results of processing steps
                    data = self.session.Tracking[trial_name]

                    maze_configs.append(data.processing['Trial outcome']['maze_configuration'])
                    origins.append(data.processing['Trial outcome']['threat_origin_arm'])
                    escapes.append(data.processing['Trial outcome']['threat_escape_arm'])

        left_origins = len([b for b in origins if 'Left' in b])
        left_escapes = len([b for b in escapes if 'Left' in b])

        if len(set(maze_configs)) > 1:
            outcomes_perconfig = {name:None for name in set(maze_configs)}
            ors = namedtuple('origins', 'numorigins numleftorigins')
            escs = namedtuple('escapes', 'numescapes numleftescapes')
            for conf in set(maze_configs):
                origins_conf = ors(len([o for i,o in enumerate(origins) if maze_configs[i] == conf]),
                          len([o for i, o in enumerate(origins) if maze_configs[i] == conf and 'Left' in origins[i]]))
                escapes_conf = escs(len([o for i, o in enumerate(escapes) if maze_configs[i] == conf]),
                          len([o for i, o in enumerate(escapes) if maze_configs[i] == conf and 'Left' in escapes[i]]))
                if len([c for c in maze_configs if c == conf]) < origins_conf.numorigins:
                    raise Warning('Something went wrong, dammit')
                outcomes_perconfig[conf] = dict(origins=origins_conf, escapes=escapes_conf)
        else: outcomes_perconfig = None

        if not 'processing' in self.session.Metadata.__dict__.keys():
            setattr(self.session.Metadata, 'processing', {})
        self.session.Metadata.processing['Session outcomes'] = dict(origins=origins, escapes=escapes,
                                                                    outcomes_perconfig=outcomes_perconfig)

    # experiment PROCESSOR #############################################################################################
    def experiment_processor(self):
        exp_name = self.session.Metadata.experiment
        session = self.session
        if not exp_name in [cls.__repr__(self) for cls in self.exp_spec_classes]:
            return
        else:
            try:
                idx =  [cls.__repr__(self) for cls in self.exp_spec_classes].index(exp_name)
                exp = self.exp_spec_classes[idx]
                exp(session)
            except:
                raise Warning('No class was made to analyse experiment {}'.format(exp_name))


class mazecohortprocessor:
    def __init__(self, cohort):
        self.colors = dict(left=[.2, .3, .7], right=[.7, .3, .2], centre=[.3, .7, .2],
                           shelter='c', threat='y')

        name = cohort_options['name']
        print(colored('Maze processing cohort {}'.format(name), 'green'))

        metad =  cohort.Metadata[name]
        tracking_data = cohort.Tracking[name]

        self.process_explorations(tracking_data)
        self.process_trials(tracking_data)



    def process_explorations(self, tracking_data):
        all_platforms = ['Left_far_platform', 'Left_medium_platform', 'Right_medium_platform',
                         'Right_far_platform', 'Shelter_platform', 'Threat_platform']

        all_transitions, times = {name:[] for name in all_platforms}, {name:[] for name in all_platforms}
        fps = None
        for exp in tracking_data.explorations:
            if not isinstance(exp, tuple):
                print(exp.metadata)
                if fps is None: fps = exp.metadata.videodata[0]['Frame rate'][0]
                exp_platforms = [p for p in exp.processing['ROI analysis']['all rois'].keys() if 'platform' in p]

                for p in exp_platforms:
                    if p not in all_platforms: raise Warning()

                timeinroi = {name:val for name,val in exp.processing['ROI analysis']['time_in_rois'].items()
                            if 'platform' in name}
                transitions = {name:val for name,val in exp.processing['ROI analysis']['transitions_count'].items()
                            if 'platform' in name}

                for name in all_platforms:
                    if name in transitions: all_transitions[name].append(transitions[name])
                    if name in timeinroi: times[name].append(timeinroi[name])

        all_platforms = ['Left_far_platform', 'Left_medium_platform', 'Shelter_platform', 'Threat_platform',
                         'Right_medium_platform',  'Right_far_platform', ]

        f, axarr = plt.subplots(2, 1, facecolor=[0.1, 0.1, 0.1])
        f.tight_layout()

        timeax, transitionsax = axarr[0], axarr[1]
        for idx, name in enumerate(all_platforms):
            if not name: continue
            type = name.split('_')[0].lower()
            col = self.colors[type]
            if 'far' in name: col = np.add(col, 0.25)

            timeax.bar(idx-.5, np.mean(times[name])/fps, color=col, label=name)
            transitionsax.bar(idx-.5, np.mean(all_transitions[name]), color=col, label=name)


        for ax in axarr:
            ax.axvline(1, color='w')
            ax.axvline(3, color='w')

        timeax.set(title='Seconds per platform', xlim=[-1, 5], ylim=[0, 300], facecolor=[0.2, 0.2, 0.2])
        make_legend(timeax, [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=8)
        transitionsax.set(title='Number of entries per platform', xlim=[-1, 5],  facecolor=[0.2, 0.2, 0.2])

        plt.show()

    def process_trials(self, tracking_data):
        maze_configs, origins, escapes, outcomes = [], [], [], []
        for trial in tracking_data.trials:
            maze_configs.append(trial.processing['Trial outcome']['maze_configuration'])
            origins.append(trial.processing['Trial outcome']['threat_origin_arm'])
            escapes.append(trial.processing['Trial outcome']['threat_escape_arm'])
            outcomes.append(trial.processing['Trial outcome']['trial_outcome'])

        # clean up
        while None in origins:
            idx = origins.index(None)
            origins.pop(idx)
            escapes.pop(idx)
            maze_configs.pop(idx)
            outcomes.pop(idx)

        num_trials = len([t for t in outcomes if t])
        left_origins = len([b for i, b in enumerate(origins) if 'Left' in b and outcomes[i] is not None])
        central_origins = len([b for i, b in enumerate(origins) if 'Central' in b and outcomes[i] is not None])
        left_escapes = len([b for i, b in enumerate(escapes) if 'Left' in b and outcomes[i] is not None])
        central_escapes = len([b for i, b in enumerate(escapes) if 'Central' in b and outcomes[i] is not None])


        if len(set(maze_configs)) > 1:
            outcomes_perconfig = {name:None for name in set(maze_configs)}
            ors = namedtuple('origins', 'numorigins numleftorigins')
            escs = namedtuple('escapes', 'numescapes numleftescapes')
            for conf in set(maze_configs):
                origins_conf = ors(len([o for i,o in enumerate(origins) if maze_configs[i] == conf and outcomes[i]]),
                          len([o for i, o in enumerate(origins) if maze_configs[i] == conf and 'Left' in origins[i]
                               and outcomes[i]]))
                escapes_conf = escs(len([o for i, o in enumerate(escapes) if maze_configs[i] == conf and outcomes[i]]),
                          len([o for i, o in enumerate(escapes) if maze_configs[i] == conf and 'Left' in escapes[i]
                               and outcomes[i]]))
                if len([c for c in maze_configs if c == conf]) < origins_conf.numorigins:
                    raise Warning('Something went wrong, dammit')
                outcomes_perconfig[conf] = dict(origins=origins_conf, escapes=escapes_conf)
        else: outcomes_perconfig = None


        def plotty(ax, numtr, lori, cori, lesc, cesc, title='Origin and escape probabilities'):
            ax.bar(0, lori / numtr, color=[.1, .2, .4], width=.25, label='L ori')
            ax.bar(0.25, cori / numtr, color=[.2, .4, .1], width=.25, label='C ori')
            ax.bar(0.5, 1 - ((lori + cori) / numtr), color=[.4, .2, .1], width=.25, label='R ori')

            ax.axvline(0.75, color=[1, 1, 1])

            ax.bar(1, lesc / numtr, color=[.2, .3, .7], width=.25, label='L escape')
            ax.bar(1.25, cesc / numtr, color=[.3, .7, .2], width=.25, label='C escape')
            ax.bar(1.5, 1 - ((lesc + cesc) / numtr), color=[.7, .3, .2], width=.25, label='R escape')

            ax.set(title='{} - {} trials'.format(title, numtr), ylim=[0, 1], facecolor=[0.2, 0.2, 0.2])
            make_legend(ax, [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=8)


        if outcomes_perconfig is None:
            f, axarr = plt.subplots(1, 1,  facecolor=[0.1, 0.1, 0.1])
        else:
            f, axarr = plt.subplots(3, 1,  facecolor=[0.1, 0.1, 0.1])


        if outcomes_perconfig is not None:
            ax = axarr[0]
            plotty(ax, num_trials, left_origins, central_origins, left_escapes, central_escapes)
        else:
            plotty(axarr, num_trials, left_origins, central_origins, left_escapes, central_escapes)

        if outcomes_perconfig is not None:
            ax = axarr[1]
            o = outcomes_perconfig['Right']
            num_trials = o['escapes'].numescapes
            left_escapes = o['escapes'].numleftescapes
            left_origins = o['origins'].numleftorigins
            plotty(ax, num_trials, left_origins, 0, left_escapes, 0, title='Probs for R config')


            ax = axarr[2]
            o = outcomes_perconfig['Left']
            num_trials = o['escapes'].numescapes
            left_escapes = o['escapes'].numleftescapes
            left_origins = o['origins'].numleftorigins
            plotty(ax, num_trials, left_origins, 0, left_escapes, 0, title='Probs for L config')

        f.tight_layout()
        plt.show()

