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
import scipy.stats
import random
import seaborn as sns
from scipy.stats import norm

from Plotting.Plotting_utils import make_legend, save_all_open_figs
from Utils.Data_rearrange_funcs import flatten_list
from Utils.maths import line_smoother

from Config import cohort_options


arms_colors = dict(left=(255, 0, 0), central=(0, 255, 0), right=(0, 0, 255), shelter=(200, 180, 0),
                        threat=(0, 180, 200))

class mazeprocessor:
    # TODO session processor
    def __init__(self, session, settings=None, debugging=False):
        self.escape_duration_limit = 45  # number of seconds within which the S platf must be reached to consider the trial succesf.


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


class mazecohortprocessor:
    def __init__(self, cohort):

        fig_save_fld = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\Presentations\\181017_graphs'

        self.colors = dict(left=[.2, .3, .7], right=[.7, .3, .2], centre=[.3, .7, .2], center=[.3, .7, .2],
                           central=[.3, .7, .2], shelter='c', threat='y')

        name = cohort_options['name']
        print(colored('Maze processing cohort {}'.format(name), 'green'))

        metad =  cohort.Metadata[name]
        tracking_data = cohort.Tracking[name]

        self.process_trials(tracking_data)

        self.sample_escapes_probabilities(tracking_data)

        # self.plot_velocites_grouped(tracking_data, metad, selector='escape_arm')
        # self.process_status_at_stim(tracking_data)

        # self.process_explorations(tracking_data)

        # self.process_by_mouse(tracking_data)

        # plt.show()

        save_all_open_figs(target_fld=fig_save_fld, name=name, format='svg')
        plt.show()
        a = 1

    def coin_simultor(self, num_samples=24, num_iters=10000):
        probs = []
        for itern in tqdm(range(num_iters)):
            data = [random.randint(0, 1) for x in range(num_samples)]
            prob_one = len([n for n in data if n==1])/len(data)
            probs.append(prob_one)

        f, ax = plt.subplots(1, 1, facecolor=[0.1, 0.1, 0.1])
        ax.set(facecolor=[0.2, 0.2, 0.2], xlim=[0,1], ylim=[0, 4000])
        basecolor = [.3, .3, .3]

        ax.hist(probs, color=(basecolor), bins=75)
        avg = np.mean(probs)
        # ax.axvline(avg, color=basecolor, linewidth=4, linestyle=':')
        print('mean {}'.format(avg))


    def sample_escapes_probabilities(self, tracking_data, num_samples=15, num_iters=10000):
        sides = ['Left', 'Central', 'Right']
        # get escapes
        maze_configs, origins, escapes, outcomes = [], [], [], []
        for trial in tracking_data.trials:
            outcome = trial.processing['Trial outcome']['trial_outcome']
            if not outcome:
                print(trial.name, ' no escape')
                continue
            if trial.processing['Trial outcome']['maze_configuration'] == 'Left':
                print(trial.name, ' left config')
                continue

            maze_configs.append(trial.processing['Trial outcome']['maze_configuration'])
            origins.append(trial.processing['Trial outcome']['threat_origin_arm'])
            escapes.append(trial.processing['Trial outcome']['threat_escape_arm'])
            outcomes.append(outcome)

        num_trials = len(outcomes)
        left_escapes = len([b for b in escapes if 'Left' in b])
        central_escapes = len([b for b in escapes if 'Central' in b ])

        print('\nFound {} trials, escape probabilities: {}-Left, {}-Centre'.format(num_trials,
                                                                             round(left_escapes/num_trials, 2),
                                                                             round(central_escapes/num_trials, 2)))


        if num_samples > num_trials: raise Warning('Too many samples')

        probs = {name:[] for name in sides}
        for iter in tqdm(range(num_iters)):
            sel_trials = random.sample(escapes, num_samples)
            p = 0
            for side in sides:
                probs[side].append(len([b for b in sel_trials if side in b])/num_samples)
                p += len([b for b in sel_trials if side in b])/num_samples
                if p > 1: raise Warning('Something went wrong....')

        f, ax = plt.subplots(1, 1,  facecolor=[0.1, 0.1, 0.1])
        ax.set(facecolor=[0.2, 0.2, 0.2])
        basecolor = [.3, .3, .3]

        for idx, side in enumerate(probs.keys()):
            if not np.any(np.asarray(probs[side])): continue
            # sns.kdeplot(probs[side], bw=0.05, shade=True, color=np.add(basecolor, 0.2*idx), label=side, ax=ax)
            ax.hist(probs[side], color=np.add(basecolor, 0.2*idx), bins=50, label=side)
            # sns.distplot(probs[side], bins=3, label=side, ax=ax)
            # sns.distplot(probs[side], fit=norm, hist=False, kde=False, norm_hist=True, ax=ax)

        for idx, side in enumerate(probs.keys()):
            if not np.any(np.asarray(probs[side])): continue
            avg = np.mean(probs[side])
            ax.axvline(avg, color=np.add(basecolor, 0.2 * idx), linewidth=4, linestyle=':')
            print('side {} mean {}'.format(side,avg))

        make_legend(ax, [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=16)

        a = 1

    def plot_velocites_grouped(self, tracking_data, metadata, selector='exp'):
        # divide cohrt data into groups
        tags = []
        assigned_tags = {}
        for session in metadata.sessions_in_cohort:
            if selector == 'exp':
                tags.append(session[1].experiment)
                assigned_tags[str(session[1].session_id)] = session[1].experiment
            elif selector == 'stim_type':
                tags = ['visual', 'audio']
                break
            elif selector == 'escape_arm':
                tags = ['Left', 'Central', 'Right' ]
                break

        tags = [t for t in set(tags)]
        colors = [ [.2, .2, .4], [.2, .4, .2],  [.4, .4, .2], [.4, .2, .2], [.2, .4, .4]]
        colors = {name:col for name,col in zip(tags, colors)}

        max_counterscounters = {t:0 for t in tags}
        for trial in tracking_data.trials:
            if not trial.processing['Trial outcome']['trial_outcome']: continue
            sessid = trial.name.split('-')[0]
            if selector == 'exp':
                tag = assigned_tags[sessid]
            elif selector == 'stim_type':
                tag = trial.metadata['Stim type']
            elif selector == 'escape_arm':
                tag = trial.processing['Trial outcome']['threat_escape_arm'].split('_')[0]
                if tag not in max_counterscounters.keys():
                    print(trial.name, 'has weird tag: {}'.format(tag))
                    continue
            max_counterscounters[tag] = max_counterscounters[tag] + 1

        f, axarr = plt.subplots(3, 2, facecolor=[0.1, 0.1, 0.1])
        axarr = axarr.flatten()
        # f.tight_layout()

        velocities = {name:np.zeros((3600, max_counterscounters[name])) for name in tags}
        head_velocities = {name:np.zeros((3600, max_counterscounters[name])) for name in tags}
        body_lengths = {name:np.zeros((3600, max_counterscounters[name])) for name in tags}
        head_body_angles = {name:np.zeros((3600, max_counterscounters[name])) for name in tags}

        counters =  {t:0 for t in tags}
        for trial in tracking_data.trials:
            if not trial.processing['Trial outcome']['trial_outcome'] or trial.processing['Trial outcome']['trial_outcome'] is None :
                print(trial.name, 'no escape')
                continue
            sessid = trial.name.split('-')[0]
            if selector == 'exp': tag = assigned_tags[sessid]
            elif selector == 'stim_type': tag = trial.metadata['Stim type']
            elif selector == 'escape_arm': tag = trial.processing['Trial outcome']['threat_escape_arm'].split('_')[0]

            if tag not in max_counterscounters.keys(): continue

            vel = line_smoother(trial.dlc_tracking['Posture']['body']['Velocity'].values)
            acc = line_smoother(np.diff(vel))
            head_ang_vel = np.abs(line_smoother(trial.dlc_tracking['Posture']['body']['Head ang vel'].values))
            body_len = line_smoother(trial.dlc_tracking['Posture']['body']['Body length'].values)

            pre_stim_mean_bl = np.mean(body_len[(1800-121):1799])
            body_len = np.divide(body_len, pre_stim_mean_bl)

            body_ang =  line_smoother(trial.dlc_tracking['Posture']['body']['Orientation'].values)
            head_ang = line_smoother(trial.dlc_tracking['Posture']['body']['Head angle'].values)
            while np.any(body_ang[body_ang>360]): body_ang[body_ang>360] -= 360
            while np.any(head_ang[head_ang>360]): head_ang[head_ang>360] -= 360
            hb_angle = np.abs(line_smoother(np.diff(np.radians(body_ang)-np.radians(head_ang))))

            axarr[0].plot(vel, color=colors[tag], alpha=0.15)
            axarr[2].plot(head_ang_vel, color=colors[tag], alpha=0.15)
            axarr[1].plot(vel, color=colors[tag], alpha=0.15)
            axarr[4].plot(body_len, color=colors[tag], alpha=0.15)
            axarr[5].plot(hb_angle, color=colors[tag], alpha=0.15)

            v, a, b, h = np.zeros(3600), np.zeros(3600), np.zeros(3600), np.zeros(3600)
            v[:len(vel)] = vel
            a[:len(vel)] = head_ang_vel
            b[:len(vel)] = body_len
            h[:len(hb_angle)] = hb_angle
            vel, head_ang_vel, body_len, hb_angle = v, a, b, h

            velocities[tag][:, counters[tag]] = vel
            head_velocities[tag][:, counters[tag]] = head_ang_vel
            body_lengths[tag][:, counters[tag]] = body_len
            head_body_angles[tag][:, counters[tag]] = hb_angle
            counters[tag] = counters[tag] + 1

        medians = []
        for name, val in velocities.items():
            if name == 'PathInt2': lg = '2Arms Maze'
            elif name == 'PathInt': lg = '3Arms Maze'
            else: lg = name
            avg_vel = np.mean(val, axis=1)
            sem_avg_vel = np.std(val, axis=1) / math.sqrt(max_counterscounters[name])
            median_vel = np.median(val, axis=1)
            medians.append(median_vel)
            try:
                low_perc = np.percentile(val, 25, axis=1)
                high_perc = np.percentile(val, 75, axis=1)
            except:
                 a = 2
            axarr[0].errorbar(x=np.linspace(0, len(avg_vel), len(avg_vel)), y=avg_vel, yerr=sem_avg_vel,
                              color=np.add(colors[name], 0.3), alpha=0.65,
                          linewidth=4, label=lg)
            axarr[1].plot(median_vel, color=np.add(colors[name], 0.3), alpha=0.65, linewidth=5, label=lg)
            # axarr[1].plot(low_perc, color=np.subtract(colors[name], 0.3), alpha=0.65, linewidth=2, label=None)
            # axarr[1].plot(high_perc, color=np.subtract(colors[name], 0.3), alpha=0.65, linewidth=2, label=None)

        for name, val in head_velocities.items():
            if name == 'PathInt2': lg = '2Arms Maze'
            elif name == 'PathInt': lg = '3Arms Maze'
            else: lg = name

            avg_vel = np.median(val, axis=1)
            sem_avg_vel = np.std(val, axis=1) / math.sqrt(max_counterscounters[name])
            axarr[2].errorbar(x=np.linspace(0, len(avg_vel), len(avg_vel)), y=avg_vel, yerr=sem_avg_vel,
                              color=np.add(colors[name], 0.3), alpha=0.65,
                              linewidth=4, label=lg)

        for name, val in body_lengths.items():
            if name == 'PathInt2':   lg = '2Arms Maze'
            elif name == 'PathInt': lg = '3Arms Maze'
            else:  lg = name

            avg_bl = np.median(val, axis=1)
            sem_avg_bl = np.std(val, axis=1) / math.sqrt(max_counterscounters[name])
            axarr[4].errorbar(x=np.linspace(0, len(avg_bl), len(avg_bl)), y=avg_bl, yerr=sem_avg_bl,
                              color=np.add(colors[name], 0.3), alpha=0.65,
                              linewidth=4, label=lg)

        for name, val in head_body_angles.items():
            if name == 'PathInt2': lg = '2Arms Maze'
            elif name == 'PathInt':  lg = '3Arms Maze'
            else:   lg = name

            avg_ang = np.median(val, axis=1)
            sem_avg_ang = np.std(val, axis=1) / math.sqrt(max_counterscounters[name])
            axarr[5].errorbar(x=np.linspace(0, len(avg_ang), len(avg_ang)), y=avg_ang, yerr=sem_avg_ang,
                              color=np.add(colors[name], 0.3), alpha=0.65,
                              linewidth=4, label=lg)
        # median_pvals = []
        # for i in range(3600):
        #     s, p = scipy.stats.ranksums(velocities['FlipFlop Maze'][i, :],
        #                                 velocities['PathInt2'][i, :])
        #     median_pvals.append(p)
        # axarr[3].set_yscale('log')
        # axarr[3].plot(median_pvals, color=[.7, .7, .7], linewidth=2)
        # axarr[3].axhline(0.05/3600, color='w', linestyle=':')

        for ax in axarr:
            ax.axvline(1800, color='w')
            make_legend(ax, [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=16)

        xlims = [1780, 1950]
        axarr[0].set(title='MEAN VELOCITY aligned to stim', xlim=xlims, xlabel='Frames', ylabel='px/frame',
                     ylim=[-0.5, 20], facecolor=[.0, .0, .0])
        axarr[1].set(title='MEDIAN Velocity aligned to stim', xlim=xlims,  xlabel='Frames', ylabel='px/frame',
                     ylim=[-0.5, 20], facecolor=[.0, .0, .0])
        axarr[3].set(title='Medians Wilcoxon rank-sum p-Val', xlim=xlims, ylim=[0, 1],
                     xlabel='Frames', ylabel='px/frame', facecolor=[.0, .0, .0])
        axarr[2].set(title='MEAN Head Angular VELOCITY', xlim=xlims, ylim=[-0.1, 20], xlabel='Frames',
                     ylabel='def/frame',  facecolor=[.0, .0, .0])
        axarr[4].set(title='MEAN normalised body LENGTH', xlim=xlims, ylim=[0.5, 1.25], xlabel='Frames',
                     ylabel='def/frame', facecolor=[.0, .0, .0])
        axarr[5].set(title='MEAN head-body angle', xlim=xlims, ylim=[0, 0.6], xlabel='Frames',
                     ylabel='def/frame', facecolor=[.0, .0, .0])
        a = 1

    def process_by_mouse(self, tracking_data):
        sess_id = None
        mice = []
        confs, ors, escs, outs = [], [], [], []
        configs, origins, escapes, outcomes = [], [], [], []
        counter = 0
        for trial in tracking_data.trials:
            tr_sess_id = int(trial.name.split('-')[0])
            if counter == 0:
                sess_id = tr_sess_id
                mice.append(sess_id)
            counter += 1
            ors.append(trial.processing['Trial outcome']['threat_origin_arm'])
            escs.append(trial.processing['Trial outcome']['threat_escape_arm'])
            outs.append(trial.processing['Trial outcome']['trial_outcome'])
            confs.append(trial.processing['Trial outcome']['maze_configuration'])

            if tr_sess_id != sess_id:
                origins.append(ors), escapes.append(escs), outcomes.append(outs), mice.append(tr_sess_id)
                configs.append(confs)
                confs, ors, escs, outs = [], [], [], []
                sess_id = tr_sess_id
        origins.append(ors), escapes.append(escs), outcomes.append(outs), configs.append(confs)

        all_configs = ['All']+ [c for c in set(configs[0])]
        sides = ['Left', 'Central', 'Right']
        for conf in all_configs:
            f, axarr = plt.subplots(round(len(origins)/4), 5, facecolor=[0.1, 0.1, 0.1])
            f.tight_layout()
            axarr = axarr.flatten()

            for mousenum in range(len(origins)):
                ori_probs, escs_probs = {side: None for side in sides}, {side: None for side in sides}
                if conf == 'All':
                    ors = [o for i,o in enumerate(origins[mousenum]) if outcomes[mousenum][i]]
                    escs = [e for i,e in enumerate(escapes[mousenum]) if outcomes[mousenum][i]]
                else:
                    ors = [o for i, o in enumerate(origins[mousenum])
                           if configs[mousenum][i] == conf and outcomes[mousenum][i]]
                    escs = [e for i, e in enumerate(escapes[mousenum])
                            if configs[mousenum][i] == conf and outcomes[mousenum][i]]
                num_trials = len(ors)

                for idx, side in enumerate(sides):
                    ori_probs[side] = len([o for i, o in enumerate(ors) if side in o])/num_trials
                    escs_probs[side] = len([e for i, e in enumerate(escs) if side in e])/num_trials

                    axarr[mousenum].bar(idx - 4, ori_probs[side], color=self.colors[side.lower()])
                    axarr[mousenum].bar(idx + 4, escs_probs[side], color=self.colors[side.lower()])

                axarr[mousenum].axvline(0, color='w')
                axarr[mousenum].set(title='m{} - {} trial'.format(mice[mousenum], num_trials), ylim=[0, 1],
                                    facecolor=[0.2, 0.2, 0.2])

        plt.show()
        a = 1

    def process_status_at_stim(self, tracking_data):
        statuses = dict(positions=[], orientations=[], velocities=[], body_lengts=[], origins=[], escapes=[])

        orientation_th = 30  # degrees away from facing south considered as facing left or right
        orientation_arm_probability = {ori:[0, 0, 0] for ori in ['looking_left', 'looking_down', 'looking_right']}
        non_alternations, alltrials = 0, 0
        for trial in tracking_data.trials:
            if 'status at stimulus' in trial.processing.keys():
                # get Threat ROI location
                try:
                    trial_sess_id = int(trial.name.split('-')[0])
                    threat_loc = None
                    for expl in tracking_data.explorations:
                        if not isinstance(expl, tuple):
                            sess_id = expl.metadata.session_id
                            if sess_id == trial_sess_id:
                                threat = expl.processing['ROI analysis']['all rois']['Threat_platform']
                                threat_loc = ((threat.topleft[0]+threat.bottomright[0])/2,
                                              (threat.topleft[1]+threat.bottomright[1])/2)
                                break
                    if threat_loc is None: raise Warning('Problem')

                    # prep status
                    status = trial.processing['status at stimulus']
                    if status is None: continue
                    outcome = trial.processing['Trial outcome']
                    ori = status.status['Orientation']
                    while ori>360: ori -= 360
                    pos = (status.status['x'], status.status['y'])

                    statuses['positions'].append((pos[0]-threat_loc[0], pos[1]-threat_loc[1]))
                    statuses['orientations'].append(ori)
                    statuses['velocities'].append(status.status['Velocity'])
                    statuses['body_lengts'].append(status.status['Body length'])
                    statuses['origins'].append(outcome['threat_origin_arm'])
                    statuses['escapes'].append(outcome['threat_escape_arm'])

                    # Check correlation between escape arm and orientation at stim
                    if not outcome['trial_outcome']:
                        print(trial.name, ' no escape')
                        continue
                    if 0 < ori < 180-orientation_th:
                        look = 'looking_right'
                    elif 180-orientation_th <= ori <= 180 + orientation_th:
                        look = 'looking_down'
                    else:
                        look = 'looking_left'
                    if 'Left' in outcome['threat_escape_arm']: esc = 0
                    elif 'Central' in outcome['threat_escape_arm']: esc = 1
                    else: esc = 2
                    orientation_arm_probability[look][esc] = orientation_arm_probability[look][esc] + 1

                    # Check probability of alternating
                    alltrials += 1
                    if outcome['threat_origin_arm'] == outcome['threat_escape_arm']: non_alternations += 1
                except:
                    raise Warning('ops')
                    pass

            else:
                print('no status')

        # Print probability of alternation (oirigin vs escape)
        print('Across this dataset, the probability of escaping on the arm of origin was: {}'.format(non_alternations/alltrials))

        # Plot probability of arm given orientation
        f, axarr = plt.subplots(3, 1, facecolor=[0.1, 0.1, 0.1])
        for idx, look in enumerate(orientation_arm_probability.keys()):
            outcomes = orientation_arm_probability[look]
            num_trials = sum(outcomes)
            ax = axarr[idx]

            for i, out in enumerate(outcomes):
                ax.bar(i, out/num_trials)
            ax.set(title='Escape probs for trials {}'.format(look), ylim=[0, 1], xlabel='escape arm', ylabel='probability',
                   facecolor=[0.2, 0.2, 0.2])


        # Plot location and stuff
        f, axarr = plt.subplots(3, 2, facecolor=[0.1, 0.1, 0.1])
        f.tight_layout()
        f.set_size_inches(10, 20)
        axarr = axarr.flatten()


        axarr[0].scatter(np.asarray([p[0] for p in statuses['positions']]), np.asarray([p[1] for p in statuses['positions']]),
                   c=np.asarray(statuses['orientations']), s=np.multiply(np.asarray(statuses['velocities']), 10))

        sides = ['Left', 'Central', 'Right']
        for side in sides:
            axarr[2].scatter(np.asarray([p[0] for i, p in enumerate(statuses['positions'])
                                         if side in statuses['origins'][i]]),
                             np.asarray([p[1] for i, p in enumerate(statuses['positions'])
                                         if side in statuses['origins'][i]]),
                             c=self.colors[side.lower()], s=35, edgecolors='k', alpha=0.85)

            axarr[3].scatter(np.asarray([p[0] for i, p in enumerate(statuses['positions'])
                                         if side in statuses['escapes'][i]]),
                             np.asarray([p[1] for i, p in enumerate(statuses['positions'])
                                         if side in statuses['escapes'][i]]),
                             c=self.colors[side.lower()], s=35, edgecolors=['k'], alpha=0.85)

        axarr[4].scatter(np.asarray([p[0] for i, p in enumerate(statuses['positions'])
                                     if statuses['escapes'][i] == statuses['origins'][i]]),
                         np.asarray([p[1] for i, p in enumerate(statuses['positions'])
                                     if statuses['escapes'][i] == statuses['origins'][i]]),
                         c=[.5, .2, .7], s=35, edgecolors=['k'], alpha=0.85, label='Same')
        axarr[4].scatter(np.asarray([p[0] for i, p in enumerate(statuses['positions'])
                                     if statuses['escapes'][i] != statuses['origins'][i]]),
                         np.asarray([p[1] for i, p in enumerate(statuses['positions'])
                                     if statuses['escapes'][i] != statuses['origins'][i]]),
                         c=[.2, .7, .5], s=35, edgecolors=['k'], alpha=0.85, label='Different')
        make_legend(axarr[4], [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=8)

        axarr[5].scatter(np.asarray([p[0] for i, p in enumerate(statuses['positions'])
                                     if statuses['orientations'][i] > 180+30]),
                         np.asarray([p[1] for i, p in enumerate(statuses['positions'])
                                     if statuses['orientations'][i] > 180+30]),
                         c=[.5, .2, .7], s=35, edgecolors=['k'], alpha=0.85, label='Looking Right')
        axarr[5].scatter(np.asarray([p[0] for i, p in enumerate(statuses['positions'])
                                     if 180-30 <= statuses['orientations'][i] <= 180+30]),
                         np.asarray([p[1] for i, p in enumerate(statuses['positions'])
                                     if 180-30 <= statuses['orientations'][i] <= 180+30]),
                         c=[.7, .5, .2], s=35, edgecolors=['k'], alpha=0.85, label='Looking Down')
        axarr[5].scatter(np.asarray([p[0] for i, p in enumerate(statuses['positions'])
                                     if statuses['orientations'][i] < 180-30]),
                         np.asarray([p[1] for i, p in enumerate(statuses['positions'])
                                     if statuses['orientations'][i] < 180-30]),
                         c=[.2, .7, .5], s=35, edgecolors=['k'], alpha=0.85, label='looking Left')
        make_legend(axarr[5], [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=8)

        titles = ['Position, Velocity, Orientation', '', 'Position, arm of ORIGIN', 'Position, arm of ESCAPE',
                  'Position and same arm', 'Position, ORIENTATION', '', '']
        for idx, ax in enumerate(axarr):
            ax.set(title = titles[idx], xlim=[-75, 75], ylim=[-75, 75], xlabel='x pos', ylabel='y pos',
                   facecolor=[0.2, 0.2, 0.2])
        # plt.show()

    def process_explorations(self, tracking_data):
        all_platforms = ['Left_far_platform', 'Left_med_platform', 'Right_med_platform',
                         'Right_far_platform', 'Shelter_platform', 'Threat_platform']
        alt_all_platforms = ['Left_far_platform', 'Left_medium_platform', 'Right_medium_platform',
                         'Right_far_platform', 'Shelter_platform', 'Threat_platform']

        transitions_list = []
        all_transitions, times = {name:[] for name in all_platforms}, {name:[] for name in all_platforms}
        fps = None
        for exp in tracking_data.explorations:
            if not isinstance(exp, tuple):
                if fps is None: fps = exp.metadata.videodata[0]['Frame rate'][0]
                exp_platforms = [p for p in exp.processing['ROI analysis']['all rois'].keys() if 'platform' in p]

                transitions_list.append(exp.processing['ROI transitions'])

                for p in exp_platforms:  # This is here because some experiments have the platforms named differently
                    if p not in all_platforms:
                        all_platforms = alt_all_platforms
                        all_transitions, times = {name: [] for name in all_platforms}, {name: [] for name in
                                                                                        all_platforms}
                        break
                        # raise Warning()

                timeinroi = {name:val for name,val in exp.processing['ROI analysis']['time_in_rois'].items()
                            if 'platform' in name}
                transitions = {name:val for name,val in exp.processing['ROI analysis']['transitions_count'].items()
                            if 'platform' in name}

                for name in all_platforms:
                    if name in transitions: all_transitions[name].append(transitions[name])
                    if name in timeinroi: times[name].append(timeinroi[name])

        all_platforms = [all_platforms[0], all_platforms[1], all_platforms[4],
                         all_platforms[5], all_platforms[2], all_platforms[3]]

        # Get platforms to and from threat
        all_alternations = []
        for sess in transitions_list:
            alternations = [(sess[i-1], sess[i+1]) for i,roi in enumerate(sess)
                            if 0<i<len(sess)-1 and 'threat' in roi.lower()]
            all_alternations.append(alternations)

        all_alternations = flatten_list(all_alternations)
        num_threat_visits = len(all_alternations)
        num_threat_aternations = len([pp for pp in all_alternations if pp[0]!=pp[1]])
        alternation_probability = round((num_threat_aternations/num_threat_visits), 2)
        print('Average prob: {}'.format(alternation_probability))

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

        # plt.show()

    def process_trials(self, tracking_data, select_by_speed='fastest'):
        """ select by speed can be:
                fastest: 33% of trials with fastest time to peak acceleration
                slowest: 33% of trials with slowest time to peak acceleration
                False: take all trials

            The code looks at the peak to acceleration in the first 2s (60 frames)
        """
        maze_configs, origins, escapes, outcomes = [], [], [], []
        frames_to_peak_acc = []
        for trial in tracking_data.trials:
            outcome = trial.processing['Trial outcome']['trial_outcome']
            if not outcome:
                print(trial.name, ' no escape')
                continue
            maze_configs.append(trial.processing['Trial outcome']['maze_configuration'])
            origins.append(trial.processing['Trial outcome']['threat_origin_arm'])
            escapes.append(trial.processing['Trial outcome']['threat_escape_arm'])
            outcomes.append(outcome)

            # Get time to peak acceleration
            head_ang_vel = np.abs(line_smoother(trial.dlc_tracking['Posture']['body']['Head ang vel'].values)[1800:1860])
            frames_to_peak_acc.append(np.argmax(head_ang_vel))


        num_trials = len(outcomes)
        left_origins = len([b for b in origins if 'Left' in b])
        central_origins = len([b for  b in origins if 'Central' in b])
        left_escapes = len([b for b in escapes if 'Left' in b])
        central_escapes = len([b for b in escapes if 'Central' in b ])

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


        def plotty(ax, numtr, lori, cori, lesc, cesc, title='Origin and escape probabilities'):

            basecolor = [.3, .3, .3]

            ax.bar(0, lori / numtr, color=basecolor, width=.25, label='L ori')
            ax.bar(0.25, cori / numtr, color=np.add(basecolor, .2), width=.25, label='C ori')
            ax.bar(0.5, 1 - ((lori + cori) / numtr), color=np.add(basecolor, .4), width=.25, label='R ori')

            ax.axvline(0.75, color=[1, 1, 1])

            ax.bar(1, lesc / numtr, color=basecolor, width=.25, label='L escape')
            ax.bar(1.25, cesc / numtr, color=np.add(basecolor, .2), width=.25, label='C escape')
            ax.bar(1.5, 1 - ((lesc + cesc) / numtr), color=np.add(basecolor, .4), width=.25, label='R escape')

            ax.set(title='{} - {} trials'.format(title, numtr), ylim=[0, 1], facecolor=[0.2, 0.2, 0.2])
            # make_legend(ax, [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=8)


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
            for idx, config in enumerate(sorted(outcomes_perconfig.keys())):
                ax = axarr[idx+1]

                o = outcomes_perconfig[config]
                num_trials = o['escapes'].numescapes
                left_escapes = o['escapes'].numleftescapes
                left_origins = o['origins'].numleftorigins
                plotty(ax, num_trials, left_origins, 0, left_escapes, 0, title='{} config'.format(config))

        f.tight_layout()
        #plt.show()

