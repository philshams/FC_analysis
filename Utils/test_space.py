
class ProcessingTrialsMaze:
    """ Applies processsing steps that are specific to maze experiments (e.g. get arm of escape) """

    def __init__(self, session, settings=None, debugging=True):
        print(colored('\n      Maze-specific single trials processing', 'green'))
        self.session = session
        self.settings = settings
        self.debugging = debugging

        # Subdivide frame
        self.rois, self.boundaries = self.subdivide_frame()
        self.maze_rois = []
        self.get_maze_structure()

        # Process each trial
        tracking_items = self.session.Tracking.keys()
        if tracking_items:
            for item in tracking_items:
                retval = self.get_trial_traces(item)  # get coordinates

                if retval:
                    self.assign_pose_to_roi(item)  # get the roi the mouse is on at each frame
                    # self.get_intersections(item)
                    self.get_origin_escape_arms(item)  # get escape arms
                    self.extract_escape_stats(item)  # get stats of escape
                    self.get_status_at_timepoint(item)  # get status at stim

                    if debugging:  # display results for debugging
                        self.debug_preview()
                        self.plot_trace()
                        plt.show()
                else:
                    from warnings import warn
                    if 'whole' not in item.lower() and 'exploration' not in item.lower():
                        warn('Something went wrong when applying maze-processing')
                        slack_chat_messenger('Something went wrong with maze-processing, trial {}'.format(item))
                    print(colored('          did not apply maze-processing to trial {}'.format(item), 'yellow'))

    """ Get position of mouse on the maze for all frames """

    def get_maze_structure(self):
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

        # Get the templates
        exp_name = self.session.Metadata.experiment
        base_fld = self.settings['templates folder']
        bg_folder = os.path.join(base_fld, 'Bgs')
        templates_fld = os.path.join(base_fld, exp_name)
        f_name = '{}.png'.format(self.session.name)
        if not f_name in os.listdir(bg_folder):
            cv2.imwrite(os.path.join(bg_folder, f_name), gray)

        platf_templates = [os.path.join(templates_fld, f) for f in os.listdir(templates_fld) if 'platform' in f]
        bridge_templates = [os.path.join(templates_fld, f) for f in os.listdir(templates_fld) if 'bridge' in f]

        # Calculate the position of the templates and save resulting image
        display, platforms = loop_over_templates(platf_templates, bg)
        display, bridges = loop_over_templates(bridge_templates, display, bridge_mode=True)
        self.maze_rois = {**platforms, **bridges}
        cv2.imwrite(os.path.join(base_fld, 'Matched\\{}.png'.format(self.session.name)), display)

    @clock
    def assign_pose_to_roi(self, trial_name):
        data = self.session.Tracking[trial_name]
        if 'Posture' in data.dlc_tracking.keys():
            data_length = len(data.dlc_tracking['Posture']['body'].x.values)
            pos = np.zeros((data_length, 2))
            pos[:, 0] = data.dlc_tracking['Posture']['body'].x.values
            pos[:, 1] = data.dlc_tracking['Posture']['body'].y.values

            centers, roi_names = [], []
            for name, points in self.maze_rois.items():
                center_x = (points.topleft[0] + points.bottomright[0]) / 2
                center_y = (points.topleft[1] + points.bottomright[1]) / 2

                center = np.asarray([center_x, center_y])
                centers.append(center)
                roi_names.append(name)

            distances = np.zeros((data_length, len(centers)))
            for idx, center in enumerate(centers):
                cnt = np.tile(center, data_length).reshape((data_length, 2))
                dist = np.hypot(np.subtract(cnt[:, 0], pos[:, 0]), np.subtract(cnt[:, 1], pos[:, 1]))
                distances[:, idx] = dist

            sel_rois = np.argmin(distances, 1)
            roi_at_each_frame = tuple([roi_names[x] for x in sel_rois])

            # store data
            data.processing['Maze rois'] = self.maze_rois
            e = pd.Series(roi_at_each_frame)
            data.dlc_tracking['Posture']['body'] = data.dlc_tracking['Posture']['body'].assign(maze_roi=e.values)

    @clock
    def get_trial_traces(self, trial_name):
        data = self.session.Tracking[trial_name]
        if isinstance(data, dict):
            # Doesnt work for wholetrace or exploration data, which are saved as dictionaries
            return False

        dlc_tracer, tracer = namedtuple('trace', 'x y velocity orientation  headangvel'), namedtuple('trace', 'x y')
        if 'Posture' in data.dlc_tracking.keys():
            # We have deeplabcut data
            try:
                self.trace = dlc_tracer(data.dlc_tracking['Posture']['body']['x'].values,
                                        data.dlc_tracking['Posture']['body']['y'].values,
                                        data.dlc_tracking['Posture']['body']['Velocity'].values,
                                        data.dlc_tracking['Posture']['body']['Orientation'].values,
                                        data.dlc_tracking['Posture']['body']['Head ang vel'].values)
            except:
                self.trace = tracer(data.std_tracking['x'],
                                    data.std_tracking['y'])
        elif data.std_tracking['x'] is not None:
            self.trace = tracer(data.std_tracking['x'],
                                data.std_tracking['y'])
        else:
            print('Could not find trial data for trial {}'.format(trial_name))
            return False

        # Get stimulus-locked traces
        window = int(len(self.trace.x) / 2)
        if len(self.trace._fields) == 2:  # we used STD tracking
            self.prestim_trace = tracer(self.trace.x[:window], self.trace.y[:window])
            self.poststim_trace = tracer(self.trace.x[window:], self.trace.y[window:])
            self.shelter_to_stim, self.stim_to_shelter = self.get_leaveenter_rois(tracer)
        else:  # used DLC tracking
            self.prestim_trace = dlc_tracer(self.trace.x[:window], self.trace.y[:window],
                                            self.trace.velocity[:window], self.trace.orientation[:window],
                                            self.trace.headangvel[:window])
            self.poststim_trace = dlc_tracer(self.trace.x[window:], self.trace.y[window:],
                                             self.trace.velocity[window:], self.trace.orientation[window:],
                                             self.trace.headangvel[window:])
            self.shelter_to_stim, self.stim_to_shelter = self.get_leaveenter_rois(dlc_tracer)

        # Store results
        if not 'processing' in data.__dict__.keys():
            setattr(data, 'processing', {})
        if not 'PlottingItems' in data.processing.keys():
            data.processing['PlottingItems'] = dict(prestim_trace=self.prestim_trace,
                                                    poststim_trace=self.poststim_trace,
                                                    shelt_to_stim_trace=self.shelter_to_stim,
                                                    stim_to_shelt_trace=self.stim_to_shelter)

        return True

    def subdivide_frame(self):
        """
        Subdivide the frame into regions: when the mouse is in one of those regions we know that it is on one
        of the arms. This can be done using user defined ROIs if present, alternative it is assumed that the maze
        is centered on the frame
        """
        # handles to metadata items
        rois = self.session.Metadata.videodata[0]['User ROIs']
        self.bg = self.session.Metadata.videodata[0]['Background']

        # handle to frame subdivision lines
        """
        First two lines split the frame in half vertically and horizontally, the second two are parallel to 
        the vertical midline and are located at the horizontal edges of the shelter
        """
        boundaries = namedtuple('boundaries', 'x_midline y_midline l_shelteredge r_shelteredge')
        shelter_width = 100  # assumed shelterwidth if not specified in a ROI

        if not 'Shelter' in rois.keys() or rois['Shelter'] is None:
            # Split the video assuming the the shelter is centered on it
            width, height = np.shape(self.bg)
            # split everything in hald and assign
            width = int(float(width / 2))
            height = int(float(height / 2))
            shelter_width = int(float(shelter_width / 2))
            limits = boundaries(width, height, width - shelter_width, width + shelter_width)

        else:
            # Split the video using the user defined rois
            shelter = rois['Shelter']
            threat = rois['Threat']

            point = namedtuple('point', 'x y')

            shelter_cntr = point(int(shelter[0] + (shelter[2] / 2)), int(shelter[1] + (shelter[3] / 2)))
            threat_cntr = point(int(threat[0] + (threat[2] / 2)), int(threat[1] + (threat[3] / 2)))

            # Get midpoint between  between the rois
            midpoint = point(int((shelter_cntr.x + threat_cntr.x) / 2), int((shelter_cntr.y + threat_cntr.y) / 2))

            # now get the edges of the shelter
            edges = int(midpoint.x - (shelter[2] / 2)), int(midpoint.x + (shelter[2] / 2))

            # now put everything together in the limits
            limits = boundaries(midpoint.x, midpoint.y, edges[0], edges[1])
        return rois, limits

    """
    Extract stuff
    """

    def get_leaveenter_rois(self, tracer):
        """ Get the times at which the mouse enters and exits the rois and the corresponding traces """

        def get_leave_enter_times(roi, trace, pre=True):
            # from x, width, y, height to --> x0, x1, y0, y1
            roi = (roi[0], roi[0] + roi[2], roi[1], roi[1] + roi[3])

            # Get the frames in which the mouse is in the shelter
            x_inshelt = np.where((roi[0] < trace.x) & (trace.x < roi[1]))[0]
            y_inshelt = np.where((roi[2] < trace.y) & (trace.y < roi[3]))[0]
            inshelt = np.intersect1d(x_inshelt, y_inshelt)

            if not len(inshelt):
                return False

            # If its before the stimulus we want the last frame, else we want the last frame
            if pre:
                if len(inshelt):  # means no point on the trace was in sheter
                    return inshelt[-1]
                else:
                    return len(trace) - 1
            else:
                if len(inshelt):
                    return inshelt[0]
                else:
                    return 0

        tostim, fromstim = False, False
        if 'Shelter' in self.rois.keys() and self.rois['Shelter'] is not None:
            leaves_shelter = get_leave_enter_times(self.rois['Shelter'], self.prestim_trace)
            enters_shelter = get_leave_enter_times(self.rois['Shelter'], self.poststim_trace, pre=False)

            if len(tracer._fields) == 2:  # we used STD
                tostim = tracer(self.prestim_trace.x[leaves_shelter:], self.prestim_trace.y[leaves_shelter:])
                fromstim = tracer(self.poststim_trace.x[:enters_shelter + 1],
                                  self.poststim_trace.y[:enters_shelter + 1])
            else:
                tostim = tracer(self.prestim_trace.x[leaves_shelter:], self.prestim_trace.y[leaves_shelter:],
                                self.prestim_trace.velocity[leaves_shelter:],
                                self.prestim_trace.orientation[leaves_shelter:],
                                self.prestim_trace.headangvel[leaves_shelter:])
                fromstim = tracer(self.poststim_trace.x[:enters_shelter + 1],
                                  self.poststim_trace.y[:enters_shelter + 1],
                                  self.poststim_trace.velocity[:enters_shelter + 1],
                                  self.poststim_trace.orientation[:enters_shelter + 1],
                                  self.poststim_trace.headangvel[:enters_shelter + 1])
        return tostim, fromstim

    def get_status_at_timepoint(self, name, time: int = None, timename: str = 'stimulus'):
        """
        Get the status of the mouse [location, orientation...] at a specific timepoint.
        If not time is give the midpoint of the tracking traces is taken as stim time
        """
        if not 'session' in name.lower() or 'exploration' in name.lower:
            data = self.session.Tracking[name]

            if data.dlc_tracking.keys():
                if time is None:  # if a time is not give take the midpoint
                    time = int(len(self.trace.x) / 2)

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

    def get_origin_escape_arms(self, name):
        def get_arm(trace, midline, halfwidth):
            maxleft = abs(np.min(trace.x) - midline)
            maxright = abs(np.max(trace.x) - midline)
            if maxleft < halfwidth and maxright < halfwidth:
                return 'Central'
            else:
                if maxleft > maxright:
                    return 'Left'
                else:
                    return 'Right'

        # Get origin and escape arms, check the leftmost and rightmost point in each trace and see which one is most
        # distant from the midline
        midline = self.boundaries.x_midline
        central_corr_halfwidth = abs(self.boundaries.l_shelteredge - midline)

        # Get arms and print results
        origin = get_arm(self.shelter_to_stim, midline, central_corr_halfwidth)
        escape = get_arm(self.stim_to_shelter, midline, central_corr_halfwidth)
        if self.debugging:
            print('Arm of origin: {}   -   Arm of escape: {}'.format(origin, escape))

        # Store results
        self.session.Tracking[name].processing['Origin'] = origin
        self.session.Tracking[name].processing['Escape'] = escape

    def extract_escape_stats(self, name):
        """  extract stuff like max velocity... from the post-stim trace """
        data = self.session.Tracking[name]

        if len(self.poststim_trace._fields) > 2:  # we used deeplabcut
            maxvel = max(self.poststim_trace.velocity)
            maxheadangvel = max(self.poststim_trace.headangvel)

            escape_stats = dict(max_velocity=maxvel, max_ang_vel=maxheadangvel)

            data.processing['Escape stats'] = escape_stats

        else:
            from warnings import warn
            warn('Havent implemented the extraction of escape stats from STD tracking yet')
            slack_chat_messenger('Need to implement the extraction of maze escape stats from STD tracking bro')

    """
    Debugging related stuff
    """

    def debug_preview(self):
        """
        Display the results of the processing so that we can check that everything went okay
        """
        f, self.ax = plt.subplots()
        self.ax.imshow(self.bg, cmap='gray')
        self.ax.axvline(self.boundaries.x_midline, color=[.8, .8, .8], linewidth=2, alpha=0.75)
        self.ax.axvline(self.boundaries.l_shelteredge, color=[.8, .8, .8], linewidth=2, alpha=0.75)
        self.ax.axvline(self.boundaries.r_shelteredge, color=[.8, .8, .8], linewidth=2, alpha=0.75)
        self.ax.axhline(self.boundaries.y_midline, color=[.8, .8, .8], linewidth=2, alpha=0.75)

    def plot_trace(self):
        self.ax.plot(self.shelter_to_stim.x, self.shelter_to_stim.y, color=[.8, .2, .2], linewidth=2)
        self.ax.plot(self.stim_to_shelter.x, self.stim_to_shelter.y, color=[.2, .2, .8], linewidth=2)


class ProcessingSessionMaze:
    def __init__(self, session):
        print(colored('\n      Maze-specific whole session processing', 'green'))

        self.session = session

        # Process stuff
        self.get_origins_escapes()
        self.get_probs()

    @clock
    def get_origins_escapes(self):
        # Get origina nd escape arms
        info = namedtuple('info', 'arm xpos status')
        origins, escapes = [], []
        origins_dict, escapes_dict = {}, {}
        for trial_name in self.session.Tracking.keys():
            if 'session' in trial_name.lower() or 'exploration' in trial_name.lower():
                continue
            origins.append(self.session.Tracking[trial_name].processing['Origin'])
            escapes.append(self.session.Tracking[trial_name].processing['Escape'])

            if self.session.Tracking[trial_name].processing['status at stimulus'] is not None:
                origins_dict[trial_name] = info(self.session.Tracking[trial_name].processing['Origin'],
                                                self.session.Tracking[trial_name].
                                                processing['status at stimulus'][0].body.x,
                                                self.session.Tracking[trial_name].processing['status at stimulus'])
                escapes_dict[trial_name] = info(self.session.Tracking[trial_name].processing['Escape'],
                                                self.session.Tracking[trial_name].
                                                processing['status at stimulus'][0].body.x,
                                                self.session.Tracking[trial_name].processing['status at stimulus'])

        if not 'processing' in self.session.keys():
            setattr(self.session, 'processing', {})

        self.session.processing['Escapes'] = escapes
        self.session.processing['Origins'] = origins
        self.session.processing['Origins dict'] = escapes_dict
        self.session.processing['Escapes dict'] = origins_dict

    @clock
    def get_probs(self):
        def get_probs_as_func_of_x_pos(data):
            # Get probs as a function of (adjusted) hor. position on threat platform at stim onset
            x_events = []  # get arm and x position from data
            for d in data:
                x_events.append(([1 if d[0] == 'Left'
                                  else 2 if d[0] == 'Central'
                else 3 if d[0] == 'Right'
                else 0],
                                 d.status.status.adjustedx))
                x_events = sorted(x_events, key=lambda tup: tup[1])  # sorted by X position

            # Bin based on X pos
            boundaries = [-100, 100]
            lims = np.linspace(boundaries[0], boundaries[1], 5)
            step = abs(np.diff(lims)[0]) - 1
            binned_events = []
            for l in lims[:-1]:
                L = l + step
                events = [(outcome, x) for outcome, x in x_events if l <= x <= L]
                x_events = [x for x in x_events if x not in events]
                binned_events.append(events)

            # Get probs for each bin
            binned_probs = []
            p = namedtuple('p', 'left central right')
            for probs in binned_events:
                num_events = len(probs)
                if num_events:
                    lprobs = len([x for x in probs if x[0] == [1]]) / num_events
                    cprobs = len([x for x in probs if x[0] == [2]]) / num_events
                    rprobs = len([x for x in probs if x[0] == [3]]) / num_events
                    binned_probs.append(p(lprobs, cprobs, rprobs))
                else:
                    binned_probs.append(p(0, 0, 0))

            return binned_probs

        # Define tuple to store results
        probs = namedtuple('probabilities', 'per_arm per_x')
        """ probabilities:
                - per arm: probability of origin or escape along one arm
                - per x  : probability as a function of horizontal position [adjusted position]
        """

        # Get probability of escape or origin for each arm
        possibilites = [0, 1, 2, 3]
        s_possibilities = ['None', 'Left', 'Central', 'Right']
        num_origins, num_escapes = len(self.session.processing['Origins']), len(self.session.processing['Escapes'])
        if num_origins:
            origins_probs = [self.session.processing['Origins'].count(x) / num_origins for x in s_possibilities]
        else:
            origins_probs = [0 for x in s_possibilities]

        if num_escapes:
            escapes_probs = [self.session.processing['Escapes'].count(x) / num_escapes for x in s_possibilities]
        else:
            escapes_probs = [0 for x in s_possibilities]

        # Get Binned probs
        if self.session.processing['Escapes dict'].keys():
            x_binned_escapes = get_probs_as_func_of_x_pos(self.session.processing['Escapes dict'].values())
            x_binned_origins = get_probs_as_func_of_x_pos(self.session.processing['Origins dict'].values())

            # Set output
            escape_probs = probs(escapes_probs, x_binned_escapes)
            origin_probs = probs(origins_probs, x_binned_origins)
            self.session.processing['Escape probabilities'] = escape_probs
            self.session.processing['Origin probabilities'] = origin_probs
        else:
            self.session.processing['Escape probabilities'] = None
            self.session.processing['Origin probabilities'] = None


class Processing_cohortMaze:
    def __init__(self, cohort):
        # Set up
        coh_name = cohort_options['name']
        self.coh = cohort
        self.coh_metadata = cohort.Metadata[coh_name]
        self.coh_tracking = cohort.Tracking[coh_name]
        self.coh_trials = cohort.Tracking[coh_name].trials
        self.get_additional_metadata()

        # Do stuff
        self.get_origins_escapes_per_mouse(coh_name)

    def get_additional_metadata(self):
        mice_in_cohort = [sess[1].mouse_id for sess in self.coh_metadata.sessions_in_cohort]
        self.coh_metadata = [self.coh_metadata, mice_in_cohort]

    def get_origins_escapes_per_mouse(self, coh_name):
        origins, escapes = {}, {}
        paths = namedtuple('pahts', 'none left central right')
        for tr in self.coh_trials:
            sess_id = tr.name.split('-')[0]
            ids = [s[0].split('_')[0] for s in self.coh_metadata[0].sessions_in_cohort]
            sess_metadata = self.coh_metadata[0].sessions_in_cohort[ids.index(sess_id)][1]
            mouse_id = sess_metadata.mouse_id
            if not mouse_id in origins.keys():
                origins[mouse_id] = []
                escapes[mouse_id] = []

            if tr.processing['Escape'] == 'Left':
                escapes[mouse_id].append(1)
            elif tr.processing['Escape'] == 'Central':
                escapes[mouse_id].append(2)
            elif tr.processing['Escape'] == 'Right':
                escapes[mouse_id].append(3)
            else:
                escapes[mouse_id].append(0)

            if tr.processing['Origin'] == 'Left':
                origins[mouse_id].append(1)
            elif tr.processing['Origin'] == 'Central':
                origins[mouse_id].append(2)
            elif tr.processing['Origin'] == 'Right':
                origins[mouse_id].append(3)
            else:
                origins[mouse_id].append(0)

        self.coh.loc[coh_name].Processing['Origins'] = origins
        self.coh.loc[coh_name].Processing['Escapes'] = escapes
