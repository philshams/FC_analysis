import matplotlib.pyplot as plt  # used for debugging
import numpy as np
from collections import namedtuple
from copy import deepcopy


class ProcessingMaze():
    def __init__(self, session, debugging=True):
        """
        Processing steps that are specific to maze experiments (e.g. extract arm of escape, orgin...)
        :param session:
        """
        self.session = session
        self.debugging = debugging
        # Subdivide frame
        self.boundaries = self.subdivide_frame()

        # Create window to display results
        if debugging:
            self.debug_preview()

        # Process each trial
        tracking_items =self.session.Tracking.keys()
        if tracking_items:
            for item in tracking_items:
                retval = self.get_trial_trace(item)
                if retval:
                    self.get_intersections(item)
                    self.get_status_at_timepoint(item)
                    self.get_origin_escape_arms(item)
                    if debugging:
                        self.plot_trace()

                        plt.show()

    """
    Prep functions
    """
    def subdivide_frame(self):
        """
        Subdivide the frame into regions: when the mouse is in one of those regions we know that it is on one
        of the arms. This can be done using user defined ROIs if present, alternative it is assumed that the maze
        is centered on the frame
        :return:
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
            width = int(float(width/2))
            height = int(float(height/2))
            shelter_width = int(float(shelter_width/2))
            limits = boundaries(width, height, width-shelter_width, width+shelter_width)

        else:
            # Split the video using the user defined rois
            # Get the centres of the rois
            shelter = rois['Shelter']
            threat = rois['Threat']

            point = namedtuple('point', 'x y')

            shelter_cntr = point(int(shelter[0]+(shelter[2]/2)), int(shelter[1]+(shelter[3]/2)))
            threat_cntr = point(int(threat[0]+(threat[2]/2)), int(threat[1]+(threat[3]/2)))

            # Get midpoint between  between the rois
            midpoint = point(int((shelter_cntr.x + threat_cntr.x)/2), int((shelter_cntr.y + threat_cntr.y)/2))

            # now get the edges of the shelter
            edges = int(midpoint.x-(shelter[2]/2)), int(midpoint.x+(shelter[2]/2))

            # now put everything together in the limits
            limits = boundaries(midpoint.x, midpoint.y, edges[0], edges[1])
        return limits

    def get_trial_trace(self, trial_name):
        data = self.session.Tracking[trial_name]

        tracer = namedtuple('trace', 'x y')
        if 'Posture' in data.dlc_tracking.keys():
            # We have deeplabcut data
            self.trace = tracer(data.dlc_tracking['Posture']['body']['x'].values,
                           data.dlc_tracking['Posture']['body']['y'].values)
        elif data.std_tracking['x'] is not None:
            self.trace = tracer(data.std_tracking['x'],
                           data.std_tracking['y'])
        else:
            return False
        return True

    """
    Extract stuff
    """
    def get_intersections(self, item):
        # TODO check min time inbetween crossings
        # TODO divide intersection between those occurring in the top and bottom halves of the frame
        # TODO divide intersections between escape and origin
        th, th2 = 5, 5
        # Find all points close to the boundary lines [ frames at which the mouse is crossing the boundaries]
        temp_intersections = {'x_midline': np.where(abs(self.trace.x-self.boundaries.x_midline) < th)[0],
                              'y_midline': np.where(abs(self.trace.y-self.boundaries.y_midline) < th)[0],
                              'l_shelteredge': np.where(abs(self.trace.x-self.boundaries.l_shelteredge) < th)[0],
                              'r_shelteredge': np.where(abs(self.trace.x-self.boundaries.r_shelteredge) < th)[0]}

        # discart consecutive points [i.e. only keep real crossings
        self.intersections = {}
        for name, values in temp_intersections.items():
            if len(values):
                firstval = np.array(values[0])
                goodvals = values[np.where(np.diff(values)>th2)[0]]
                if np.diff(values)[-1]>th2:
                    goods = np.insert(goodvals, 0, firstval)
                    self.intersections[name] = np.insert(goods, 0, values[-1])
                else:
                    self.intersections[name] = np.insert(goodvals, 0, firstval)
            else:
                self.intersections[name] = np.array([])

        # Get position at intersection times
        position_at_intersections = {}
        for name, timepoints in self.intersections.items():
            # print('Found {} intersection with {}'.format(len(timepoints), name))
            if len(timepoints):
                points = [c for idx, c in enumerate(zip(self.trace.x, self.trace.y)) if idx in timepoints]
                position_at_intersections[name] = points
            else:
                position_at_intersections[name] = []

        # Collate intersections timepoints and coordinates at intersections
        per_line = namedtuple('each', 'timepoints coordinates')
        all_lines = namedtuple('all', 'x_midline l_shelteredge r_shelteredge y_midline')
        names = position_at_intersections.keys()

        lines = []
        for idx, name in enumerate(names):
            line = per_line(self.intersections[name], position_at_intersections[name])
            lines.append(line)
        intersections_complete = all_lines(lines[0], lines[1], lines[2], lines[3])

        # get tracking data
        data = self.session.Tracking[item]
        # If processing is not in the trackind data class. add it
        if not 'processing' in data.__dict__.keys():
            setattr(data, 'processing', {})
        data.processing['boundaries intersections'] = intersections_complete

    def get_status_at_timepoint(self, name, time: int = None, timename: str = 'stimulus'):
        """
        Get the status of the mouse [location, orientation...] at a specific timepoint.
        If not time is give the midpoint is take
        """
        if not 'session' in name.lower() or 'exploration' in name.lower:
            data = self.session.Tracking[name]

            if time is None:  # if a time is not give take the midpoint
                time = int(len(data.dlc_tracking['Posture']['body']['x'].values)/2)

            # Create a named tuple with all the params from processing (e.g. head angle) and the selected time point
            # and store the values
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

    def get_origin_escape_arms(self, name):
        # Get point maximally distant from x midline, check where it is and thats your arm


        # TODO improve by getting times the mouse leaves and enters the shelter and limiting the search to the interval
        # between these times
        """
        Find escape and origin arm. Get timepoints at which the mouse intersected the boundary lines:
        last and first intersections denote origin and escape arms, respectively
        :param name:
        :return:
        """

        data = self.session.Tracking[name]
        stim_time = int(len(self.trace.x)/2)
        inters = data.processing['boundaries intersections']

        # Divide relevant intersection between those that happened before and after the stimulus
        l_inters = inters.l_shelteredge
        r_inters = inters.r_shelteredge
        c_inters = inters.x_midline
        y_inters = inters.y_midline

        pre_l, post_l = [t for t in l_inters.timepoints if t<stim_time], [t for t in l_inters.timepoints if t>stim_time]
        pre_r, post_r = [t for t in r_inters.timepoints if t<stim_time], [t for t in r_inters.timepoints if t>stim_time]
        pre_c, post_c = [t for t in c_inters.timepoints if t<stim_time], [t for t in c_inters.timepoints if t>stim_time]
        pre_cy, post_y = [t for t in y_inters.timepoints if t<stim_time], [t for t in y_inters.timepoints if t>stim_time]

        pres = [max(l) if l else -1 for l in [pre_l, pre_r, pre_c]]  # empty lists get assigned values that are def wrong
        poss = [min(l) if l else 10000 for l in [post_l, post_r, post_c]]

        # get escape arms: last interaction before stim and first interaction after determine escape arms
        arm_names = {'0':'left', '1':'right', '2':'centre'}
        origin = arm_names[str(pres.index(max(pres)))]  # location of last boundary cross before stim
        escape = arm_names[str(poss.index(min(poss)))]  # location of last boundary cross before stim

        # check that escapes actually happened
        if not post_y:
            # No escape occurred
            escape = False

        # Store results
        if self.debugging:
            print('Origin - {}, Escape - {}'.format(origin, escape))
        data.processing['Origin arm'] = origin
        data.processing['Escape arm'] = escape

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
        window = int(len(self.trace.x)/2)
        self.ax.plot(self.trace.x[:window], self.trace.y[:window], color=[.8, .2, .2], linewidth=2)
        self.ax.plot(self.trace.x[window:], self.trace.y[window:], color=[.2, .2, .8], linewidth=2)

        names = self.intersections.keys()
        colors = [[.2, .5, .5], [.4, .6, .2], [.1, .1, .4], [.3, .8, .2]]
        colors = dict(zip(names, colors))
        # Plot intersection points
        for name, timepoints in self.intersections.items():
            # print('Found {} intersection with {}'.format(len(timepoints), name))
            if len(timepoints):
                points = [c for idx, c in enumerate(zip(self.trace.x, self.trace.y)) if idx in timepoints]
                self.ax.plot([point[0] for point in points], [point[1] for point in points], 'o', markersize=7,
                             color=colors[name])





