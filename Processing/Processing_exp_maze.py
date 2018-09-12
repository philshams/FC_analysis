import matplotlib.pyplot as plt  # used for debugging
import numpy as np
from collections import namedtuple


class ProcessingMaze():
    """ Applies processsing steps that are specific to maze experiments (e.g. get arm of escape) """

    # TODO create images-heatmaps of traces for plotting

    def __init__(self, session, debugging=True):
        self.session = session
        self.debugging = debugging

        # Subdivide frame
        self.rois, self.boundaries = self.subdivide_frame()

        # Process each trial
        tracking_items =self.session.Tracking.keys()
        if tracking_items:
            for item in tracking_items:
                retval = self.get_trial_traces(item)  # get coordinates

                if retval:
                    # self.get_intersections(item)
                    self.get_status_at_timepoint(item)  # get status at stim
                    self.get_origin_escape_arms(item)   # get escape arms

                    if debugging:                  # display results for debugging
                        self.debug_preview()
                        self.plot_trace()
                        plt.show()

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
            width = int(float(width/2))
            height = int(float(height/2))
            shelter_width = int(float(shelter_width/2))
            limits = boundaries(width, height, width-shelter_width, width+shelter_width)

        else:
            # Split the video using the user defined rois
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
        return rois, limits

    def get_trial_traces(self, trial_name):
        data = self.session.Tracking[trial_name]
        if isinstance(data, dict):
            # Doesnt work for wholetrace or exploration data, which are saved as dictionaries
            return False

        tracer = namedtuple('trace', 'x y')
        if 'Posture' in data.dlc_tracking.keys():
            # We have deeplabcut data
            self.trace = tracer(data.dlc_tracking['Posture']['body']['x'].values,
                           data.dlc_tracking['Posture']['body']['y'].values)
        elif data.std_tracking['x'] is not None:
            self.trace = tracer(data.std_tracking['x'],
                           data.std_tracking['y'])
        else:
            print('Could not find trial data for trial {}'.format(trial_name))
            return False

        # Get traces
        window = int(len(self.trace.x) / 2)
        tracer = namedtuple('trace', 'x y')
        self.prestim_trace = tracer(self.trace.x[:window], self.trace.y[:window])
        self.poststim_trace = tracer(self.trace.x[window:], self.trace.y[window:])
        self.shelter_to_stim, self.stim_to_shelter = self.get_leaveenter_shelter(tracer)

        # Store results
        if not 'processing' in data.__dict__.keys():
            setattr(data, 'processing', {})
        if not 'PlottingItems' in data.processing.keys():
            data.processing['PlottingItems'] = dict(prestim_trace=self.prestim_trace,
                                                    poststim_trace=self.poststim_trace,
                                                    shelt_to_stim_trace=self.shelter_to_stim,
                                                    stim_to_shelt_trace=self.stim_to_shelter)

        return True

    """
    Extract stuff
    """

    def get_leaveenter_shelter(self, tracer):
        def get_leave_enter_time(shelter, trace, pre=True):
            # Get the frames in which the mouse is in the shelter
            x_inshelt = np.where((shelter[0] < trace.x) & (trace.x < shelter[1]))[0]
            y_inshelt = np.where((shelter[2] < trace.y) & (trace.y < shelter[3]))[0]
            inshelt = np.intersect1d(x_inshelt, y_inshelt)

            if not len(inshelt):
                return False
            # If its before the stimulus we want the last frame, else we want the last frame
            if pre:
                if len(inshelt):  # means no point on the trace was in sheter
                    return inshelt[-1]
                else:
                    return  len(trace)-1
            else:
                if len(inshelt):
                    return inshelt[0]
                else:
                    return 0

        tostim, fromstim = False, False
        if 'Shelter' in self.rois.keys() and self.rois['Shelter'] is not None:
            shelter = self.rois['Shelter']  # x, with, y, height
            shelter = (shelter[0], shelter[0]+shelter[2], shelter[1], shelter[1]+shelter[3])  # x0, x1, y0, y1

            leaves_shelter = get_leave_enter_time(shelter, self.prestim_trace)
            enters_shelter = get_leave_enter_time(shelter, self.poststim_trace, pre=False)

            tostim = tracer(self.prestim_trace.x[leaves_shelter:], self.prestim_trace.y[leaves_shelter:])
            fromstim = tracer(self.poststim_trace.x[:enters_shelter+1], self.poststim_trace.y[:enters_shelter+1])

        return tostim, fromstim

    def get_status_at_timepoint(self, name, time: int = None, timename: str = 'stimulus'):
        """
        Get the status of the mouse [location, orientation...] at a specific timepoint.
        If not time is give the midpoint of the tracking traces is taken as stim time
        """
        if not 'session' in name.lower() or 'exploration' in name.lower:
            data = self.session.Tracking[name]

            if time is None:  # if a time is not give take the midpoint
                time = int(len(self.trace.x)/2)

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

        names = self.intersections.keys()
        colors = [[.2, .5, .5], [.4, .6, .2], [.1, .1, .4], [.3, .8, .2]]
        colors = dict(zip(names, colors))

