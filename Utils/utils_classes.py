from collections import namedtuple


class Session_metadata():
    """  class to store metadata for a session """
    def __init__(self):
        self.session_id = None
        self.experiment = None
        self.date = None
        self.mouse_id = None
        self.video_file_paths = []
        self.tdms_file_paths = []
        self.trials_id_list = []
        self.software = None
        self.videodata = []
        self.number = []

    def __repr__(self):
        return 'Session {} Metadata'.format(self.session_id)

    def __str__(self):
        return 'Session {} Metadata'.format(self.session_id)

class Session_stimuli():
    """  class to store metadata for a session """
    def __init__(self):
        self.session_id = None
        self.stimuli = {'visual': [], 'audio': [], 'digital': []}

    def __repr__(self):
        return 'Session {} Stimuli'.format(self.session_id)

    def __str__(self):
        return 'Session {} Stimuli'.format(self.session_id)

class DataBase():
    """ Class to initialise an empty database """
    def __init__(self):
        # self.sessions = {'Metadata': None, 'Tracking': None}
        self.sessions = namedtuple('sessions', 'Number Metadata Tracking Registration Stimuli')


class Cohort():
    """ Class to initialise empty cohorts """
    def __init__(self):
        self.name = None
        self.metadata = namedtuple('metadata', 'created selector_type selector sessions_in_cohort')
        self.tracking_data = namedtuple('tracking_data', 'explorations whole_sessions trials')

    def __repr__(self):
        return 'Cohort {}'.format(self.name)

    def __str__(self):
        return 'Cohort {}'.format(self.name)



