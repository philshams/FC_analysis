from collections import namedtuple


class Session_metadata():
    """  class to store metadata for a session """
    def __init__(self):
        self.session_id = None
        self.experiment = None
        self.date = None
        self.mouse_id = None
        self.stimuli = {'visual': [], 'audio': [], 'digital': []}
        self.video_file_paths = []
        self.tdms_file_paths = []
        self.trials_id_list = []
        self.last_modified = None
        self.created = None
        self.software = None
        self.videodata = []

    def __repr__(self):
        return 'Session {} Metadata'.format(self.session_id)

    def __str__(self):
        return 'Session {} Metadata'.format(self.session_id)


class DataBase():
    """ Class to initialise an empty database """
    def __init__(self):
        # self.sessions = {'Metadata': None, 'Tracking': None}
        self.sessions = namedtuple('sessions', 'Metadata Tracking')


class Trial():
    """ Class to initialise and empty Trial object """
    def __init__(self):
        self.name = ''
        self.id = None
        self.metadata = None
        self.std_tracing = namedtuple('std_tracking', 'x y orientation direction velocity')
        self.dlc_tracking = {}

    def __repr__(self):
        return 'Trial: {}_{}'.format(self.name, self.id)

    def __str__(self):
        return 'Trial: {}'.format(self.name)


class Exploration():
    """ Class to initialise and empty Exploration object """
    def __init__(self):
        self.name = ''
        self.id = None
        self.metadata = {}
        self.processing = {}
        self.tracking = None

    def __repr__(self):
        return 'Exploration of sessoin {}'.format(self.name, self.id)

    def __str__(self):
        return 'Expl sess: {}'.format(self.name)


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



