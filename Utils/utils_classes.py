from collections import namedtuple

class Session_metadata():
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


class DataBase():
    def __init__(self):
        # self.sessions = {'Metadata': None, 'Tracking': None}
        self.sessions = namedtuple('sessions', 'Metadata Tracking')

class Trial():
    def __init__(self):
        self.name = None
        self.id = None
        self.metadata = None
        # self.std_tracking = {'x': None, 'y': None, 'orientation': None, 'direction': None, 'velocity': None}
        self.std_tracing = namedtuple('std_tracking', 'x y orientation direction velocity')
        self.dlc_tracking = {}


class All_trials():
    def __init__(self):
        self.visual = []
        self.audio = []
        self.digital = []


class Cohort():
    def __init__(self):
        self.name = None
        self.metadata = namedtuple('metadata', 'created selector_type selector sessions_in_cohort')
        self.tracking_data = namedtuple('tracking_data', 'explorations whole_sessions trials')


        # self.metadata = {
        #     'created':None,
        #     'selector type':None,
        #     'selector':None,
        #     'sessions in cohort':None
        # }
        # self.tracking_data = {'explorations': [], 'whole sessions': [], 'trials': {}}


class Processing_class():
    def __init__(self):
        self.velocity = {}


