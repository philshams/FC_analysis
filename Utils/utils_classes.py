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
        self.sessions = {'Metadata': None, 'Video': None, 'Tracking': None}


class Trial():
    def __init__(self):
        self.name = None
        self.id = None
        self.metadata = None
        self.std_tracking = {'x': None, 'y': None, 'orientation': None, 'direction': None, 'velocity': None}
        self.dlc_tracking = {}


class All_trials():
    def __init__(self):
        self.visual = []
        self.audio = []
        self.digital = []


class Cohort():
    def __init__(self):
        self.name = None
        self.metadata = {
            'created':None,
            'selector type':None,
            'selector':None,
            'sessions in cohort':None
        }
        self.tracking_data = {'explorations': [], 'whole sessions': [], 'trials': {}}


