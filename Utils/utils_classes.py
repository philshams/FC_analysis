class Session_metadata():
    def __init__(self):
        self.session_id = None
        self.experiment = None
        self.date = None
        self.mouse_id = None
        self.stimuli = {'visual': [], 'audio': [], 'digital': []}
        self.video_file_path = []
        self.trials_id_list = []


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


