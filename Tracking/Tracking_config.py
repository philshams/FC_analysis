"""
Parameters for std tracking
{
    'Experiment name': Name of the experiment --> used to select the settings based on experiment_type setting in config.py
    'arena_roi: string --> name of the 3d ROI specified in background extraction
    ' preview': Bool --> Display online the results of the tracking. For debugging
    'stop frame': int --> frame at which to stop the tracking. -1 = don't stop
    'th_scaling': float --> changes threshold
    'tail_th_scaling: float --> changes threshold for tail detection (to extract mouse orientation)
    'num mice: int --> number of mice being tracked
    'min cnt area': int --> minimal area of a mouse
    'max cnt area': int --> max area covered by a mouse
    'fast track wnd': int --> number of seconds before and after stimulus to track the mouse for when fast stracking
}


Parameters for dlc tracking
{
    'clips': None --> empty container for extracted cliips
    'clips_folder': path --> where to save clips
    'dlc_network_path': path --> need to check what this does
    'dlc_network_snapshot': where the trained network is
    'scorer': stirng --> name of the scorer
    'store trial videos': Bool --> save the trial videos as .avi if True. 
}


"""


maze_config = {'arena_roi':'short_bridge', 'preview':False, 'stopframe':-1, 'th_scaling':0.6, 'tail_th_scaling':0.225,
               'num mice':1, 'min ctn area':200, 'max cnt area':5000, 'fast track wnd':60}

fearcond_config = {'arena_roi':'arena_floor', 'preview':False, 'stopframe':-1, 'th_scaling':0.75, 'tail_th_scaling':0.35,
               'num mice':1, 'min ctn area':200, 'max cnt area':5000, 'fast track wnd':60}

# Deep Lab Cut config
dlc_config_settings = {
    'clips': {'visual': {}, 'audio': {},'digital': {}},
    'clips_folder': 'C:\\Users\\Federico\\Documents\\GitHub\\DeepLabCut\\videos',
    'dlc_network_path': 'C:\\Users\\Federico\\Documents\\GitHub\\BrancoLab_RandomCode\\Analysis_V2\\Tracking',
    'dlc_network_snapshot': 'C:\\Users\\Federico\\Documents\\GitHub\\DeepLabCut\\pose-tensorflow\\models\\mazeAug01-trainset95shuffle1\\train\\snapshot-350000',
    'scorer': 'DeepCut_resnet50_FC_50000',
    'store trial videos': True
}
