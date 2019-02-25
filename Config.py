'''
USER-INPUTTED PARAMETERS, TO BE USED IN DOWNSTREAM SCRIPTS
'''

def setup():
    '''
    ........................................SETUP........................................
    '''
    excel_path = 'D:\\data\\experiments.xlsx'
    save_folder = 'D:\\data'
    DLC_folder = 'C:\\Drive\\Behaviour\\DeepLabCut'

    load_database = True        # This is False only if you are creating a new database
    update_database = False      # Add recently added sessions or metadata to database

    load_name = '190221'        # name of file to load
    save_name = '190221'        # name to save the results of the analysis

    selector_type = 'number'       # selects which sessions to analyze ('all', 'experiment', 'session', 'date', 'mouse', 'number')
    selector = [45]

    return excel_path, save_folder, DLC_folder, load_database, update_database, load_name, save_name, selector_type, selector



def tracking_options():
    '''
    .................................TRACKING & ANALYSIS OPTIONS.................................
    '''
    track_options = {

        'run DLC': False,                  # Use DLC to analyze the raw videos

        'register arena': True,            # Register arena to model arena for tracking

        'track session': True }            # Track the mouse for the entire session using DLC



    analysis_options = {

        'save stimulus clips': False,      # Save videos of the peri-stimulus period

        'DLC clips': True,                 # Save videos of DLC model mouse in model arena

        'sub-goal': False,                 # Analyze sub-goals

        'planning': False,                 # Analyze planning points

        'procedural': False }              # Analyze procedural learning



    fisheye_map_location = 'C:\\Drive\\DLC\\transforms\\fisheye_maps.npy'

    video_analysis_settings = {'seconds pre stimulus': 3,
                               'seconds post stimulus': 10 }

    return track_options, analysis_options, fisheye_map_location, video_analysis_settings



def dlc_options():
    '''
    ........................................DLC OPTIONS................................................
    '''
    dlc_config_settings = {
        'clips_folder': 'D:\\data\\Analysis_Videos',
        'dlc_network_posecfg': 'D:\\data\\DLC_nets\\Barnes-Philip-2018-11-22\\dlc-models\\iteration-6\\Barnes2018-11-22-trainset95shuffle1\\test',
        'dlc_network_snapshot': 'D:\\data\\DLC_nets\\Barnes-Philip-2018-11-22\\dlc-models\\iteration-6\\Barnes2018-11-22-trainset95shuffle1\\train\\snapshot-950000',
        'scorer': 'DeepCut_resnet50_Philip_50000',
        'config_file': 'D:\\data\\DLC_nets\\Barnes-Philip-2018-11-22\\config.yaml',
        'body parts': ['nose','L eye','R eye','L ear','neck','R ear','L shoulder','upper back','R shoulder','L hind limb','Lower back','R hind limb','derriere'],
        'inverse_fisheye_map_location': 'C:\\Drive\\DLC\\transforms\\inverse_fisheye_maps.npy'
    }

    return dlc_config_settings



