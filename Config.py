'''
USER-INPUTTED PARAMETERS, TO BE USED IN DOWNSTREAM SCRIPTS
'''

def setup():
    '''
    ........................................SETUP........................................
    '''
    excel_path = 'D:\\data\\experiments_paper.xlsx'
    save_folder = 'D:\\data'
    DLC_folder = 'C:\\Drive\\Behaviour\\DeepLabCut'

    load_database = True                      # This is False only if you are creating a new database
    update_database = False                  # Add recently added sessions or metadata to database

    load_name = '190625'                      # name of file to load
    save_name = '190625'                      # name to save the results of the analysis


    selector_type = 'experiment'                  # selects which sessions to analyze ('all', 'experiment', 'session', 'date', 'mouse', 'number')

    selector = [ 'Circle wall down (no baseline)' ]#,  'Circle wall down (dark)'] # 'Circle lights on off (baseline)','Circle void up',
    selector = ['Circle wall down']


    selector_type = 'number'                  # selects which sessions to analyze ('all', 'experiment', 'session', 'date', 'mouse', 'number')
    selector = [ x for x in range(156,165) ] #[65,94] ] # [49, 53] ]


    return excel_path, save_folder, DLC_folder, load_database, update_database, load_name, save_name, selector_type, selector



def tracking_options():
    '''
    .................................TRACKING & ANALYSIS OPTIONS.................................
    '''
    track_options = {

        'run DLC': True,                  # Use DLC to analyze the raw videos

        'register arena': True,            # Register arena to model arena for tracking

        'track session': False,              # Track the mouse for the entire session using DLC

        'parallel processes': 1 }          # Number of sessions to analyze simultaneously





    analysis_options = {

        'raw clips': False,                 # Save only raw videos of the peri-stimulus period

        'DLC clips': False,                 # Save videos of DLC model mouse in model arena

        'spontaneous homings': False,       # Analyze spontaneous homings

        'procedural': False,                # Analyze procedural learning

        'exploration': False,               # Analyze exploration

        'proper analysis': False,           # exploration, speed, reaction time

        'simulate': False,                   # Simulate each strategy

        'summary': False }                   # Make the moon-like summary plots


    fisheye_map_location = 'C:\\Drive\\DLC\\transforms\\fisheye_maps.npy'

    video_analysis_settings = {'seconds pre stimulus': 3,
                               'seconds post stimulus': 10 }

    return track_options, analysis_options, fisheye_map_location, video_analysis_settings



def dlc_options():
    '''
    ........................................DLC OPTIONS................................................
    '''
    dlc_config_settings = {
        'clips_folder': 'D:\\data\\Analysis',
        'dlc_network_posecfg': 'D:\\data\\DLC_nets\\Barnes-Philip-2018-11-22\\dlc-models\\iteration-9\\Barnes2018-11-22-trainset95shuffle1\\test',
        'dlc_network_snapshot': 'D:\\data\\DLC_nets\\Barnes-Philip-2018-11-22\\dlc-models\\iteration-9\\Barnes2018-11-22-trainset95shuffle1\\train\\snapshot-1000000',
        'scorer': 'DeepCut_resnet101_Philip_50000',
        'config_file': 'D:\\data\\DLC_nets\\Barnes-Philip-2018-11-22\\config.yaml',
        'body parts': ['nose','L eye','R eye','L ear','neck','R ear','L shoulder','upper back','R shoulder','L hind limb','Lower back','R hind limb','derriere'],
        'inverse_fisheye_map_location': 'C:\\Drive\\DLC\\transforms\\inverse_fisheye_maps.npy'
    }

    return dlc_config_settings


