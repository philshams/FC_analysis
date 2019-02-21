"""
Script to set up the analysis of behavioural data: defines relevant parameters and flags to control
the different functionalities of the analysis scripts
"""

#######################
#  SETUP              #
#######################
datalog = 'D:\\data\\experiments.xlsx'
save_folder = 'D:\\data'
DLC_folder = 'C:\\Drive\\Behaviour\\DeepLabCut'

# If loading a pre-existing database, specify name of the file it was saved to. otherwise db is generated from
# scratch from the datalog.csv file
load_database = False  # This is False only if you are creating a new database
update_database = True # add recently added sessions to database

load_name = '190221'  # name of file to load
save_name = '190221' # name to save the results of the analysis

# selects which sessions to analyze ('all', 'experiment', 'session', 'date', 'mouse' or COHORT)
selector_type = 'number'
selector = [45] #3380 is broken

# selector_type = 'all'


#######################
#  TRACKING           #
#######################
track_mouse = True             # Run tracking
track_options = {
    'register arena': True, # Register arena to model arena for tracking
    'analyze wall': True,
    'save stimulus clips': True, # Save videos of the peri-stimulus period

    'track whole session': True,  # Track the mouse for the entire session using DLC

    'run DLC': False, # Use DLC to analyze the raw videos

    'do not overwrite': False,
    }

fisheye_map_location = 'C:\\Drive\\DLC\\transforms\\fisheye_maps.npy'



video_analysis_settings = {
    'fast track wndw pre': 2,
    'fast track wndw post': 9.5
}

#######################
#  DLC                #
#######################
dlc_config_settings = {
    'clips_folder': 'D:\\data\\Analysis_Videos',
    'dlc_network_posecfg': 'D:\\data\\DLC_nets\\Barnes-Philip-2018-11-22\\dlc-models\\iteration-6\\Barnes2018-11-22-trainset95shuffle1\\test',
    'dlc_network_snapshot': 'D:\\data\\DLC_nets\\Barnes-Philip-2018-11-22\\dlc-models\\iteration-6\\Barnes2018-11-22-trainset95shuffle1\\train\\snapshot-950000',
    'scorer': 'DeepCut_resnet50_Philip_50000',
    'config_file': 'D:\\data\\DLC_nets\\Barnes-Philip-2018-11-22\\config.yaml',
    'body parts': ['nose','L eye','R eye','L ear','neck','R ear','L shoulder','upper back','R shoulder','L hind limb','Lower back','R hind limb','derriere'],
    'inverse_fisheye_map_location': 'C:\\Drive\\DLC\\transforms\\inverse_fisheye_maps.npy'
}

























#######################
#  PROCESSING         #
#######################
"""
Process tracking data (e.g. extract velocity trace from coordinates)
"""
processing = False
processing_options = {
    'cfg': './Processing/processing_cfg.yml'
}

#######################
#  DEBUGGING          #
#######################
debug = False  # If true runs a gui to examine tracking data

#######################
#  COHORT #
#######################
"""
Cohort gives the option to pool the data from all the sessions analysed for group analysis
"""
cohort = False                       # make a cohort or process existing one
cohort_options = {
    'name': 'test_cohort',     # Name of the cohort
    'selector type': 'date',     # what to select the sessions to pool by [e.g. by experiment, by date...]
    'selector': ['180921'],       # actual values to select by {e.g. session ID number]
    'data to pool': ['tracking']    # what data from the sessions you want to pool in the cohort (e.g. tracking)
}


#######################
#  PLOTTING           #
#######################
"""
Plotting still needs to be implemented 
"""
plotting = False
plotting_options = {
    'cfg': './Plotting/plotting_cfg.yml'
}


#######################
#  MISC.              #
#######################

###more CONFIGURATION




# 'arena_roi': 'Barnes',
# 'preview': False,
# 'stopframe': -1,
# 'th_scaling': 0.6,
# 'tail_th_scaling': 0.225,
# 'num mice': 1,
# 'min ctn area': 200,
# 'max cnt area': 5000,


#extract_rois_background = False  # If 'bg get rois' = True, set this to True to manually extract the rois from bg
# 'bg get rois': False,          # allow user to define 3 ROIs when extracting background [threat, shelter variable]
# 'track_exploration': False,    # Track the mouse during the exploration using the standard tracking
# 'track_mouse_fast': True,      # if true only track segments of videos around the stimuli
# 'use_stdtracking': False,       # Use standard tracking (written by FC)
# 'stdtracking_justCoM': True,   # When using the standard tracking just extract the Centre of Mass and
# not other variables [e.g. orientation]. This is TRUE by default
# 'use_deeplabcut': True,        # Use deepLabCut to track the mouse
# configure yaml files for std and dlc tracking

# exp_type = 'Barnes'