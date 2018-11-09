"""
Script to set up the analysis of behavioural data: defines relevant parameters and flags to control
the different functionalities of the analysis scripts
"""

datalog = 'C:\Drive\DLC\data\experiments.xlsx'
save_folder = 'C:\Drive\DLC\data'
DLC_folder = 'C:\Drive\DLC\DeepLabCut'

# If loading a pre-existing database, specify name of the file it was saved to. otherwise db is generated from
# scratch from the datalog.csv file
load_database = True  # This is False only if you are creating a new database
update_database = True # add recently added sessions to database

load_name = '181107'  # name of file to load
save_name = '181107' # name to save the results of the analysis

# selects which sessions to analyze ('all', 'experiment', 'session', 'date' or COHORT)
# selector_type = 'all'
# selector = ['Barnes US wall down']
selector_type = 'all'
selector = ['Barnes US wall (20kHz)']



"""
Flags and params that control the execution of the different parts of the code
"""
#######################
#  TRACKING           #
#######################
startf = 100  # Skip the first n frames of the first video when tracking

track_mouse = True             # Run tracking
track_options = {
    'register arena': True, # Register arena to model arena for tracking
    'save stimulus clips': True, # Save videos of the peri-stimulus period

    'track whole session': False,  # Track the mouse for the entire session using the standard tracking
    'track stimulus responses': False, # Track the mouse during stimulus responses using DLC
    'use standard tracking': True, # Use standard tracking of center of mass instead of DLC

    'do not overwrite': True,
    'cfg_std': 'C:\\Drive\\DLC\\PNS_analysis\\Tracking\\Configs\\cfg_std_Barnes.yml',
    'cfg_dlc': 'C:\\Drive\\DLC\\PNS_analysis\\Tracking\\Configs\\cfg_dlc_Barnes.yml'
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
"""
For each session being processed save an .avi video of each peri-stimulus period
"""



###more CONFIGURATION
dlc_config_settings = {
    'clips': '',
    'clips_folder': 'C:\Drive\DLC\data\wall_data',
    'dlc_network_posecfg': 'C:\\Drive\\DLC\\data\\videos_train_dlc\\Barnes_USSep25_2018-trainset95shuffle1\\test',
    'dlc_network_snapshot': 'C:\\Drive\\DLC\\data\\videos_train_dlc\\Barnes_USSep25_2018-trainset95shuffle1\\train\\snapshot-50000',
    'scorer': 'DeepCut_resnet50_Philip_50000',
    'store trial videos': True
}

video_analysis_settings = {
    'fast track wndw pre': 5,
    'fast track wndw post': 10
}



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