"""
Script to set up the analysis of behavioural data: defines relevant parameters and flags to control
the different functionalities of the analysis scripts
"""

verbose= True   # Display details of on-going processes (e.g. execution times..)


# If loading a pre-existing database, specify name of the file it was saved to. otherwise db is generated from
# scratch from the datalog.csv file
load_database = False  # This is False only if you are creating a new database

# Specify if you want to update the database to add recently added sessions to database
update_database = False
load_name = '181002_safe'  # name of file to load

# Specify name with which to save the results of the analysis
save_name = '181012'

"""
Specify set of sessions to analyse 
Selector_type [what to select the sessions by] can be:
    all --> will process all sessions in the database
    experiment --> will only process sessions with given experiment tag (strings)
    session --> will only process sessions with given session ids (integers)
    'date' --> will only process sessions with give date (strings)
    
Selector [e.g. the specific date selected] depends on the type being used,
 it specify the values by which the sessions will be selected

Experiment_type specifies which kind of experiment we are analysing (e.g. maze or fearcond), some scripts
might behave differently depending on the type of experiment
"""
# TODO add "new" to selector type
selector_type = 'experiment'  # selects which session to an 'new', 'experiment', 'session', 'date' or COHORT to
selector = ['Barnes US wall up']
exp_type = 'Barnes'


"""
Flags and params that control the execution of the different parts of the code
"""
#######################
#  TRACKING           #
#######################
startf = 100  # Skip the first n frames of the first video when tracking

extract_rois_background = False  # If 'bg get rois' = True, set this to True to manually extract the rois from bg

track_mouse = True             # <----- !!!!!!!!!!  Run tracking
track_options = {
    'bg get rois': False,          # allow user to define 3 ROIs when extracting background [threat, shelter variable]
    'track whole session': False,  # Track the mouse for the entire session using the standard tracking
    'track_exploration': False,    # Track the mouse during the exploration using the standard tracking
    'track_mouse_fast': True,      # if true only track segments of videos around the stimuli
    'use_stdtracking': False,       # Use standard tracking (written by FC)
    'stdtracking_justCoM': True,   # When using the standard tracking just extract the Centre of Mass and
                                   # not other variables [e.g. orientation]. This is TRUE by default
    'use_deeplabcut': True,        # Use deepLabCut to track the mouse
    # configure yaml files for std and dlc tracking
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
debug = True  # If true runs a gui to debug tracking data

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
get_trials_clips = True


#######################
#  MESSAGING          #
#######################
"""
Options to send slack message, slack channel messages or emails with progress of the analysis and results
"""
send_messages = False
slack_username = 'U9ES1UXSM'
slack_env_var_token = 'SLACK_BRANCO_TOKEN'


