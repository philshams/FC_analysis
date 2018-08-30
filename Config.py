import platform

"""
Script to set up the analysis of behavioural data: defines relevant parameters and flags to control
the different functionalities of the analysis scripts
"""
# If loading a pre-existant database, specify name of the .h5 file it was saved to. otherwise db is generated from
# scratch from the datalog.csv file
load_database = True  # This is False only if you are creating a new database, if you are working on a pre-existing
# database it will be set as True

# Specify if you want to update the database loading info from datalog.csv to add recently added sessions to database
update_database = False
load_name = 'SquareMazeAllSessions_completed'

# Specify name with which to save the results of the analysis
save_name = 'SquareMazeAllSessions'

"""
Specify set of sessions to analyse 
Selector_type [what to select the sessions by] can be:
    all --> will process all sessions in the datbase
    experiment --> will only process sessions with given experiment tag (strings)
    session --> will only process sessions with given session ids (integers)
    'date' --> will only process sessions with give date (strings)
    
Selector [e.g. the specific date selected] depends on the type being used,
 it specify the values by which the sessions will be selected

Experiment_type specifies which kind of experiment we are analysing (e.g. maze or fearcond), some scripts
might behave differently depending on the type of experiment


"""

selector_type = 'session'  # selects which session to an 'new', 'experiment', 'session', 'date'
selector = [74]  # ['180603', '180604', '180605', '180606', '180607', '180625', '180626', '080628']  # [74, 75, 78, 79, 80, 81]  #
exp_type = 'maze'


"""
Flags and params that control the execution of the different parts of the code
"""
#######################
#  TRACKING           #
#######################
# analysis start frame: beginning of videos is usually empty, we can skip that
startf = 8000  # Skip the first n frames of the first video when tracking [

extract_background = False

track_mouse = False             # <----- !!!!!!!!!!
track_options = {
    'bg get rois': True,          # allow user to define 3 ROIs when extracting background [threat, shelter variable]
    'track whole session': False,  # Track the mouse for the entire session
    'track_exploration': True,    # Track the mouse during the exploration using the standard tracking
    'track_mouse_fast': True,      # if true only track segments of videos around the stimuli
    'use_stdtracking': True,       # Use standard tracking (written by FC)
    'stdtracking_justCoM': True,   # When using the standard tracking just extract the Centre of Mass and
                                   # not other variables [e.g. orientation]. This is TRUE by default
    'use_deeplabcut': True,        # Use deepLabCut to track the mouse
    'cfg_std': 'C:\\Users\\Federico\\Documents\\GitHub\\FC_analysis\\Tracking\\Configs\\cfg_std_maze.yml',
    # configure yaml files for std and dlc tracking
    'cfg_dlc': 'C:\\Users\\Federico\\Documents\\GitHub\\FC_analysis\\Tracking\\Configs\\cfg_dlc_maze.yml'
    }


#######################
#  PROCESSING         #
#######################
processing = RecursionError
processing_options = {
    'cfg': 'C:\\Users\\Federico\\Documents\\GitHub\\FC_analysis\\Processing\\processing_cfg.yml'
}

#######################
#  DEBUGGING          #
#######################
debug = False

#######################
#  COHORT #
#######################
"""
Cohort gives the option to pool the data from all the sessions analysed for group analysis
"""
cohort = False
cohort_options = {
    'name': 'CH_ThreeSessions',   # Name of the cohort
    'selector type': 'session',  # what to select the sessions to pool by [e.g. by experiment, by date...]
    'selector': [62, 63, 64],  # actual values to select by {e.g. session ID number]
    'data to pool': ['tracking']  # what data from the sessions you want to pool in the cohort (e.g. tracking)
}


#######################
#  PLOTTING           #
#######################
"""
Plotting still needs to be implemented 
"""
plotting = True
plotting_individuals = False
plotting_cohort = True


#######################
#  MISC.              #
#######################
"""
For each sessions bein processed save a .avi video of each peri-stimulus period
"""
get_trials_clips = False


#######################
#  PATHS              #
#######################
""" where to find the Datalog.csv file, where to save the results of the analysis"""

if platform.system() == 'Windows':
    datalog_path = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis\\datalog.xls'
    savelogpath = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis'
else:
    savelogpath = '/Users/federicoclaudi/Desktop'
    


#######################
#  MESSAGING          #
#######################
"""
Options to send slack message, slack channel messages or emails with progress of the analysis and results
"""
send_messages = True
slack_username = 'U9ES1UXSM'
slack_env_var_token = 'SLACK_BRANCO_TOKEN'


