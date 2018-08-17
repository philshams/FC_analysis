import platform

"""
Script to set up the analysis of behavioural data: defines relevant parameters and flags to control
the different functionalities of the analysis scripts
"""
# Specify if you want to update the database loading info from datalog.csv
update_sessions_dict = True

# If loading a pre-existant database, specify name of the .h5 file it was saved to
load_database = True
load_name = 'DLCtest_completed'

# Specify name with which to save the results of the analysis
save_name = 'DLCtest'

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
selector = [48, 49, 50]  # ['180607', '180603', '180604', '180605', '180606']
exp_type = 'maze'


"""
Flags and params that control the execution of the different parts of the code
"""
#######################
#  IMG PROCESS        #
#######################
# analysis start frame: beginning of videos is usually empty, we can skip that
startf = 6000  # Skip the first n frames when tracking

extract_background = False
track_mouse = True             # <----- !!!!!!!!!!
track_options = {
    'track whole session': False,  # Track the mouse for the entire session
    'track_exploration': False,  # Track the mouse during the exploration using the standard tracking
    'track_mouse_fast': True,    # if true only track segments of videos around the stimuli
    'use_stdtracking': True,      # Use standard tracking (written by FC)
    'stdtracking_justCoM': True,  # When using the standard tracking just extract the Centre of Mass and
                                  # not other variables. This is TRUE by default
    'use_deeplabcut': True,       # Use deepLabCut to track the mouse
    'cfg_std': 'Tracking/Configs/cfg_std_fearcond.yml',
    'cfg_dlc': 'Tracking/Configs/cfg_std_fearcond.yml'
    }


#######################
#  COHORT #
#######################
"""
Cohort gives the otpion to pool the data from all the sessions analysed for group analysis
"""
Cohort = False


#######################
#  PLOTTING           #
#######################
"""
Plotting still needs to be implemented 
"""
plotting = False
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
    datalog_path = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis\\Experiments_log.xls'
    savelogpath = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis'
else:
    savelogpath = '/Users/federicoclaudi/Desktop'
    







