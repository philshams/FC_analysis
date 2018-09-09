#### USAGE ####

The code aims to unify all analysis [tracking, stats, plotting, modelling] for all experiments into one unified structure
to facilitate both the storage/usage of data and the usage of the analysis code itself.
To do so, the information from all the experiments done by the user is loaded from a "datalog.csv" file containing a list of
all the sessions [See: datalog section]. This list is used to create a database containing the info about all sessions
[e.g. experiment ID, mouse ID, date ...]. The results for any analysis performed on each session are stored under
that session's entry in the database. For example the results for a session's tracking are stored under Database > Session
> Tracking [for more info: Tracking section].

In the Config.py file, it is possible to select a subsection of sessions to process by specifying the selection criterion (e.g. by date, ID..)
and the selection tag (e.g. the specific date). In the same file the user can also select which processes to apply (e.g.
tracking, plotting...)

Once all the settings are correctly set in the Config.py, simply run Start_analysis.py to start the analysis process.

Usage steps:
    * Create datalog.csv if necessary
    * Specify which steps to run in Config.py
    * run Start_analysis.py


### Datalog ####
The datalog.csv file contains info about all the sessions that will be loaded into the database.
Each session can be one or more lines in the .csv file: each line corresponds to each video taken during the session.
For each line one must specify the session ID, date, experiment type, mouse ID, stim frames and video file path.
This info is then used to create the a entry for each session into the database. All this info is then stored into
session > metadata.


#### Database ####
The database containing all the sessions is constructed by loading the info from a .csv file [see below: datalog section].
It is then filled in accordingly


Database:
        . Session1:
                * Metadata [date, experiment, mouse ID, stims, ...]
                * Tracking:
                         - Trial 1: _ metadata (trial) [name, session, video, ...]
                                    _ std tracking
                                    _ dlc tracking
                                    _ ...
                         - Trial 2
                         - Trial 3
                         ...
        . Session2:
                * Metadata
                * Tracking:
                         - Trial 1
                         - Trial 2
                         - Trial 3
                         ...
         ...

        . Cohort: - All trials [list of all trials]


### Tracking ###
Tracking can be done in different ways, how the tracking will be done is specified in the Config.py file.

One option is to extract the background from each video. If this is true the code will load each video [for the sessions
being processed] and ask the user to select 3 ROIs. If one doesn't need ROIs there should be an option to skip that.
At the moment the background is always the first frame of the first video of each session as in my experiments that's
always empty. If one wishes to get the background for multiple videos please set all the other processing steps (tracking,
plotting...) to false so that the background extraction will take place much more quickly.

The tracking options are:
    * track_exploration: track the mouse using Fede's tracking (std tracking) between the start of the first video and
                    the time of the first stimulus
    * track whole session: track the mouse using std tracking fro the entire session across all videos
    * stdtracking_justCoM: the std tracking will only extract the mouse's centre of mass and not other stuff
                    like orientation, velocity...
    * track_mouse_fast: track only the peri-stimulus times (either with std tracking or deeplabcut [dlc] tracking).
                     This extracts 2 minutes clips around the stims trials and only tracks that
    * use_stdtracking: when tracking the trials in track_mouse_fast mode, use std tracking
    * use_dlctracking: when tracking the trials in track_mouse_fast mode, use dlc tracking


The tracking data are saved in Session>Tracking. This will contain > Exploration, > Whole Session, > Trials.
For each of these entries and for each trial there will be a > metadata containing all relevant info (e.g. session,
stim time ... ). For the trials there will be further entries like > std_tracking, > dlc_tracking ... containing the
tracking data.


!!! The behaviour of the tracking functions needs to be tuned to the specific experiment being processed !!!
This is done by setting the params in "Tracking_config.py". This contains the setting for both std and dlc tracking.
It is possible to store settings for different types of experiment in the same location, the correct set will be loaded
according the the experiment type specified in Config.py. [see Tracking_config.py for an explanation of what the
params do].



