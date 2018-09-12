from tqdm import tqdm
import sys
import warnings
import os

from Utils import Image_processing
from Utils.loadsave_funcs import save_data, load_data, load_paths
from Utils.Setup_funcs import create_database
from Utils.Data_rearrange_funcs import create_cohort, check_session_selected
from Utils.Messaging import slack_chat_messenger

from Tracking.Tracking_main import Tracking
from Processing import Processing_main
from Plotting import Single_trial_summary
from Debug.DebugTrack_GUI_Main import start_gui

from Config import load_database, update_database, load_name, save_name\
    , selector_type, selector,\
    extract_rois_background, track_mouse, track_options, \
    plotting, cohort, processing, debug, send_messages


########################################################################################################################
########################################################################################################################
# START #
# The analysis is all contained within this class
class Analysis():
    def __init__(self):
        """
        Set up the analysis:
        - get or define global vars
        - load database
        - call relevant functions [e.g. tracking, plotting...]
        """
        self.loaded_database_size = 0   # Get the size of the loaded database, check that the saving is going correctly

        # Show what we are planning to do
        self.print_planned_processing()

        # Get paths for data loading and saving
        paths = load_paths()
        self.save_fld = paths['save fld']
        self.datalog_path = paths['datalog']

        # Flags to keep track if TF is set up for DLC analysis
        self.TF_setup = False
        self.TF_settings = None
        self.clips_l = []

        # Load the database
        self.load_database()

        # Call the main func that orchestrates the application of the processes
        self.main()

        # Save the final results
        self.save_results(obj=self.db, mod='_completed')

        # Close everything
        sys.exit()

    def main(self):
        """"
        Once all is set up we apply sub-processes to individual sessions or cohorts
        """
        # TRACK SINGLE SESSIONS
        if not selector_type == 'cohort':
            # Loop over all the sessions - Tracking 
            if track_mouse or extract_rois_background:
                for session_name in tqdm(sorted(self.db.index)):
                    session = self.db.loc[session_name]
                    
                    # Check if this is one of the sessions we should be processing
                    selected = check_session_selected(session.Metadata, selector_type, selector)
                    if selected:
                        print('---------------\nTracking session {}'.format(session_name))

                        self.video_analysis(session) # <-- main tracking function

                if send_messages:
                    slack_chat_messenger('Finished STD tracking')

                if track_mouse:
                    # Finish DLC tracking [extract pose on saved clips]
                    try:
                        self.db = Tracking.tracking_use_dlc(self.db, self.clips_l)
                    except:
                        warnings.warn('Something went wront with DLC tracking ')

                    if send_messages:
                        slack_chat_messenger('Finished DLC tracking')
                self.save_results(obj=self.db, mod='_backupsave')

        # PROCESS SINGLE SESSIONS
        if processing or debug or plotting:
            # Loop over all the sessions - Other processes
            for session_name in tqdm(sorted(self.db.index)):
                session = self.db.loc[session_name]
                selected = check_session_selected(session.Metadata, selector_type, selector)
                if selected:
                    print('---------------\nProcessing session {}'.format(session_name))

                    if processing:
                        self.processing_session(session)

                    if debug:
                        start_gui(session)

                    if plotting:
                        self.plotting_session(session)
            if send_messages:
                slack_chat_messenger('Finished processing')

        # WORK ON COHORTS
        if cohort:
            self.cohort_analysis()

########################################################################################################################

    def video_analysis(self, session):
        """ EXTRACT useful information from the videos for one session"""
        # Process background: get maze edges and user selected ROIs
        if extract_rois_background:
            maze_edges, user_rois = Image_processing.process_background(session['Metadata'].videodata[0]['Background'],
                                                                        track_options)
            session.Metadata.videodata[0]['Maze Edges'] = maze_edges
            session.Metadata.videodata[0]['User ROIs'] = user_rois

        # Tracking
        if track_mouse:
            try:
                tracked = Tracking(session, self.TF_setup, self.TF_settings, self.clips_l)
                self.TF_setup = tracked.TF_setup
                self.TF_settings = tracked.TF_settings
                self.clips_l = tracked.clips_l
            except:
                warnings.warn('Something went wrong with tracking')
        self.save_results(obj=self.db, mod='_tracking')

    def processing_session(self, session):
            Processing_main.Processing(session, self.db)
            self.save_results(obj=self.db, mod='_processing')

    def plotting_session(self, session):
            Single_trial_summary.Plotter(session)


########################################################################################################################
    def cohort_analysis(self):
        # Create a cohort and store it in database
        self.db = create_cohort(self.db)  # Get all the trial data in one place
        self.save_results(obj=self.db, mod='_cohort')

########################################################################################################################
########################################################################################################################
    # LOADING AND SAVING
    def save_results(self, obj=None, mod=None):
        """ calse savedata to handle saving database to file """
        save_data(self.save_fld, save_name, self.loaded_database_size, object=obj, name_modifier=mod)

    def load_database(self):
        """
        Take care of creating a new database from scratch, using the metadata in the datalog.csv file unless the
        user wants to load a database from a pre-existing file
        """
        if load_database:
            self.db = load_data(self.save_fld, load_name)

            if update_database:  # add new sessions from datalog.csv
                self.db = create_database(self.datalog_path, database=self.db)

        else:  # Create database from scratch
            self.db = create_database(self.datalog_path)

        path = os.path.join(self.save_fld, load_name)
        self.loaded_database_size = os.path.getsize(path)
        print('Loaded database from file "{}" with size {}'.format(load_name, self.loaded_database_size))

        self.save_results(obj=self.db, mod='Metadata')

    def print_planned_processing(self):
        """ When starting a new run, print the options specified in Config.py for the user to check """
        import json
        print(""""
            Load database: {}
                update database: {}
                load name: {}
                save name: {}
                
            Selector type: {}
                selector:  {}
            
            Extract background: {}
            Tracking: {}
                track options: {}
                
            Processing: {}
            Plotting: {}
            Debug: {}
            
            Cohort; {}
            
            Send Messages: {}
        """.format(load_database, update_database, load_name, save_name,
                   selector_type, selector, extract_rois_background, track_mouse,
                   json.dumps(track_options, indent=30), plotting, cohort, processing, debug, send_messages))


#  START
if __name__ == "__main__":
    Analysis()



