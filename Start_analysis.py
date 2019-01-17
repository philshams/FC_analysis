from Config import load_database, update_database, load_name, save_name, selector_type, selector, plotting_options, \
    track_mouse, track_options, plotting, cohort, processing, debug, datalog, save_folder

from tqdm import tqdm
import sys
import warnings
import os
from termcolor import colored

from Utils import video_funcs
from Utils.loadsave_funcs import save_data, load_data, load_paths, load_yaml
from Utils.Setup_funcs import create_database
from Utils.Data_rearrange_funcs import create_cohort, check_session_selected
import warnings
warnings.filterwarnings('ignore')

# from Processing import Processing_main

# from Plotting import Single_trial_summary


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
        self.save_fld = save_folder
        self.datalog_path = datalog

        # Flags to keep track if TF is set up for DLC analysis
        self.TF_setup = False
        self.TF_settings = None

        # Load the database
        self.load_database()
        # to see all entries: self.db
        # to drop an entry: self.db = self.db.drop(['desired_entry_to_drop'])

        # Call the main func that orchestrates the application of the processes
        self.main()

        # Save the final results
        self.save_results(obj=self.db, mod='')

        # Close everything
        sys.exit()

    def main(self):
        """"
        Once all is set up we apply sub-processes to individual sessions
        """
        # TRACK SINGLE SESSIONS

        # Loop over all the sessions - Tracking
        if track_mouse:
            # other_db = load_data(self.save_fld, '181107')

            for session_name in self.db.index[::-1]:
                session = self.db.loc[session_name]

                # add info from other database to this database
                # other_session = other_db.loc[session_name]
                # session['Metadata'].videodata[0]['Arena Transformation'] = other_session['Metadata'].videodata[0]['Arena Transformation']
                # session['Metadata'].videodata[0]['Background'] = other_session['Metadata'].videodata[0]['Background']

                # Check if this is one of the sessions we should be processing
                selected = check_session_selected(session.Metadata, selector_type, selector)
                if selected:
                    print(colored('Tracking session: {}'.format(session_name), 'green'))
                    from Tracking.Tracking_main import Tracking
                    Tracking(session, self.TF_setup, self.TF_settings).session
                    self.db.loc[session_name]['Tracking'] = session['Tracking']
                    self.save_results(obj=self.db, mod='')

            self.save_results(obj=self.db, mod='')

        # PROCESS SINGLE SESSIONS
        if processing or plotting:
            # Loop over all the sessions - Other processes
            for session_name in tqdm(sorted(self.db.index)):
                session = self.db.loc[session_name]
                selected = check_session_selected(session.Metadata, selector_type, selector)
                if selected:
                    print(colored('---------------\nProcessing session {}'.format(session_name), 'green', attrs=['bold']))

                    if processing:
                        Processing_main.Processing(session, self.db)

                    if plotting:
                        # plotting_settings = load_yaml(plotting_options['cfg'])
                        Single_trial_summary.Plotter(session)

            self.save_results(obj=self.db, mod='')


        if debug:
            from Debug.Visualise_tracking import App
            sessions = {}
            for session_name in tqdm(sorted(self.db.index)):
                session = self.db.loc[session_name]
                selected = check_session_selected(session.Metadata, selector_type, selector)
                if selected:
                    sessions[session_name] = session

            app = App(sessions)

        return

########################################################################################################################
    # ANALYSIS
    # def video_analysis(self, session):
    #     """ EXTRACT useful information from the videos for one session"""
    #     # Process background: get maze edges and user selected ROIs
    #     # maze_edges, user_rois = video_funcs.process_background(session['Metadata'].videodata[0]['Background'],
    #     #                                                             track_options)
    #     # session.Metadata.videodata[0]['Maze Edges'] = maze_edges
    #     # session.Metadata.videodata[0]['User ROIs'] = user_rois
    #
    #     # Tracking
    #     if track_mouse:

    # def processing_session(self, session):

    # def plotting_session(self, session):



########################################################################################################################
    # LOADING AND SAVING
    def save_results(self, obj=None, mod=''):
        """ calls savedata to handle saving database to file """
        save_data(self.save_fld, load_name, save_name, self.loaded_database_size, object=obj, name_modifier=mod)

    def load_database(self):
        """
        Take care of creating a new database from scratch, using the metadata in the datalog.csv file unless the
        user wants to load a database from a pre-existing file
        """
        if load_database:
            self.db = load_data(self.save_fld, load_name)

            if update_database:  # add new sessions from datalog.csv
                self.db = create_database(self.datalog_path, database=self.db)

            path = os.path.join(self.save_fld, load_name)
            self.loaded_database_size = os.path.getsize(path)
            print(colored('Loaded database from: {} with size {}\n'.format(load_name, self.loaded_database_size),
                          'yellow'))

        else:  # Create database from scratch
            self.db = create_database(self.datalog_path)

        self.save_results(obj=self.db, mod='')
        self.save_results(obj=self.db,mod='_backup')

    def print_planned_processing(self):
        """ When starting a new run, print the options specified in Config.py for the user to check """
        import json

        if load_database:
            print(colored('\nLoading database: {}\n'.format(load_name),'blue'))
        else:
            print(colored('\nCreating new database: {}\n'.format(load_name), 'blue'))
        # if update_database and load_database:
        #     print(colored('Updating database'.format(load_name), 'blue'))
        if selector_type == 'all':
            print(colored('Analyzing all sessions', 'blue'))
        else:
            print(colored('Selector type: {}\nSelector: {}'.format(selector_type, selector), 'blue'))

        # print(colored('Processing: {}\nPlotting: {}\nTracking: {}\n{}'
        #     .format(processing, plotting,track_mouse,json.dumps(track_options, indent=3)),'blue'))


#  START
if __name__ == "__main__":
    Analysis()



