from Utils import Image_processing
from Utils.loadsave_funcs import save_data, load_data, load_paths
from Utils.Setup_funcs import get_sessions_metadata_from_yaml, get_session_videodata, generate_database_from_metadatas
from Plotting import Plotting_main
from Tracking.Tracking_main import Tracking
from Utils.Data_rearrange_funcs import create_cohort, check_session_selected
from Processing import Processing_main

from Config import load_database, update_database, load_name, save_name\
    , selector_type, selector,\
    extract_background, track_mouse, track_options, \
    plotting, cohort, processing


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
        # Get path used throughout the analysis
        paths = load_paths()
        self.save_fld = paths['save fld']
        self.datalog_path = paths['datalog']

        # Flags to keep track if TF is set up for DLC analysis
        self.TF_setup = False
        self.TF_sttings = None
        self.clips_l = []

        # Load the database
        self.load_database()

        # Call the main funct that orchestrates the application of the subprocesses
        self.main()

        # Save the final results
        self.save_results(obj=self.db, mod='_completed')

    def main(self):
        """"
        Once all is set up we apply sub-processes to individual sessions or cohorts
        """
        ################################################
        ## WORK ON SINGLE COHORTS
        ################################################
        if not selector_type == 'cohort':
            # Loop over all the sessions
            for session_name in sorted(self.db.index):
                session = self.db.loc[session_name]
                # Check if this is one of th sessions we should be processing
                selected = check_session_selected(session.Metadata, selector_type, selector)
                if not selected:
                    continue

                # Process the session, appply the selected subprocesses
                print('---------------\nProcessing session {}'.format(session_name))

                # TRACKING #######################################
                if extract_background or track_mouse:
                    self.video_analysis(session)

                # PROCESSING
                if processing:
                    self.processing_session(session)

                # PLOTTING INDIVIDUAL
                if plotting:  # individuals - work in progress
                    self.plotting_session(session)

            # Finish DLC tracking [extract pose on saved clips]
            self.db = Tracking.tracking_use_dlc(self.db, self.clips_l)

        ################################################
        ## WORK ON COHORTS
        ################################################
        # COHORT
        if cohort:
            self.cohort_analysis()
            if plotting:
                Plotting_main.setup_plotting(None, self.db, selector='cohort')

########################################################################################################################
########################################################################################################################
    # WORK ON SINGLE SESSIONS
    def video_analysis(self, session):
        # extract info from sessions videos [e.g. first frame, fps, videos lenth...]
        session = get_session_videodata(session)

        # Process background: get maze edges and user selected ROIs
        if extract_background:
            # Get bg and save
            maze_edges, user_rois = Image_processing.process_background(session['Video']['Background'],
                                                                        track_options)
            session['Video']['Maze Edges'] = maze_edges
            session['Video']['User ROIs'] = user_rois

            self.save_results(obj=self.db, mod='_bg')

        # Track animal on videos   <---!!!!
        if track_mouse:
            tracked = Tracking(session, self.db, self.TF_setup, self.TF_sttings, self.clips_l)
            self.db = tracked.database
            self.TF_setup = tracked.TF_setup
            self.TF_sttings = tracked.TF_settings
            self.clips_l = tracked.clips_l

            self.save_results(obj=self.db, mod='_tracking')

    def processing_session(self, session):
            # Processing
            Processing_main.Processing(session, self.db)

            self.save_results(obj=self.db, mod='_processing')

    def plotting_session(self, session):
            # Plot for individual mice
            Plotting_main.setup_plotting(session, self.db)

########################################################################################################################
########################################################################################################################
    # WORK ON COHORTS
    def cohort_analysis(self):
        # Create a cohort and store it in database
        self.db = create_cohort(self.db)  # Get all the trial data in one place
        self.save_results(obj=self.db, mod='_cohort')

########################################################################################################################
########################################################################################################################
    # LOADING AND SAVING
    def save_results(self, obj=None, mod=None):
        save_data(self.save_fld, save_name, object=obj, name_modifier=mod)

    def load_database(self):
        """
        Take care of creating a new database from scratch, using the metadata in the datalog.csv file, or
        loading a pre-existing database
        """

        # Load database
        if load_database:
            self.db = load_data(self.save_fld, load_name)
            # Update database with recently added sessions
            if update_database:
                self.db.sessions = get_sessions_metadata_from_yaml(self.datalog_path, database=self.db)
        else:
            # Create database from scratch
            sessions_metadata = get_sessions_metadata_from_yaml(self.datalog_path)
            self.db = generate_database_from_metadatas(sessions_metadata)

#######################
#  START              #
#######################
if __name__ == "__main__":
    Analysis()



