from Utils import Image_processing
from Utils.loadsave_funcs import save_data, load_data, load_paths
from Utils.Setup_funcs import get_sessions_metadata_from_yaml, get_session_videodata, generate_database_from_metadatas
from Plotting import Plotting_main
from Tracking.Tracking_main import Tracking
from Utils.Data_rearrange_funcs import create_cohort
from Processing import Processing_main

from Config import load_database, update_database, load_name, save_name\
    , selector_type, selector,\
    extract_background, track_mouse, track_options, \
    plotting, cohort, processing


# The analysis is all contained within this class
class Analysis():
    def __init__(self):
        """
        Set up the analysis:
        - get global vars
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

        # Load the database
        self.load_database()

        # Call the main funct that orchestrates the application of the subprocesses
        self.main()

        # Save the final results
        self.save_results(obj=self.db, mod='_completed')

    def load_database(self):
        """
        Take care of creating a new database from scratch, using the metadata in the datalog.csv file, or
        loading a pre-existing database
        """

        # Load database
        if load_database:
            db = load_data(self.save_fld, load_name)
            # Update database with recently added sessions
            if update_database:
                db.sessions = get_sessions_metadata_from_yaml(self.datalog_path, database=db)
        else:
            # Create database from scratch
            sessions_metadata = get_sessions_metadata_from_yaml(self.datalog_path)
            self.db = generate_database_from_metadatas(sessions_metadata)

    def main(self):
        """"
        Once all is set up we apply sub-processes to individual sessions or cohorts
        """
        ###############################################################################################################
        ## WORK ON SINGLE COHORTS
        ###############################################################################################################
        # Loop over all the sessions and call all the relevant functions [ For the sessions that need to be processed]
        for session_name in sorted(self.db.index):
            session = self.db.loc[session_name]
            # Check if this is one of th sessions we should be processing
            if not selector_type == 'any':
                if selector_type == 'experiment' and session.Metadata.experiment not in selector:
                    continue
                elif selector_type == 'date' and str(session.Metadata.date) not in selector:
                    continue
                elif selector_type == 'session' and session.Metadata.session_id not in selector:
                    continue

            print('---------------\nProcessing session {}'.format(session_name))

            if extract_background or track_mouse:
                self.video_analysis(session)

            if processing:
                self.processing_analysis(session)

            if plotting:  # individuals - work in progress
                self.plotting_analysis(session)

            if cohort:
                self.cohort_analysis()
                if plotting:
                    Plotting_main.setup_plotting(None, self.db, selector='cohort')

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
                tracked = Tracking(session, self.db, self.TF_setup, self.TF_sttings)
                self.db = tracked.database
                self.TF_setup = tracked.TF_setup
                self.TF_sttings = tracked.TF_settings

                self.save_results(obj=self.db, mod='_tracking')

    def processing_analysis(self, session):
            # Processing
            Processing_main.Processing(session, self.db)

            self.save_results(obj=self.db, mod='_processing')

    def plotting_analysis(self, session):
            # Plot for individual mice
            Plotting_main.setup_plotting(session, self.db)

    def cohort_analysis(self):
        # Create a cohort and store it in database
        self.db = create_cohort(self.db)  # Get all the trial data in one place
        self.save_results(obj=self.db, mod='_cohort')

    def save_results(self, obj=None, mod=None):
        save_data(self.save_fld, save_name, object=obj, name_modifier=mod)


#######################
#  START              #
#######################
if __name__ == "__main__":
    Analysis()



