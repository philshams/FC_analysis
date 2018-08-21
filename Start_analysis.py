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
    plotting, Cohort, processing

# The analysis is all contained within this class
class Analysis():
    def __init__(self):
        # Get the paths for folder where to load and save stuff
        paths = load_paths()
        save_fld = paths['save fld']
        datalog_path = paths['datalog']

        # Keep track if TF is set up for DLC analysis
        self.TF_setup = False
        self.TF_sttings = None

        # Load database
        if load_database:
            db = load_data(save_fld, load_name)
            # Update database with recently added sessions
            if update_database:
                db.sessions = get_sessions_metadata_from_yaml(datalog_path, database=db)
        else:
            # Create database from scratch
            sessions_metadata = get_sessions_metadata_from_yaml(datalog_path)
            db = generate_database_from_metadatas(sessions_metadata)

        # Loop over all the sessions and call all the relevant functions [ For the sessions that need to be processed]
        for session_name in sorted(db.index):
            session = db.loc[session_name]
            # Check if this is one of th sessions we should be processing
            if not selector_type == 'any':
                if selector_type == 'experiment' and session.Metadata.experiment not in selector:
                    continue
                elif selector_type == 'date' and str(session.Metadata.date) not in selector:
                    continue
                elif selector_type == 'session' and session.Metadata.session_id not in selector:
                    continue

            print('---------------\nProcessing session {}'.format(session_name))

            # extract info from sessions videos [e.g. first frame, fps, videos lenth...]
            if extract_background or track_mouse:
                session = get_session_videodata(session)

            # Process background: get maze edges and user selected ROIs
            if extract_background:
                maze_edges, user_rois = Image_processing.process_background(session['Video']['Background'],
                                                                            track_options)
                session['Video']['Maze Edges'] = maze_edges
                session['Video']['User ROIs'] = user_rois
                save_data(save_fld, save_name, name_modifier='_background', object=db)

            # Track animal on videos   <---!!!!
            if track_mouse:
                tracked = Tracking(session, db, self.TF_setup, self.TF_sttings)
                db = tracked.database
                self.TF_setup = tracked.TF_setup
                self.TF_sttings = tracked.TF_settings
                save_data(save_fld, save_name, name_modifier='_tracking', object=db)

            # Processing
            if processing:
                Processing_main.Processing(session, db)

            # Plot for individual mice
            if plotting:
                Plotting_main.setup_plotting(session, db)

        if Cohort:
            # Create a cohort and store it in database
            db = create_cohort(db)  # Get all the trial data in one place

            print('Processing Cohort')
            # Plot for a cohort
            if plotting:
                Plotting_main.setup_plotting(None, db, selector='cohort')

        # Save results at the end of the analysis
        save_data(save_fld, save_name, object=db, name_modifier='_completed')


#######################
#  START              #
#######################
if __name__ == "__main__":
    Analysis()



