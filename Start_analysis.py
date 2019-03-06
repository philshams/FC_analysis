from Config import setup, tracking_options
excel_path, save_folder, DLC_folder, load_database, update_database, load_name, save_name, selector_type, selector = setup()
track_options, _, fisheye_map_location, _ = tracking_options()
from termcolor import colored
from Utils.registration_funcs import register_arena, get_background, get_arena_details, perform_arena_registration
from Utils.loadsave_funcs import save_data, load_data, load_paths, load_yaml, print_plans, setup_database
from Utils.Data_rearrange_funcs import check_session_selected
from multiprocessing.dummy import Pool as ThreadPool

'''
...................................START ANALYSIS...................................

'''
class Analysis():
    def __init__(self):
        '''
        Set up the analysis object, load the database, do the analysis
        '''
        # print what we are planning to do
        print_plans(load_database, load_name, selector_type, selector)

        # Load the database
        self.db = setup_database(save_folder, save_name, load_name, excel_path, load_database, update_database)
        # to see all entries: self.db
        # to drop an entry: self.db = self.db.drop(['desired_entry_to_drop'])

        # Call the main func that orchestrates the application of the processes
        self.main()


    def main(self):
        """"
        Once all is set up, analyze each session
        """
        # run analysis for the a threadpooled list of sessions
        num_parallel_processes = track_options['parallel processes']
        splitted_session_list = [self.db.index[i::num_parallel_processes] for i in range(num_parallel_processes)]
        pool = ThreadPool(num_parallel_processes)
        _ = pool.map(self.run_analysis, splitted_session_list)

        # save the data
        save_data(save_folder, save_name, object=self.db, name_modifier='')

        # We're done here.
        print(colored('done.', 'green'))

    def run_analysis(self, session_list):
        '''
        Run through the registration, tracking, and visualization steps
        '''
        # Loop over each session, starting with the most recent
        for session_name in session_list[::-1]:
            # Check if this is one of the sessions we should be processing
            session = self.db.loc[session_name]
            selected = check_session_selected(session.Metadata, selector_type, selector)

            # If it is, register and analyze that session
            if selected:
                print(colored('Analyzing session {}: {} - {}'.format(session.Metadata.number,
                                                                     session.Metadata.experiment, session.Metadata.mouse_id), 'green'))

                # First, register the session to the common coordinate space
                if track_options['register arena']:
                    session, new_registration = perform_arena_registration(session, fisheye_map_location)

                    # Input the data from the registration into the global database
                    if new_registration:
                        self.db.loc[session_name]['Registration'] = session['Registration']
                        save_data(save_folder, save_name, object=self.db, name_modifier='')

                # Now, analyze the session
                from Tracking.Tracking_main import Tracking
                Tracking(session)

                # Input the data from the analysis into the global database
                self.db.loc[session_name]['Tracking'] = session['Tracking']
                save_data(save_folder, save_name, object=self.db, name_modifier='')



#  START
if __name__ == "__main__":
    A = Analysis()



