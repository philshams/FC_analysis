from Config import setup, tracking_options
excel_path, save_folder, DLC_folder, load_database, update_database, load_name, save_name, selector_type, selector = setup()
track_options, _, fisheye_map_location, _ = tracking_options()
from termcolor import colored
from Utils.registration_funcs import register_arena, get_background, get_arena_details, perform_arena_registration
from Utils.loadsave_funcs import save_data, load_data, load_paths, load_yaml, print_plans, setup_database
from Utils.Data_rearrange_funcs import check_session_selected


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

        # Loop over each session
        for session_name in self.db.index[::-1]:
            # Check if this is one of the sessions we should be processing
            session = self.db.loc[session_name]
            selected = check_session_selected(session.Metadata, selector_type, selector)

            # If it is, register and analyze that session
            if selected:
                print(colored('Analyzing session: {}'.format(session_name), 'green'))

                # First, register the session to the common coordinate space
                if track_options['register arena']:
                    session, new_registration = perform_arena_registration(session, fisheye_map_location)

                    # Input the data from the registration into the global database
                    if new_registration:
                        self.db.loc[session_name]['Registration'] = session['Registration']
                        save_data(save_folder, load_name, save_name, object=self.db, name_modifier='')

                # Now, analyze the session
                from Tracking.Tracking_main import Tracking
                Tracking(session)

                # Input the data from the analysis into the global database
                self.db.loc[session_name]['Tracking'] = session['Tracking']
                save_data(save_folder, load_name, save_name, object=self.db, name_modifier='')

        # We're done here.
        print(colored('done.', 'green'))


#  START
if __name__ == "__main__":
    Analysis()



