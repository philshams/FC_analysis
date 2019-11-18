from setup import setup
from important_code.do_analysis import analyze_data
import os
import dill as pickle

class analysis():
    def __init__(self, dataframe):
        '''     Analyze the data and make summary plots     '''
        # load the processing parameters
        setup(self)
        # select the relevant sessions
        self.select_sessions(dataframe)
        # make analysis directory
        self.make_analysis_directory()
        # loop over analysis types
        for analysis_type in self.analysis_types.keys():
            # only do selected analysis types
            if not self.analysis_options[analysis_type]: continue
            # load analyzed data
            self.load_analysis(analysis_type)
            # analyze the data
            if self.analysis_options['analyze data']: analyze_data(self, dataframe, analysis_type)
        # Plot all traversals across the arena
        if self.analysis_options['traversals']: self.traversals()
        # Make an exploration heat map
        if self.analysis_options['exploration']: self.exploration()
        # Get escapes
        if self.analysis_options['escapes']: self.speed_traces()
        # Compare various quantities across conditions
        if self.analysis_options['comparisons']: self.comparisons()

    def load_analysis(self, analysis_type):
        '''     Load analysis that's been done      '''
        # find file path to analysis dictionary
        self.save_file = os.path.join(self.folders['save_folder'], 'analysis_data_' + analysis_type)
        # load and initialize dictionary
        self.analysis = {}
        for experiment in self.experiments:
            save_folder = os.path.join( self.dlc_settings['clips_folder'], experiment, analysis_type)
            try:
                with open(save_folder, 'rb') as dill_file:
                    self.analysis[experiment] = pickle.load(dill_file)
            except: self.analysis[experiment] = {}


    def select_sessions(self, dataframe):
        '''     Get a list of the user-selected sessions for analysis      '''
        # initialize selected sessions list
        self.selected_sessions = []
        self.experiments = []
        # loop over all sessions
        for session_name in dataframe.db.index[::-1]:
            # Get the session
            metadata = dataframe.db.loc[session_name].Metadata
            # Check if this is one of the sessions we should be processing
            if metadata['experiment'] in self.flatten(self.analysis_experiments['experiments']):
                self.selected_sessions.append(session_name)
                # add experiment to a list of experiments
                if metadata['experiment'] not in self.experiments: self.experiments.append(metadata['experiment'])



    def make_analysis_directory(self):
        '''     Make a folder to store summary analysis     '''
        self.summary_plots_folder = os.path.join(self.folders['save_folder'], 'Summary Plots')
        if not os.path.isdir(self.summary_plots_folder): os.makedirs(self.summary_plots_folder)

    def traversals(self):
        '''     Analyze traversals across the arena     '''
        # import the relevant script
        from important_code.plot_analysis import plot_traversals
        # plot traversals
        plot_traversals(self)

    def exploration(self):
        '''     Plot exploration heatmaps       '''
        # import the relevant script
        from important_code.plot_analysis import plot_exploration
        # plot exploration
        plot_exploration(self)

    def speed_traces(self):
        '''     Plot escapes        '''
        # import the relevant script
        from important_code.plot_analysis import plot_speed_traces, plot_escape_paths, plot_edginess, plot_efficiency
        # plot speed traces
        plot_efficiency(self)
        plot_edginess(self)

        plot_escape_paths(self)
        plot_speed_traces(self)




    def flatten(self, iterable):
        '''     flatten a nested array      '''
        it = iter(iterable)
        for e in it:
            if isinstance(e, (list, tuple)):
                for f in self.flatten(e): yield f
            else:
                yield e