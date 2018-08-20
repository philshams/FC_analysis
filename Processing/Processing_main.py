import yaml

from Utils.loadsave_funcs import load_yaml
from Processing import Processing_utils
from Plotting import Plt_individual_stimresponses

from Config import processing_options


class Processing():
    def __init__(self, session, database):
        # load processing settings from yaml file
        settings = load_yaml(processing_options['cfg'])

        # Get data from a single bodypart from dlc tracking and plot
        single_bp_trialdata = Processing_utils.from_dlc_to_single_bp(session, settings)

        # Calculate velocity and other stuff
        single_bp_trialdata = Processing_utils.single_bp_calc_stuff(single_bp_trialdata)

        # Plot
        Plt_individual_stimresponses.plotter(single_bp_trialdata)


