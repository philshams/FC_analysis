import yaml

from Utils.loadsave_funcs import load_yaml
from Processing import Processing_utils
from Plotting import Plt_individual_stimresponses

from Config import processing_options


class Processing():
    def __init__(self, session, database):
        # load processing settings from yaml file
        settings = load_yaml(processing_options['cfg'])

        body_data = Processing_utils.from_dlc_to_single_bp(session, settings['center of mass bodypart'])
        head_data = Processing_utils.from_dlc_to_single_bp(session, settings['head'])
        tail_data = Processing_utils.from_dlc_to_single_bp(session, settings['tail'])

        # Calculate velocity and other stuff
        body_data = Processing_utils.single_bp_calc_stuff(body_data)
        head_data = Processing_utils.single_bp_calc_stuff(head_data)
        tail_data = Processing_utils.single_bp_calc_stuff(tail_data)

        # Reconstruct posture
        body_data = Processing_utils.pose_reconstruction(head_data, body_data, tail_data)

        # Plot
        Plt_individual_stimresponses.plotter(body_data)


