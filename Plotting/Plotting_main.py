from Plotting.Plotting_individuals_maze import plot_individuals_maze
from Plotting.Plotting_cohort_maze import  plot_status_at_timepoint

from Config import plotting_individuals, exp_type, plotting_cohort


def setup_plotting(session_data, db, selector='individual'):
    if selector == 'individual':
        if plotting_individuals:
            if exp_type == 'maze':
                plot_individuals_maze(session_data)

    else:
        if plotting_cohort:
            if exp_type == 'maze':
                cohort_trials_data = db['All_trials']['processed']
                plot_status_at_timepoint(cohort_trials_data)


