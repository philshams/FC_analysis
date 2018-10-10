from Utils.imports import *
from Utils.maths import line_smoother
import math

def plot_velocites_grouped(self, tracking_data, metadata, selector='exp'):
    # Get experiment tags and plotting colors
    tags = []
    assigned_tags = {}
    for session in metadata.sessions_in_cohort:
        if selector == 'exp':
            tags.append(session[1].experiment)
            assigned_tags[str(session[1].session_id)] = session[1].experiment
        elif selector == 'stim_type':
            tags = ['visual', 'audio']
            break
        elif selector == 'escape_arm':
            tags = ['Left', 'Central', 'Right']
            break
    tags = [t for t in set(tags)]
    colors = [[.2, .2, .4], [.2, .4, .2], [.4, .4, .2], [.4, .2, .2], [.2, .4, .4]]
    colors = {name: col for name, col in zip(tags, colors)}

    # Get number of trials per tag
    max_counterscounters = {t: 0 for t in tags}
    for trial in tracking_data.trials:
        if not trial.processing['Trial outcome']['trial_outcome']: continue
        sessid = trial.name.split('-')[0]
        if selector == 'exp':
            tag = assigned_tags[sessid]
        elif selector == 'stim_type':
            tag = trial.metadata['Stim type']
        elif selector == 'escape_arm':
            tag = trial.processing['Trial outcome']['threat_escape_arm'].split('_')[0]
            if tag not in max_counterscounters.keys():
                print(trial.name, 'has weird tag: {}'.format(tag))
                continue
        max_counterscounters[tag] = max_counterscounters[tag] + 1

    # initialise figure
    f, axarr = plt.subplots(3, 2, facecolor=[0.1, 0.1, 0.1])
    axarr = axarr.flatten()
    axes_dict = {'vel':axarr[0], 'acc':axarr[1], 'angvel':axarr[2], 'angacc':axarr[3],
                 'nbl':axarr[4], 'bang':axarr[5]}
    # f.tight_layout()

    # itialise container vars
    template_dict = {name: np.zeros((3600, max_counterscounters[name])) for name in tags}

    variables_dict = dict(
                    velocities=template_dict.copy(),
                    accelerations=template_dict.copy(),
                    head_velocities=template_dict.copy(),
                    head_accelerations=template_dict.copy(),
                    body_lengths=template_dict.copy(),
                    head_body_angles=template_dict.copy())

    # loop over trials and assign the data to the correct tag in the variables dict
    counters = {t: 0 for t in tags}
    for trial in tracking_data.trials:
        # Skip no escape trials
        if not trial.processing['Trial outcome']['trial_outcome'] or \
                trial.processing['Trial outcome']['trial_outcome'] is None:
            print(trial.name, 'no escape')
            continue
        # Get tag
        sessid = trial.name.split('-')[0]
        if selector == 'exp':
            tag = assigned_tags[sessid]
        elif selector == 'stim_type':
            tag = trial.metadata['Stim type']
        elif selector == 'escape_arm':
            tag = trial.processing['Trial outcome']['threat_escape_arm'].split('_')[0]
        if tag not in max_counterscounters.keys(): continue

        # Get variables
        vel = line_smoother(trial.dlc_tracking['Posture']['body']['Velocity'].values)
        acc = line_smoother(np.diff(vel))
        head_ang_vel = np.abs(line_smoother(trial.dlc_tracking['Posture']['body']['Head ang vel'].values))
        head_ang_acc = line_smoother(np.diff(head_ang_vel))
        body_len = line_smoother(trial.dlc_tracking['Posture']['body']['Body length'].values)
        pre_stim_mean_bl = np.mean(body_len[(1800 - 121):1799])  # bodylength randomised to the 4s before the stim
        body_len = np.divide(body_len, pre_stim_mean_bl)

        # Head-body angle variable
        body_ang = line_smoother(trial.dlc_tracking['Posture']['body']['Orientation'].values)
        head_ang = line_smoother(trial.dlc_tracking['Posture']['body']['Head angle'].values)
        while np.any(body_ang[body_ang > 360]): body_ang[body_ang > 360] -= 360
        while np.any(head_ang[head_ang > 360]): head_ang[head_ang > 360] -= 360
        hb_angle = np.abs(line_smoother(np.diff(np.radians(body_ang) - np.radians(head_ang))))

        # store variables in dictionray and plot them
        variables_dict = {'vel': vel, 'acc': acc, 'angvel': head_ang_vel, 'angacc': head_ang_acc,
                          'nbl': body_len, 'bang': hb_angle}
        for vname, vdata in variables_dict.items():
            axes_dict[vname].plot(vdata, color=colors[tag], alpha=0.15)

        # make sure all vairbales are of the same length
        tgtlen = 3600
        for vname, vdata in variables_dict.items():
            while len(vdata) < tgtlen:
                vdata = np.append(vdata, 0)
            variables_dict[vname] = vdata

        # Update container vars and tags counters
        variables_dict['velocities'][tag][:, counters[tag]] = variables_dict['vel']
        variables_dict['accelerations'][tag][:, counters[tag]] = variables_dict['acc']
        variables_dict['head_velocities'][tag][:, counters[tag]] = variables_dict['angvel']
        variables_dict['head_accelerations'][tag][:, counters[tag]] = variables_dict['angacc']
        variables_dict['body_lengths'][tag][:, counters[tag]] = variables_dict['nbl']
        variables_dict['head_body_angles'][tag][:, counters[tag]] = variables_dict['bang']
        counters[tag] = counters[tag] + 1

    # Plot means and SEM
    def means_plotter(var, ax):
        for name, val in var.items():
            if name == 'PathInt2':  lg = '2Arms Maze'
            elif name == 'PathInt': lg = '3Arms Maze'
            else: lg = name

            avg = np.median(val, axis=1)
            sem = np.std(val, axis=1) / math.sqrt(max_counterscounters[name])
            ax.errorbar(x=np.linspace(0, len(avg), len(avg)), y=avg, yerr=sem,
                              color=np.add(colors[name], 0.3), alpha=0.65, linewidth=4, label=lg)

    means_plotter(variables_dict['velocities'], axes_dict['vel'])
    means_plotter(variables_dict['accelerations'], axes_dict['acc'])
    means_plotter(variables_dict['head_velocities'], axes_dict['angvel'])
    means_plotter(variables_dict['head_accelerations'], axes_dict['angacc'])
    means_plotter(variables_dict['body_lengths'], axes_dict['nbl'])
    means_plotter(variables_dict['head_body_angles'], axes_dict['bang'])

    # Update axes
    xlims, facecolor = [1780, 1950], [.0, .0, .0]
    axes_params = {'vel': ('MEAN VELOCITY aligned to stim', [-0.5, 20],'Frames', 'px/frame'),
                   'acc': ('MEDIAN ACCELERATION aligned to stim', [-0.5, 1], 'Frames', ''),
                   'angvel': ('MEAN Head Angular VELOCITY', [-0.1, 20],'Frames', 'deg/frame'),
                   'angacc': ('MEAN Head Angular ACCELERATION', [-1, 1],'Frames', ''),
                   'nbl': ('MEAN normalised body LENGTH', [0.5, 1.25],'Frames', 'arbritary unit'),
                   'bang': ('MEAN head-body angle', [0, 0.6],'Frames', 'deg')}

    for axname, ax in axes_dict.items():
        ax.axvline(1800, color='w')
        make_legend(ax, [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=16)
        ax.set(title=axes_params[axname][0], xlim=xlims, ylim=axes_params[axname][1],
               xlabel=axes_params[axname][2], ylabel=axes_params[axname][3], facecolor=facecolor)

    a = 1
