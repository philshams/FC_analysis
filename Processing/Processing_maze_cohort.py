from Utils.imports import *

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('axes', edgecolor=[0.8, 0.8, 0.8])
params = {'legend.fontsize': 16,
          'legend.handlelength': 2}
col = [0, 0, 0]
# col = [.8, .8, .8]
matplotlib.rc('axes', edgecolor=col)
matplotlib.rcParams['text.color'] = col
matplotlib.rcParams['axes.labelcolor'] = col
matplotlib.rcParams['axes.labelcolor'] = col
matplotlib.rcParams['xtick.color'] = col
matplotlib.rcParams['ytick.color'] = col

plt.rcParams.update(params)

import array
import math
import scipy.stats
import random
import seaborn as sns
from scipy.stats import norm

from Plotting.Plotting_utils import make_legend, save_all_open_figs
from Utils.Data_rearrange_funcs import flatten_list
from Utils.maths import line_smoother

from Config import cohort_options

arms_colors = dict(left=(255, 0, 0), central=(0, 255, 0), right=(0, 0, 255), shelter=(200, 180, 0),
                        threat=(0, 180, 200))


class mazecohortprocessor:
    def __init__(self, cohort):

        fig_save_fld = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\Presentations\\181017_graphs'

        self.colors = dict(left=[.2, .3, .7], right=[.7, .3, .2], centre=[.3, .7, .2], center=[.3, .7, .2],
                           central=[.3, .7, .2], shelter='c', threat='y')

        name = cohort_options['name']
        print(colored('Maze processing cohort {}'.format(name), 'green'))

        metad =  cohort.Metadata[name]
        tracking_data = cohort.Tracking[name]

        self.sample_escapes_probabilities(tracking_data)

        self.plot_velocites_grouped(tracking_data, metad, selector='exp')

        self.process_trials(tracking_data)


        # self.process_status_at_stim(tracking_data)

        # self.process_explorations(tracking_data)

        # self.process_by_mouse(tracking_data)

        # plt.show()

        save_all_open_figs(target_fld=fig_save_fld, name=name, format='svg')
        plt.show()
        a = 1

    def coin_simultor(self, num_samples=10, num_iters=10000):
        probs = []
        for itern in tqdm(range(num_iters)):
            data = [random.randint(0, 1) for x in range(num_samples)]
            prob_one = len([n for n in data if n==1])/len(data)
            probs.append(prob_one)

        f, ax = plt.subplots(1, 1, facecolor=[0.1, 0.1, 0.1])
        ax.set(facecolor=[0.2, 0.2, 0.2], xlim=[0,1], ylim=[0, 4000])
        basecolor = [.3, .3, .3]

        ax.hist(probs, color=(basecolor), bins=75)
        avg = np.mean(probs)
        # ax.axvline(avg, color=basecolor, linewidth=4, linestyle=':')
        print('mean {}'.format(avg))

    def sample_escapes_probabilities(self, tracking_data, num_samples=24, num_iters=10000, replacement=True):
        sides = ['Left', 'Central', 'Right']
        # get escapes
        maze_configs, origins, escapes, outcomes = [], [], [], []
        for trial in tracking_data.trials:
            outcome = trial.processing['Trial outcome']['trial_outcome']
            if not outcome:
                print(trial.name, ' no escape')
                continue
            if trial.processing['Trial outcome']['maze_configuration'] == 'Left':
                print(trial.name, ' left config')
                continue

            maze_configs.append(trial.processing['Trial outcome']['maze_configuration'])
            origins.append(trial.processing['Trial outcome']['threat_origin_arm'])
            escapes.append(trial.processing['Trial outcome']['threat_escape_arm'])
            outcomes.append(outcome)

        num_trials = len(outcomes)
        left_escapes = len([b for b in escapes if 'Left' in b])
        central_escapes = len([b for b in escapes if 'Central' in b ])

        print('\nFound {} trials, escape probabilities: {}-Left, {}-Centre'.format(num_trials,
                                                                             round(left_escapes/num_trials, 2),
                                                                             round(central_escapes/num_trials, 2)))

        if not num_samples: num_samples = num_trials
        if num_samples > num_trials: raise Warning('Too many samples')

        probs = {name:[] for name in sides}
        for iter in tqdm(range(num_iters)):
            if not replacement:
                sel_trials = random.sample(escapes, num_samples)
            else:
                sel_trials = [random.choice(escapes) for _ in escapes]
                sel_trials = sel_trials[:num_samples]
            p = 0
            for side in sides:
                probs[side].append(len([b for b in sel_trials if side in b])/num_samples)
                p += len([b for b in sel_trials if side in b])/num_samples
                if p > 1: raise Warning('Something went wrong....')

        f, ax = plt.subplots(1, 1,  facecolor=[0.1, 0.1, 0.1])
        ax.set(facecolor=[0.2, 0.2, 0.2])
        basecolor = [.3, .3, .3]

        for idx, side in enumerate(probs.keys()):
            if not np.any(np.asarray(probs[side])): continue
            # sns.kdeplot(probs[side], bw=0.05, shade=True, color=np.add(basecolor, 0.2*idx), label=side, ax=ax)
            ax.hist(probs[side], color=np.add(basecolor, 0.2*idx), bins = 150,   label=side)
            # sns.distplot(probs[side], bins=3, label=side, ax=ax)
            # sns.distplot(probs[side], fit=norm, hist=False, kde=False, norm_hist=True, ax=ax)

        # for idx, side in enumerate(probs.keys()):
        #     if not np.any(np.asarray(probs[side])): continue
        #     avg = np.mean(probs[side])
        #     ax.axvline(avg, color=np.add(basecolor, 0.2 * idx), linewidth=4, linestyle=':')
        #     print('side {} mean {}'.format(side,avg))

        make_legend(ax, [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=16)

        a = 1

    def plot_velocites_grouped(self, tracking_data, metadata, selector='exp'):
        def line_smoother(data, order=0):
            return data

        align_to_det = False
        detections_file = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis\\det_times.txt'
        smooth_lines = False
        aligne_at_stim_time = False
        if smooth_lines:
            from Utils.maths import line_smoother

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
        # colors = [[.2, .2, .4], [.2, .4, .2], [.4, .4, .2], [.4, .2, .2], [.2, .4, .4]]
        colors = [[.40, .28, .22], [.35, .35, .35]]
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
        axes_dict = {'vel': axarr[0], 'time_to_escape_bridge': axarr[1], 'angvel': axarr[2], 'time_to_shelter': axarr[3],
                     'nbl': axarr[4], 'vel_at_esc_br': axarr[5]}
        # f.tight_layout()

        # itialise container vars
        template_dict = {name: np.zeros((3600, max_counterscounters[name])) for name in tags}

        variables_dict = dict(
            velocities={name: np.zeros((3600, max_counterscounters[name])) for name in tags},
            time_to_escape_bridge={name: [] for name in tags},
            head_velocities={name: np.zeros((3600, max_counterscounters[name])) for name in tags},
            time_to_shelter={name: [] for name in tags},
            body_lengths={name: np.zeros((3600, max_counterscounters[name])) for name in tags},
            velocity_at_escape_bridge={name: [] for name in tags})

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

            time_to_escape_bridge = [i for i,r in enumerate(trial.processing['Trial outcome']['trial_rois_trajectory'][1800:]) if 'Threat' not in r][0] # line_smoother(np.diff(vel), order=10)
            head_ang_vel = np.abs(line_smoother(trial.dlc_tracking['Posture']['body']['Body ang vel'].values), order=10)
            time_to_shelter = [i for i,r in enumerate(trial.processing['Trial outcome']['trial_rois_trajectory'][1800:]) if 'Shelter' in r][0] # line_smoother(np.diff(head_ang_vel), order=10)
            body_len = line_smoother(trial.dlc_tracking['Posture']['body']['Body length'].values)
            # pre_stim_mean_bl = np.mean(body_len[(1800 - 31):1799])  # bodylength randomised to the 4s before the stim
            pre_stim_mean_bl = body_len[1800]
            body_len = np.divide(body_len, pre_stim_mean_bl)

            vel = line_smoother(trial.dlc_tracking['Posture']['body']['Velocity'].values, order=10)
            vel = np.divide(vel, pre_stim_mean_bl)

            # Head-body angle variable
            # body_ang = line_smoother(np.rad2deg(np.unwrap(np.deg2rad(trial.dlc_tracking['Posture']['body']['Orientation']))))
            # head_ang = line_smoother(np.rad2deg(np.unwrap(np.deg2rad(trial.dlc_tracking['Posture']['body']['Head angle']))))
            # headbody_ang = np.abs(line_smoother(np.subtract(head_ang, body_ang)))
            # headbody_ang = np.subtract(headbody_ang, headbody_ang[1800])
            # hb_angle = headbody_ang
            vel_at_escape_bridge = vel[1800+time_to_escape_bridge]

            # store variables in dictionray and plot them
            temp_dict = {'vel': vel, 'time_to_escape_bridge': time_to_escape_bridge, 'angvel': head_ang_vel,
                         'time_to_shelter': time_to_shelter,
                         'nbl': body_len, 'vel_at_esc_br': vel_at_escape_bridge}


            if aligne_at_stim_time:
                for n, v in temp_dict.items():
                    try:  temp_dict[n] = np.subtract(v, v[1800])
                    except: pass


            # make sure all vairbales are of the same length
            tgtlen = 3600
            for vname, vdata in temp_dict.items():
                if isinstance(vdata, np.ndarray):
                    while len(vdata) < tgtlen:
                        vdata = np.append(vdata, 0)
                        temp_dict[vname] = vdata


            if align_to_det:
                with open(detections_file, 'r') as f:
                    check = False
                    for line in f:
                        if trial.name in line:
                            det = int(line.split(' ')[-1])
                            check = True
                            break
                if check:
                    for n,v in temp_dict.items():
                        if isinstance(v, np.ndarray):
                            v = v[det-1800:]
                            v = np.append(v, np.zeros(det-1800))
                            temp_dict[n] = v

            for vname, vdata in temp_dict.items():
                if isinstance(vdata, np.ndarray):
                    axes_dict[vname].plot(vdata, color=colors[tag], alpha=0.15)

            # Update container vars and tags counters
            variables_dict['velocities'][tag][:, counters[tag]] = temp_dict['vel']
            variables_dict['time_to_escape_bridge'][tag].append(temp_dict['time_to_escape_bridge'])
            variables_dict['head_velocities'][tag][:, counters[tag]] = temp_dict['angvel']
            variables_dict['time_to_shelter'][tag].append(temp_dict['time_to_shelter'])
            variables_dict['body_lengths'][tag][:, counters[tag]] = temp_dict['nbl']
            variables_dict['velocity_at_escape_bridge'][tag].append(temp_dict['vel_at_esc_br'])
            counters[tag] = counters[tag] + 1

        # Plot means and SEM
        def means_plotter(var, ax):
            means = []
            for name, val in var.items():
                if name == 'PathInt2': lg = '2Arms Maze'
                elif name == 'PathInt': lg = '3Arms Maze'
                else: lg = name
                avg = np.mean(val, axis=1)
                sem = np.std(val, axis=1) / math.sqrt(max_counterscounters[name])
                ax.errorbar(x=np.linspace(0, len(avg), len(avg)), y=avg, yerr=sem,
                            color=np.add(colors[name], 0.3), alpha=0.65, linewidth=4, label=lg)
                means.append(avg)
            return means

        def scatter_plotter(var, ax):
            for idx, (name, val) in enumerate(var.items()):
                if name == 'PathInt2': lg = '2Arms Maze'
                elif name == 'PathInt': lg = '3Arms Maze'
                else: lg = name
                y = np.divide(np.random.rand(len(val)), 4) + 0.5*idx +0.05
                ax.scatter(val, y, color=colors[name], alpha=0.65, s=30, label=lg)
                ax.axvline(np.median(val), color=colors[name], alpha=0.5, linewidth=2)

        mean_vels = means_plotter(variables_dict['velocities'], axes_dict['vel'])
        mean_blengths = means_plotter(variables_dict['body_lengths'], axes_dict['nbl'])
        mean_headvels = means_plotter(variables_dict['head_velocities'], axes_dict['angvel'])

        scatter_plotter(variables_dict['time_to_escape_bridge'], axes_dict['time_to_escape_bridge'])
        scatter_plotter(variables_dict['time_to_shelter'], axes_dict['time_to_shelter'])
        scatter_plotter(variables_dict['velocity_at_escape_bridge'], axes_dict['vel_at_esc_br'])

        # Update axes
        xlims, scatter_xlims, facecolor = [1785, 1890], [0, 270], [.0, .0, .0]
        t1 = np.arange(xlims[0], xlims[1] + 15, 15)
        t2 = np.arange(scatter_xlims[0], scatter_xlims[1] + 15, 15)
        t3 = np.arange(0, 0.5 + 0.05, 0.05)

        axes_params = {'vel': ('MEAN VELOCITY aligned to stim', xlims, [-0.05, 0.25], 'Frames', 'bl/frame', t1),
                       'time_to_escape_bridge': ('Time to escape bridge', scatter_xlims, [0, 1], 'Frames', '', t2),
                       'angvel': ('MEAN Head Angular VELOCITY', xlims, [-2, 10], 'Frames', 'deg/frame', t1),
                       'time_to_shelter': ('Time to sheleter', scatter_xlims, [0, 1], 'Frames', '', t2),
                       'nbl': ('MEAN normalised body LENGTH', xlims, [0.5, 1.5], 'Frames', 'arbritary unit', t1),
                       'vel_at_esc_br': ('Velocity at escape bridge', [0, 0.5], [0, 1], 'Frames', 'deg', t3)}

        for axname, ax in axes_dict.items():
            ax.axvline(1800, color='w')
            make_legend(ax, [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=16)
            ax.set(title=axes_params[axname][0], xlim=axes_params[axname][1], ylim=axes_params[axname][2],
                   xticks=axes_params[axname][5],
                   xlabel=axes_params[axname][3], ylabel=axes_params[axname][4], facecolor=facecolor)

        ####################################################################################
        ####################################################################################
        ####################################################################################

        all_means = [mean_vels, mean_headvels, mean_blengths]
        ttls, names = ['flipflop', 'twoarms'], ['velocity', 'head velocity', 'bodylength']
        colors = ['r', 'g', 'm']
        f, ax = plt.subplots(2, 2, facecolor=[0.1, 0.1, 0.1])
        ax = ax.flatten()
        for n in range(2):
            for idx, means in enumerate(all_means):
                d = means[n][1800: 1860]
                mind, maxd = np.min(d), np.max(d)
                # normed = np.divide(np.subtract(d, mind), (maxd-mind))
                normed = d

                ax[n].plot(normed, linewidth=2, alpha=0.5, color=colors[idx], label=names[idx])
                ax[n+2].plot(np.diff(normed), linewidth=2, alpha=0.5, color=colors[idx], label=names[idx])

            make_legend(ax[n], [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=16)
            ax[n].set(facecolor=facecolor, title = ttls[n])
            ax[n+2].set(facecolor=facecolor, title = ttls[n])

        # Generate a scatter plot of when the max ang vel is reached for each trial (in the 60 frames after stim)
        f, ax = plt.subplots(facecolor=[0.1, 0.1, 0.1])
        baseline_range, test_range = 1790, 60
        maxes = {}
        for idx, (name, values) in enumerate(variables_dict['body_lengths'].items()):
            baseline = np.mean(variables_dict['body_lengths'][name][1799-baseline_range:1799,:], axis=0)
            baseline_std = np.std(variables_dict['body_lengths'][name][1799-baseline_range:1799,:], axis=0)
            # th = np.add(baseline, 1.0*baseline_std)
            # above_th = np.argmax(variables_dict['body_lengths'][name][1799:1859,:]>th, axis=0).astype('float')
            th = 0.95
            # above_th = np.argmax(variables_dict['body_lengths'][name][1800:1800+test_range,:]>th, axis=0).astype('float')
            above_th = np.argmax(np.subtract(variables_dict['body_lengths'][name][1799:1859,:], th)<=0, axis=0).astype('float')
            above_th[above_th == 0] = np.nan
            #belo_th = np.argmax(np.subtract(variables_dict['body_lengths'][name][1799:1859,:], th)>=0.10, axis=0).astype('float')
            #belo_th[belo_th == 0] = np.nan

            randomy = np.divide(np.random.rand(len(above_th)), 4) + 1 - idx*0.5
            ax.scatter(above_th, randomy, color=colors[name], alpha=0.85, s=15, label=name)
            ax.axvline(np.nanmedian(above_th), color=colors[name], linewidth=2, linestyle=':')
            #ax.scatter(belo_th, np.subtract(randomy, 0.5), color=np.add(colors[name], 0.35), alpha=0.85, s=15, label=name)
            #ax.axvline(np.nanmedian(belo_th), color=np.add(colors[name], .35), linewidth=2, linestyle=':')



        ax.set(facecolor=facecolor, xlim=[0, 60])
        make_legend(ax, [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=16)

        a = 1

    def process_by_mouse(self, tracking_data):
        sess_id = None
        mice = []
        confs, ors, escs, outs = [], [], [], []
        configs, origins, escapes, outcomes = [], [], [], []
        counter = 0
        for trial in tracking_data.trials:
            tr_sess_id = int(trial.name.split('-')[0])
            if counter == 0:
                sess_id = tr_sess_id
                mice.append(sess_id)
            counter += 1
            ors.append(trial.processing['Trial outcome']['threat_origin_arm'])
            escs.append(trial.processing['Trial outcome']['threat_escape_arm'])
            outs.append(trial.processing['Trial outcome']['trial_outcome'])
            confs.append(trial.processing['Trial outcome']['maze_configuration'])

            if tr_sess_id != sess_id:
                origins.append(ors), escapes.append(escs), outcomes.append(outs), mice.append(tr_sess_id)
                configs.append(confs)
                confs, ors, escs, outs = [], [], [], []
                sess_id = tr_sess_id
        origins.append(ors), escapes.append(escs), outcomes.append(outs), configs.append(confs)

        all_configs = ['All']+ [c for c in set(configs[0])]
        sides = ['Left', 'Central', 'Right']
        for conf in all_configs:
            f, axarr = plt.subplots(round(len(origins)/4), 5, facecolor=[0.1, 0.1, 0.1])
            f.tight_layout()
            axarr = axarr.flatten()

            for mousenum in range(len(origins)):
                ori_probs, escs_probs = {side: None for side in sides}, {side: None for side in sides}
                if conf == 'All':
                    ors = [o for i,o in enumerate(origins[mousenum]) if outcomes[mousenum][i]]
                    escs = [e for i,e in enumerate(escapes[mousenum]) if outcomes[mousenum][i]]
                else:
                    ors = [o for i, o in enumerate(origins[mousenum])
                           if configs[mousenum][i] == conf and outcomes[mousenum][i]]
                    escs = [e for i, e in enumerate(escapes[mousenum])
                            if configs[mousenum][i] == conf and outcomes[mousenum][i]]
                num_trials = len(ors)

                for idx, side in enumerate(sides):
                    ori_probs[side] = len([o for i, o in enumerate(ors) if side in o])/num_trials
                    escs_probs[side] = len([e for i, e in enumerate(escs) if side in e])/num_trials

                    axarr[mousenum].bar(idx - 4, ori_probs[side], color=self.colors[side.lower()])
                    axarr[mousenum].bar(idx + 4, escs_probs[side], color=self.colors[side.lower()])

                axarr[mousenum].axvline(0, color='w')
                axarr[mousenum].set(title='m{} - {} trial'.format(mice[mousenum], num_trials), ylim=[0, 1],
                                    facecolor=[0.2, 0.2, 0.2])

        plt.show()
        a = 1

    def process_status_at_stim(self, tracking_data):
        statuses = dict(positions=[], orientations=[], velocities=[], body_lengts=[], origins=[], escapes=[])

        orientation_th = 30  # degrees away from facing south considered as facing left or right
        orientation_arm_probability = {ori:[0, 0, 0] for ori in ['looking_left', 'looking_down', 'looking_right']}
        non_alternations, alltrials = 0, 0
        for trial in tracking_data.trials:
            if 'status at stimulus' in trial.processing.keys():
                # get Threat ROI location
                try:
                    trial_sess_id = int(trial.name.split('-')[0])
                    threat_loc = None
                    for expl in tracking_data.explorations:
                        if not isinstance(expl, tuple):
                            sess_id = expl.metadata.session_id
                            if sess_id == trial_sess_id:
                                threat = expl.processing['ROI analysis']['all rois']['Threat_platform']
                                threat_loc = ((threat.topleft[0]+threat.bottomright[0])/2,
                                              (threat.topleft[1]+threat.bottomright[1])/2)
                                break
                    if threat_loc is None: raise Warning('Problem')

                    # prep status
                    status = trial.processing['status at stimulus']
                    if status is None: continue
                    outcome = trial.processing['Trial outcome']
                    ori = status.status['Orientation']
                    while ori>360: ori -= 360
                    pos = (status.status['x'], status.status['y'])

                    statuses['positions'].append((pos[0]-threat_loc[0], pos[1]-threat_loc[1]))
                    statuses['orientations'].append(ori)
                    statuses['velocities'].append(status.status['Velocity'])
                    statuses['body_lengts'].append(status.status['Body length'])
                    statuses['origins'].append(outcome['threat_origin_arm'])
                    statuses['escapes'].append(outcome['threat_escape_arm'])

                    # Check correlation between escape arm and orientation at stim
                    if not outcome['trial_outcome']:
                        print(trial.name, ' no escape')
                        continue
                    if 0 < ori < 180-orientation_th:
                        look = 'looking_right'
                    elif 180-orientation_th <= ori <= 180 + orientation_th:
                        look = 'looking_down'
                    else:
                        look = 'looking_left'
                    if 'Left' in outcome['threat_escape_arm']: esc = 0
                    elif 'Central' in outcome['threat_escape_arm']: esc = 1
                    else: esc = 2
                    orientation_arm_probability[look][esc] = orientation_arm_probability[look][esc] + 1

                    # Check probability of alternating
                    alltrials += 1
                    if outcome['threat_origin_arm'] == outcome['threat_escape_arm']: non_alternations += 1
                except:
                    raise Warning('ops')
                    pass

            else:
                print('no status')

        # Print probability of alternation (oirigin vs escape)
        print('Across this dataset, the probability of escaping on the arm of origin was: {}'.format(non_alternations/alltrials))

        # Plot probability of arm given orientation
        f, axarr = plt.subplots(3, 1, facecolor=[0.1, 0.1, 0.1])
        for idx, look in enumerate(orientation_arm_probability.keys()):
            outcomes = orientation_arm_probability[look]
            num_trials = sum(outcomes)
            ax = axarr[idx]

            for i, out in enumerate(outcomes):
                ax.bar(i, out/num_trials)
            ax.set(title='Escape probs for trials {}'.format(look), ylim=[0, 1], xlabel='escape arm', ylabel='probability',
                   facecolor=[0.2, 0.2, 0.2])


        # Plot location and stuff
        f, axarr = plt.subplots(3, 2, facecolor=[0.1, 0.1, 0.1])
        f.tight_layout()
        f.set_size_inches(10, 20)
        axarr = axarr.flatten()


        axarr[0].scatter(np.asarray([p[0] for p in statuses['positions']]), np.asarray([p[1] for p in statuses['positions']]),
                   c=np.asarray(statuses['orientations']), s=np.multiply(np.asarray(statuses['velocities']), 10))

        sides = ['Left', 'Central', 'Right']
        for side in sides:
            axarr[2].scatter(np.asarray([p[0] for i, p in enumerate(statuses['positions'])
                                         if side in statuses['origins'][i]]),
                             np.asarray([p[1] for i, p in enumerate(statuses['positions'])
                                         if side in statuses['origins'][i]]),
                             c=self.colors[side.lower()], s=35, edgecolors='k', alpha=0.85)

            axarr[3].scatter(np.asarray([p[0] for i, p in enumerate(statuses['positions'])
                                         if side in statuses['escapes'][i]]),
                             np.asarray([p[1] for i, p in enumerate(statuses['positions'])
                                         if side in statuses['escapes'][i]]),
                             c=self.colors[side.lower()], s=35, edgecolors=['k'], alpha=0.85)

        axarr[4].scatter(np.asarray([p[0] for i, p in enumerate(statuses['positions'])
                                     if statuses['escapes'][i] == statuses['origins'][i]]),
                         np.asarray([p[1] for i, p in enumerate(statuses['positions'])
                                     if statuses['escapes'][i] == statuses['origins'][i]]),
                         c=[.5, .2, .7], s=35, edgecolors=['k'], alpha=0.85, label='Same')
        axarr[4].scatter(np.asarray([p[0] for i, p in enumerate(statuses['positions'])
                                     if statuses['escapes'][i] != statuses['origins'][i]]),
                         np.asarray([p[1] for i, p in enumerate(statuses['positions'])
                                     if statuses['escapes'][i] != statuses['origins'][i]]),
                         c=[.2, .7, .5], s=35, edgecolors=['k'], alpha=0.85, label='Different')
        make_legend(axarr[4], [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=8)

        axarr[5].scatter(np.asarray([p[0] for i, p in enumerate(statuses['positions'])
                                     if statuses['orientations'][i] > 180+30]),
                         np.asarray([p[1] for i, p in enumerate(statuses['positions'])
                                     if statuses['orientations'][i] > 180+30]),
                         c=[.5, .2, .7], s=35, edgecolors=['k'], alpha=0.85, label='Looking Right')
        axarr[5].scatter(np.asarray([p[0] for i, p in enumerate(statuses['positions'])
                                     if 180-30 <= statuses['orientations'][i] <= 180+30]),
                         np.asarray([p[1] for i, p in enumerate(statuses['positions'])
                                     if 180-30 <= statuses['orientations'][i] <= 180+30]),
                         c=[.7, .5, .2], s=35, edgecolors=['k'], alpha=0.85, label='Looking Down')
        axarr[5].scatter(np.asarray([p[0] for i, p in enumerate(statuses['positions'])
                                     if statuses['orientations'][i] < 180-30]),
                         np.asarray([p[1] for i, p in enumerate(statuses['positions'])
                                     if statuses['orientations'][i] < 180-30]),
                         c=[.2, .7, .5], s=35, edgecolors=['k'], alpha=0.85, label='looking Left')
        make_legend(axarr[5], [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=8)

        titles = ['Position, Velocity, Orientation', '', 'Position, arm of ORIGIN', 'Position, arm of ESCAPE',
                  'Position and same arm', 'Position, ORIENTATION', '', '']
        for idx, ax in enumerate(axarr):
            ax.set(title = titles[idx], xlim=[-75, 75], ylim=[-75, 75], xlabel='x pos', ylabel='y pos',
                   facecolor=[0.2, 0.2, 0.2])
        # plt.show()

    def process_explorations(self, tracking_data):
        all_platforms = ['Left_far_platform', 'Left_med_platform', 'Right_med_platform',
                         'Right_far_platform', 'Shelter_platform', 'Threat_platform']
        alt_all_platforms = ['Left_far_platform', 'Left_medium_platform', 'Right_medium_platform',
                         'Right_far_platform', 'Shelter_platform', 'Threat_platform']

        transitions_list = []
        all_transitions, times = {name:[] for name in all_platforms}, {name:[] for name in all_platforms}
        fps = None
        for exp in tracking_data.explorations:
            if not isinstance(exp, tuple):
                if fps is None: fps = exp.metadata.videodata[0]['Frame rate'][0]
                exp_platforms = [p for p in exp.processing['ROI analysis']['all rois'].keys() if 'platform' in p]

                transitions_list.append(exp.processing['ROI transitions'])

                for p in exp_platforms:  # This is here because some experiments have the platforms named differently
                    if p not in all_platforms:
                        all_platforms = alt_all_platforms
                        all_transitions, times = {name: [] for name in all_platforms}, {name: [] for name in
                                                                                        all_platforms}
                        break
                        # raise Warning()

                timeinroi = {name:val for name,val in exp.processing['ROI analysis']['time_in_rois'].items()
                            if 'platform' in name}
                transitions = {name:val for name,val in exp.processing['ROI analysis']['transitions_count'].items()
                            if 'platform' in name}

                for name in all_platforms:
                    if name in transitions: all_transitions[name].append(transitions[name])
                    if name in timeinroi: times[name].append(timeinroi[name])

        all_platforms = [all_platforms[0], all_platforms[1], all_platforms[4],
                         all_platforms[5], all_platforms[2], all_platforms[3]]

        # Get platforms to and from threat
        all_alternations = []
        for sess in transitions_list:
            alternations = [(sess[i-1], sess[i+1]) for i,roi in enumerate(sess)
                            if 0<i<len(sess)-1 and 'threat' in roi.lower()]
            all_alternations.append(alternations)

        all_alternations = flatten_list(all_alternations)
        num_threat_visits = len(all_alternations)
        num_threat_aternations = len([pp for pp in all_alternations if pp[0]!=pp[1]])
        alternation_probability = round((num_threat_aternations/num_threat_visits), 2)
        print('Average prob: {}'.format(alternation_probability))

        f, axarr = plt.subplots(2, 1, facecolor=[0.1, 0.1, 0.1])
        f.tight_layout()

        timeax, transitionsax = axarr[0], axarr[1]
        for idx, name in enumerate(all_platforms):
            if not name: continue
            type = name.split('_')[0].lower()
            col = self.colors[type]
            if 'far' in name: col = np.add(col, 0.25)

            timeax.bar(idx-.5, np.mean(times[name])/fps, color=col, label=name)
            transitionsax.bar(idx-.5, np.mean(all_transitions[name]), color=col, label=name)


        for ax in axarr:
            ax.axvline(1, color='w')
            ax.axvline(3, color='w')

        timeax.set(title='Seconds per platform', xlim=[-1, 5], ylim=[0, 300], facecolor=[0.2, 0.2, 0.2])
        make_legend(timeax, [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=8)
        transitionsax.set(title='Number of entries per platform', xlim=[-1, 5],  facecolor=[0.2, 0.2, 0.2])

        # plt.show()

    def process_trials(self, tracking_data, select_by_speed=False):
        """ select by speed can be:
            True to select
                fastest: 33% of trials with fastest time to peak acceleration
                slowest: 33% of trials with slowest time to peak acceleration
            False: take all trials

            The code looks at the peak to acceleration in the first 2s (60 frames)
        """
        maze_configs, origins, escapes, outcomes = [], [], [], []
        frames_to_peak_acc = []
        for trial in tracking_data.trials:
            outcome = trial.processing['Trial outcome']['trial_outcome']
            if not outcome:
                print(trial.name, ' no escape')
                continue
            maze_configs.append(trial.processing['Trial outcome']['maze_configuration'])
            origins.append(trial.processing['Trial outcome']['threat_origin_arm'])
            escapes.append(trial.processing['Trial outcome']['threat_escape_arm'])
            outcomes.append(outcome)

            # Get time to peak acceleration
            if select_by_speed:
                accel = trial.dlc_tracking['Posture']['body']['Head ang vel'].values
                baseline = np.mean(accel[1708:1798])
                baseline_std = np.std(accel[1708:1798])
                th = baseline + (1.5 * baseline_std)
                above_th = np.argmax(accel[1799:1859] > th).astype('float')
                if above_th == 0: above_th = np.nan
                frames_to_peak_acc.append(above_th)

        if select_by_speed:
            # Take trials with time to [ang acc > th] within a certain duration
            fast_outcomes = []
            for idx, nframes in enumerate(frames_to_peak_acc):
                if nframes < 10: fast_outcomes.append(escapes[idx])
            nfast_trilas = len(fast_outcomes)
            fast_lesc_prob = len([e for e in fast_outcomes if 'Left' in e]) / nfast_trilas
            fast_cesc_prob = len([e for e in fast_outcomes if 'Central' in e]) / nfast_trilas
            fast_resc_prob = 1 - (fast_lesc_prob + fast_cesc_prob)

        num_trials = len(outcomes)
        left_origins = len([b for b in origins if 'Left' in b])
        central_origins = len([b for  b in origins if 'Central' in b])
        left_escapes = len([b for b in escapes if 'Left' in b])
        central_escapes = len([b for b in escapes if 'Central' in b ])
        r_escapes = num_trials - left_escapes - central_escapes

        if len(set(maze_configs)) > 1:
            outcomes_perconfig = {name:None for name in set(maze_configs)}
            ors = namedtuple('origins', 'numorigins numleftorigins')
            escs = namedtuple('escapes', 'numescapes numleftescapes')
            for conf in set(maze_configs):
                origins_conf = ors(len([o for i,o in enumerate(origins) if maze_configs[i] == conf]),
                                   len([o for i, o in enumerate(origins) if maze_configs[i] == conf and 'Left' in origins[i]]))
                escapes_conf = escs(len([o for i, o in enumerate(escapes) if maze_configs[i] == conf]),
                                    len([o for i, o in enumerate(escapes) if maze_configs[i] == conf and 'Left' in escapes[i]]))
                if len([c for c in maze_configs if c == conf]) < origins_conf.numorigins:
                    raise Warning('Something went wrong, dammit')
                outcomes_perconfig[conf] = dict(origins=origins_conf, escapes=escapes_conf)

                resc = len([o for i, o in enumerate(escapes) if maze_configs[i] == conf and 'Right' in escapes[i]])
                ntr = len([o for i, o in enumerate(escapes) if maze_configs[i] == conf])
                if 'Right' in conf:
                    self.coin_power(n=ntr, n_rescapes=resc)
                else:
                    self.coin_power(n=ntr, n_rescapes=resc)

        else:
            self.coin_power(n=num_trials, n_rescapes=r_escapes)
            outcomes_perconfig = None

        def plotty(ax, numtr, lori, cori, lesc, cesc, title='Origin and escape probabilities'):
            basecolor = [.3, .3, .3]
            ax.bar(0, lori / numtr, color=basecolor, width=.25, label='L ori')
            ax.bar(0.25, cori / numtr, color=np.add(basecolor, .2), width=.25, label='C ori')
            ax.bar(0.5, 1 - ((lori + cori) / numtr), color=np.add(basecolor, .4), width=.25, label='R ori')
            ax.axvline(0.75, color=[1, 1, 1])
            ax.bar(1, lesc / numtr, color=basecolor, width=.25, label='L escape')
            ax.bar(1.25, cesc / numtr, color=np.add(basecolor, .2), width=.25, label='C escape')
            ax.bar(1.5, 1 - ((lesc + cesc) / numtr), color=np.add(basecolor, .4), width=.25, label='R escape')
            ax.set(title='{} - {} trials'.format(title, numtr), ylim=[0, 1], facecolor=[0.2, 0.2, 0.2])
            make_legend(ax, [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=8)

        if outcomes_perconfig is None:
            f, axarr = plt.subplots(1, 1,  facecolor=[0.1, 0.1, 0.1])
        else:
            f, axarr = plt.subplots(3, 1,  facecolor=[0.1, 0.1, 0.1])

        if outcomes_perconfig is not None:
            ax = axarr[0]
            plotty(ax, num_trials, left_origins, central_origins, left_escapes, central_escapes)
        else:
            plotty(axarr, num_trials, left_origins, central_origins, left_escapes, central_escapes)

        if outcomes_perconfig is not None:
            ax = axarr[1]
            for idx, config in enumerate(sorted(outcomes_perconfig.keys())):
                ax = axarr[idx+1]

                o = outcomes_perconfig[config]
                num_trials = o['escapes'].numescapes
                left_escapes = o['escapes'].numleftescapes
                left_origins = o['origins'].numleftorigins
                plotty(ax, num_trials, left_origins, 0, left_escapes, 0, title='{} config'.format(config))

        f.tight_layout()
        #plt.show()

    def coin_power(self, n=100, n_rescapes=0):
        from scipy.stats import binom

        print('BEGIN TESTING COIN POWER WITH {} TRIALS AND {} R ESCAPES'.format(n, n_rescapes))
        alpha = 0.05
        outcomes = []
        for p in np.linspace(0, 1, 101):
            p = round(p, 2)
            # interval(alpha, n, p, loc=0)	Endpoints of the range that contains alpha percent of the distribution
            failure_range = binom.interval(1 - alpha, n, p)
            if failure_range[0] <= n_rescapes <= failure_range[1]:
                print('Cannot reject null hypothesis that p = {} - range: {}'.format(p, failure_range))
                outcomes.append(1)
            else:
                print('Null hypotesis: p = {} - REJECTED'.format(p))
                outcomes.append(0)
        f, ax = plt.subplots(1, 1,  facecolor=[0.1, 0.1, 0.1])
        ax.plot(outcomes, linewidth=2, color=[.6, .6, .6])
        ax.set(facecolor=[0, 0, 0])

