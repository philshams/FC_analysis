from moviepy.editor import concatenate_videoclips

from Tracking.Tracking_utils import dlc_setupTF
from Tracking import dlc_analyseVideos
from Utils.imports import *
from Utils.Data_rearrange_funcs import arrange_dlc_data

from Config import track_options


class flipflop:
    def __int__(self, main):
        """ analyse data for sessions from the flipflop maze"""

        self.expls_defition_file = 'C'  # Filepath to txt file with the start time of the post-flip explorations

        self.explorations_tracker(main)

        self.dlc_config_settings = load_yaml(track_options['cfg_dlc'])
        self.clipsfld = self.dlc_config_settings['clips_folder']

    def explorations_tracker(self, main):
        """ define and track (DLC) the pre- and post-flip explorations"""

        skipframes = 60 * main.session['Metadata'].Videodata[0]['Frame rate'][0]
        videos_to_process = [] # .avi clips of the two explorations to be processed in DLC

        # define pre-flip exploration
        all_stims = main.session['Metadata'].stimuli.values()
        all_stims = [item for sublist in all_stims for item in sublist]
        all_stims = [item for sublist in all_stims for item in sublist]
        first_stim = min(all_stims)
        pre_exp_frames = (skipframes, first_stim - 1)  # skip first few frames because I'm in it

        # define post-flip exploration
        stims_before_flip = 0
        with open(self.expls_defition_file, 'r') as f:
            for line in f:
                if main.session.name in line:
                    stims_before_flip = int(line.split(' ')[-1])
                    break
        if not stims_before_flip: raise Warning('Could not define exploration after flip')
        next_stim = all_stims[stims_before_flip]
        post_exp_frames = (all_stims[stims_before_flip-1]*skipframes, next_stim - 1)

        # check if avis already exist for the two explorations
        pre_name = '{}_preflip_exploration_{}-{}.avi'.format(main.session.name, pre_exp_frames[0], pre_exp_frames[1])
        post_name = '{}_preflip_exploration_{}-{}.avi'.format(main.session.name, post_exp_frames[0], post_exp_frames[1])
        names = (pre_name, post_name)
        exps_frames = {names[0]:pre_exp_frames, names[1]:post_exp_frames}
        videos = {name:[] for name in names}
        videofiles = [f for f in os.listdir(self.clipsfld) if 'avi' in f]
        for name in names:
            if name not in videofiles:
                # we have to create video: first step is identifying the recordings that span the range of frames
                frames_per_vid = [int(v['num frames']) for v in main.session['Metadata'].videodata]
                comulative_frames = np.cumsum(frames_per_vid)

                for frame in exps_frames[name]:
                    idx = np.argmax([frames_per_vid[i] for i,s in enumerate(comulative_frames) if s<=frame])
                    videos[name].append(main.session['Metadata'].video_file_paths[idx][0])

                if videos[name][0] == videos[name][0]:
                    clip = VideoFileClip(videos[name][0])  # cut just that one video
                    exp_clip = clip.subclip(exps_frames[name][0]*clip.fps, exps_frames[name][1]*clip.fps)
                else:
                    clip1 = VideoFileClip(videos[name][0]) # cut both videos and concatenate them
                    clip2 = VideoFileClip(videos[name][0])
                    clip1 = clip1.subclip(exps_frames[name][0]*clip1.fps)
                    clip2 = clip2.subclip(t_end=exps_frames[name][1]*clip2.fps)
                    exp_clip = concatenate_videoclips([clip1, clip2], method="compose")
                # write to video
                exp_clip.write_videofile(os.path.join(self.clipsfld, name), codec='png')

            videos_to_process.append(name)  # store a list of the videos to process so that DLC knows which to skip

        # Do the tracking!
        TF_settings = dlc_setupTF(track_options)
        dlc_analyseVideos.analyse(TF_settings, self.clipsfld, videos_to_process)

        # Retrieve the results from DLC tracking and incorporate in database
        for fname in os.listdir(self.clipsfld):
            if not '.' in fname: continue
            if fname.split('.')[1] == 'h5':
                # Check if data already present in database
                if fname.split('.')[0] in main.session.Tracking.keys(): continue

                # Check if the .h5 file belongs to one of the videos we are analysing
                if not fname in [name.split('.')[0] for name in name]: continue

                # read pandas dataframe (DLC output) and rearrange them for easier access
                Dataframe = pd.read_hdf(os.path.join(self.clipsfld, fname))
                dlc_data = arrange_dlc_data(Dataframe)

                # store in database
                main.session.Tracking[fname.split('.')[0]] = dlc_data







