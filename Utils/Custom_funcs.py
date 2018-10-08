from Utils.imports import *

def cut_crop_video(vidpath='',  save_format=['avi'],
                   cut=False, starts=0., fins=-1., fps=False, duration=False,
                   crop_sel=False, crop_coord=[0, 100, 0, 100], ret=False):
    clip = VideoFileClip(vidpath)

    if not duration: duration = clip.duration
    if not fps: fps = clip.fps
    savename = vidpath.split('.')[0]

    if crop_sel:
        clip = clip.crop(x1=crop_coord[0], width=crop_coord[1], y2=crop_coord[2], height=crop_coord[3])

    if cut:
        clip = clip.subclip(starts, fins)

    if save_format:
        if 'avi' in save_format:
            clip.write_videofile(savename+'_edited'+'.avi', codec='png', fps=fps)
        elif 'mp4' in save_format:
            clip.write_videofile(savename+'_edited'+'.mp4', codec='mpeg4', fps=fps)
        elif 'gif' in save_format:
            clip.write_gif(savename+'_edited3'+'.gif', opt='nq', fps=30)

    if ret:
        return clip


def tile_videos():
    """ given a list of videos tile them to create a composite clip """
    from moviepy.editor import clips_array
    fld = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\data\\z_TrialVideos'
    save_fld = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\Presentations'
    videos = []
    videos.append('48-visual_1-1.avi')
    videos.append('72-visual_0-4.avi')
    videos.append('85-audio_0-1.avi')
    videos.append('90-audio_0-3.avi')
    videos.append('110-audio_0-0.avi')

    edited = []
    for video in videos:
        vid_path = os.path.join(fld, video)
        edit = cut_crop_video(vidpath=vid_path, save_format=False, cut=True, starts=58, fins=71, ret=True)
        edited.append(edit)

    final = clips_array([[edited[0], edited[1], edited[2]],
                         [edited[3], edited[4], edited[4]]])

    random_id = np.random.randint(1, 10000, 1)
    final.write_videofile(os.path.join(save_fld, 'tiled_{}'.format(random_id)) + '.mp4', codec='mpeg4')


def super_impose_videos():
    """ given a list of videos, superimpose and regulate transparency """
    from moviepy.editor import CompositeVideoClip

    fld = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\data\\z_TrialVideos'
    save_fld = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\Presentations'

    videos = []
    videos.append('51-visual_0-0.avi')  # R escape
    videos.append('52-visual_0-0.avi')  # C escape
    videos.append('52-visual_0-1.avi')  # L escape

    edited = []
    alphas = [0.5, 0.6, 0.999]
    for idx, video in enumerate(videos):
        vid_path = os.path.join(fld, video)
        edit = cut_crop_video(vidpath=vid_path, save_format=False, cut=True, starts=58, fins=72, ret=True)
        transparent = edit.set_opacity(alphas[::-1][idx])
        edited.append(transparent)

    overlayed = CompositeVideoClip([edited[0], edited[1], edited[2]])
    random_id = np.random.randint(1, 10000, 1)
    overlayed.write_videofile(os.path.join(save_fld, 'overlayed_{}'.format(random_id[0])) + '.mp4', codec='mpeg4')


########################################################################################################################
if __name__ == "__main__":
    video_to_edit = 'C:\\Users\\Federico\\Documents\\GitHub\\DeepLabCut\\videos' \
                    '\\109-audio_0-0_DeepLabCutlabeled.mp4'

    cut_crop_video(video_to_edit, cut=True, starts=70, fins=76, crop_sel=True , crop_coord=[450, 300, 600, 300],
                   save_format=['mp4'])

    # tile_videos()
    # super_impose_videos()
