from moviepy.editor import VideoFileClip
from moviepy.video.fx import crop


def cut_crop_video(vidpath='',  save_format=['avi'],
                   cut=False, starts=0., fins=-1.,
                   crop_sel=False, crop_coord=[0, 100, 0, 100], ret=False):
    clip = VideoFileClip(vidpath)

    duration = clip.duration
    fps = clip.fps
    savename = vidpath.split('.')[0]

    if crop_sel:
        clip = clip.crop(x1=crop_coord[0], width=crop_coord[1], y2=crop_coord[2], height=crop_coord[3])

    if cut:
        clip = clip.subclip(starts, fins)

    if save_format:
        if 'avi' in save_format:
            clip.write_videofile(savename+'_edited'+'.avi', codec='png')
        elif 'mp4' in save_format:
            clip.write_videofile(savename+'_edited'+'.mp4', codec='mpeg4')
        elif 'gif' in save_format:
            clip.write_gif(savename+'_edited_'+'.gif', fps=30)

    if ret:
        return clip


########################################################################################################################
if __name__ == "__main__":
    video_to_edit = 'C:\\Users\\Federico\\Documents\\GitHub\\DeepLabCut\\videos\\Zina_twomice.avi'

    cut_crop_video(video_to_edit, cut=True, starts=1110, fins=1210, crop_sel=False, crop_coord=[150, 300, 450, 300],
                   save_format=['avi'])



