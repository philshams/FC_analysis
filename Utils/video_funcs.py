from Utils.imports import *
import cv2
import numpy as np
import scipy.misc
from termcolor import colored


def get_background(vidpath, start_frame = 1000, avg_over = 100):
    """ extract background: average over frames of video """

    print(colored('Extracting background:','green'))
    vid = cv2.VideoCapture(vidpath)

    # initialize the video
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    background = np.zeros((height, width)).astype(np.uint8)
    num_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # initialize the counters
    every_other = int(num_frames / avg_over)
    i = 0
    j = 0

    print('computing mean background...')
    while True:
        i += 1
        if i % every_other == 0:
            vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = vid.read()  # get the frame

            if ret:
                # store the current frame in as a numpy array
                background_mat[:, :, 0] += frame[:, :, 1]
                if two_videos and stereo:
                    background_mat[:, :, 1] += frame2[:, :, 1]
                elif stereo:
                    background_mat[:, :, 1] += frame[:, :, 2]
                j += 1

                if j >= avg_over:
                    break

                if avg_over > 200:
                    if j % 100 == 0:
                        print(str(j) + ' frames out of ' + str(avg_over) + ' done')
                else:
                    if j % 10 == 0:
                        print(str(j) + ' frames out of ' + str(avg_over) + ' done')

        if vid.get(cv2.CAP_PROP_POS_FRAMES) + 1 >= vid.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is almost equal to the total number of frames, stop
            break

    background_mat = background_mat / j
    cv2.imshow('background', background_mat.astype(np.uint8))
    vid.release()

    return background


def model_arena():

    # initialize model arena
    model_arena = np.zeros((1000,1000)).astype(np.uint8)
    cv2.circle(model_arena, (500,500), 460, 255, -1)

    # add wall - up
    cv2.rectangle(model_arena, (int(500 - 554 / 2), int(500 - 6 / 2)), (int(500 + 554 / 2), int(500 + 6 / 2)), 60, thickness=-1)
    # add wall - down
    cv2.rectangle(model_arena, (int(500 - 504 / 2), int(500 - 8 / 2)), (int(500 + 504 / 2), int(500 + 8 / 2)), 0, thickness=-1)

    # add shelter
    model_arena_shelter = model_arena.copy()
    cv2.rectangle(model_arena_shelter, (int(500 - 50), int(500 + 385 + 25 - 50)), (int(500 + 50), int(500 + 385 + 25 + 50)), (0, 0, 255),thickness=-1)
    alpha = .5
    cv2.addWeighted(model_arena, alpha, model_arena_shelter, 1 - alpha, 0, model_arena)

    # add circular wells along edge
    number_of_circles = 20
    for circle_num in range(number_of_circles):
        x_center = int(500+385*np.sin(2*np.pi/number_of_circles*circle_num))
        y_center = int(500-385*np.cos(2*np.pi/number_of_circles*circle_num))
        cv2.circle(model_arena,(x_center,y_center),25,0,-1)

    cv2.imshow('arena', model_arena)

    return model_arena

def register_arena():
    """ extract background: first frame of first video of a session
    Allow user to specify ROIs on the background image """
    print(colored('Extracting ROIs:','green'))

    model_arena = model_arena()

    cv2.startWindowThread()
    if len(background.shape) == 3:
        gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    else:
        gray = background

    blur = cv2.blur(gray, (15, 15))
    edges = cv2.Canny(blur, 25, 30)

    cv2.namedWindow('background')
    cv2.imshow('background', gray)

    rois = {'Shelter': None}
    if track_options['bg get rois']:          # Get user to define Shelter ROI
        for rname in rois.keys():
            print(colored('Please mark {}'.format(rname),'green'))
            # rois[rname] = cv2.selectROI(gray, fromCenter=False)

            cv2.setMouseCallback('background', mouse_callback, 0)  # Mouse callback

            while True:
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            print(colored('   done', 'green'))

    return edges, rois


# mouse callback function
def mouse_callback(event,x,y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)

def cut_crop_video(vidpath='', videoname = '', savepath = '', start_frame=0., end_frame=100., stim_frame = 0, fps=False,
                   save_movie = False, counter = True, display_clip = False, make_flight_image = True):
    # GET BEAHVIOUR VIDEO - ######################################
    vid = cv2.VideoCapture(vidpath)
    if not fps: fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # SETUP VIDEO CLIP SAVING - ######################################
    # file_already_exists = os.path.isfile(os.path.join(savepath,videoname+'.avi'))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # LJPG for lossless, XVID for compressed
    border_size = 20
    if save_movie:
        video_clip = cv2.VideoWriter(os.path.join(savepath,videoname+'.avi'), fourcc, fps, (width+2*border_size, height+2*border_size), counter)
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    pre_stim_color = [255, 120, 120]
    post_stim_color = [120, 120, 255]

    # RUN SAVING AND ANALYSIS OVER EACH FRAME - ######################################
    while True: #and not file_already_exists:
        ret, frame = vid.read()  # get the frame
        if ret:
            frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES) # frame number

            # MAKE ESCAPE TRAJECTORY IMAGE - #######################################
            if make_flight_image:
                # at stimulus onset, take this frame to lay all the superimposed mice on top of
                if frame_num == stim_frame:
                    flight_image = frame[:,:,0]

                # in subsequent frames, see if frame is different enough from previous image to merit joining the image
                elif frame_num > stim_frame:
                    # get the number of pixels that are darker than the flight image
                    difference_from_previous_image = (frame[:,:,0] / flight_image)<.5
                    number_of_darker_pixels = np.sum(difference_from_previous_image)

                    # if that number is high enough, add mouse to image
                    if number_of_darker_pixels > 850:
                        # add mouse where pixels are darker
                        flight_image[difference_from_previous_image] = frame[difference_from_previous_image,0]

            # SHOW BOUNDARY AND TIME COUNTER - #######################################
            if counter:
                # cv2.rectangle(frame, (0, height), (150, height - 60), (150,150,150), -1)
                if frame_num < stim_frame:
                    cur_color = tuple([x * ((frame_num - start_frame) / (stim_frame - start_frame)) for x in pre_stim_color])
                    sign = ''
                else:
                    cur_color = tuple([x * (1 - (frame_num - stim_frame) / (end_frame-stim_frame))  for x in post_stim_color])
                    sign = '+'

                # border and colored rectangle around frame
                frame = cv2.copyMakeBorder(frame, border_size, border_size, border_size, border_size,cv2.BORDER_CONSTANT, value=cur_color)

                # report video details
                cv2.putText(frame, videoname, (40, height+10), 0, .55, (180, 180, 180), thickness=1)

                # report time relative to stimulus onset
                frame_time = (frame_num - stim_frame) / fps
                frame_time = str(round(.2*round(frame_time/.2), 1))+ '0'*(abs(frame_time)<10)
                cv2.putText(frame, sign + str(frame_time) + 's', (width-110, height+10), 0, 1,(180,180,180), thickness=2)

            else:
                frame = frame[:,:,0] # or use 2D grayscale image instead

            # SHOW AND SAVE FRAME - #######################################
            if display_clip:
                cv2.imshow('Trial Clip', frame)
            if save_movie:
                video_clip.write(frame)
            if display_clip & cv2.waitKey(5) & 0xFF == ord('q'):
                break
            if frame_num >= end_frame:
                break
        else:
            print('Problem with movie playback')
            cv2.waitKey(1000)
            break

    # wrap up
    vid.release()
    if make_flight_image:
        scipy.misc.imsave(os.path.join(savepath,videoname+'.tif'), flight_image)
    if save_movie:
        video_clip.release()
    cv2.destroyAllWindows()





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
