from Utils.imports import *
import cv2
import numpy as np
import scipy.misc
from termcolor import colored
from tqdm import tqdm
import glob

def get_background(vidpath, start_frame = 1000, avg_over = 100):
    """ extract background: average over frames of video """

    vid = cv2.VideoCapture(vidpath)

    # initialize the video
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    background = np.zeros((height, width))
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # initialize the counters
    every_other = int(num_frames / avg_over)
    j = 0

    for i in tqdm(range(num_frames)):

        if i % every_other == 0:
            vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = vid.read()  # get the frame

            if ret:
                # store the current frame in as a numpy array
                background += frame[:, :, 0]
                j+=1


    background = (background / (j)).astype(np.uint8)
    cv2.imshow('background', background)
    cv2.waitKey(10)
    vid.release()

    return background


def model_arena(size):

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

    model_arena = cv2.resize(model_arena,size)

    points = np.array(([500,500+460-75],[500-460+75,500],[500,500-460+75],[500+460-75,500],
                        [500 - 504 / 2,500],[500 + 504 / 2,500])) * [size[0]/1000,size[1]/1000]

    return model_arena, points

def register_arena(background):
    """ extract background: first frame of first video of a session
    Allow user to specify ROIs on the background image """

    # create model arena
    arena, arena_points = model_arena(background.shape)

    # initialize clicked points
    background_copy = background.copy()
    blank_arena = arena.copy()
    background_data = [background_copy, np.array(([], [])).T]
    arena_data = [[], np.array(([], [])).T]
    cv2.namedWindow('registered background')
    alpha = .7
    colors = [[150, 0, 150], [0, 255, 0]]
    color_array = make_color_array(colors, background.shape)
    use_loaded_transform = False
    make_new_transform_immediately = False

    # LOOP OVER TRANSFORM FILES
    file_num = -1;
    for file_num, transform_file in enumerate(glob.glob('*transform.npy')):

        # USE LOADED TRANSFORM AND SEE IF IT'S GOOD
        loaded_transform = np.load(transform_file)
        M = loaded_transform[0]
        background_data[1] = loaded_transform[1]
        arena_data[1] = loaded_transform[2]

        # registered_background = cv2.warpPerspective(background_copy, M, background.shape)
        registered_background = cv2.warpAffine(background_copy, M, background.shape)
        registered_background_color = (cv2.cvtColor(registered_background, cv2.COLOR_GRAY2RGB)
                                       * np.squeeze(color_array[:, :, :, 0])).astype(np.uint8)
        arena_color = (cv2.cvtColor(blank_arena, cv2.COLOR_GRAY2RGB)
                       * np.squeeze(color_array[:, :, :, 1])).astype(np.uint8)

        overlaid_arenas = cv2.addWeighted(registered_background_color, alpha, arena_color, 1 - alpha, 0)
        print('Does the transform match this session? (y/n)')
        while True:
            cv2.imshow('registered background', overlaid_arenas)
            k = cv2.waitKey(10)
            if  k == ord('n'):
                break
            elif k == ord('y'):
                use_loaded_transform = True
                break
            elif k == ord('q'):
                make_new_transform_immediately = True
                break
        if use_loaded_transform:
            update_transform_data = [overlaid_arenas,background_data[1], arena_data[1], M]
            break
        elif make_new_transform_immediately:
            file_num = len(glob.glob('*transform.npy'))-1
            break

    if not use_loaded_transform:
        print('Select reference points to generate new transform')
        # initialize clicked point arrays
        background_data = [background_copy, np.array(([], [])).T]
        arena_data = [[], np.array(([], [])).T]

        # add 1-2-3-4 markers to model arena
        for i, point in enumerate(arena_points.astype(np.uint32)):
            arena = cv2.circle(arena, (point[0], point[1]), 3, 255, -1)
            arena = cv2.circle(arena, (point[0], point[1]), 4, 0, 1)
            cv2.putText(arena, str(i+1), tuple(point), 0, .55, 150, thickness=2)

            point = np.reshape(point, (1, 2))
            arena_data[1] = np.concatenate((arena_data[1], point))

        # initialize GUI
        cv2.startWindowThread()
        cv2.namedWindow('background')
        cv2.imshow('background', background_copy)
        cv2.namedWindow('model arena')
        cv2.imshow('model arena', arena)

        # create functions to react to clicked points
        cv2.setMouseCallback('background', select_transform_points, background_data)  # Mouse callback

        while True: # take in clicked points until four points are clicked
            cv2.imshow('background',background_copy)

            number_clicked_points = background_data[1].shape[0]
            if number_clicked_points == len(arena_data[1]):
                break
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # perform projective transform
        # M = cv2.findHomography(background_data[1], arena_data[1])
        M = cv2.estimateRigidTransform(background_data[1], arena_data[1],True)


        # REGISTER BACKGROUND, BE IT WITH LOADED OR CREATED TRANSFORM
        # registered_background = cv2.warpPerspective(background_copy,M,background.shape)
        registered_background = cv2.warpAffine(background_copy, M, background.shape)

        # --------------------------------------------------
        # overlay images
        # --------------------------------------------------
        alpha = .7
        colors = [[150, 0, 150], [0, 255, 0]]
        color_array = make_color_array(colors, background.shape)

        registered_background_color = (cv2.cvtColor(registered_background, cv2.COLOR_GRAY2RGB)
                                 * np.squeeze(color_array[:, :, :, 0])).astype(np.uint8)
        arena_color = (cv2.cvtColor(blank_arena, cv2.COLOR_GRAY2RGB)
                       * np.squeeze(color_array[:, :, :, 1])).astype(np.uint8)

        overlaid_arenas = cv2.addWeighted(registered_background_color, alpha, arena_color, 1 - alpha, 0)
        cv2.imshow('registered background', overlaid_arenas)

        # --------------------------------------------------
        # initialize GUI for correcting transform
        # --------------------------------------------------
        print('Left click model arena // Right click model background')
        print('Purple within arena and green along the boundary represent the model arena')
        update_transform_data = [overlaid_arenas,background_data[1], arena_data[1], M, background_copy]

        # create functions to react to additional clicked points
        cv2.setMouseCallback('registered background', additional_transform_points, update_transform_data)

        # take in clicked points until 'q' is pressed
        initial_number_clicked_points = [update_transform_data[1].shape[0], update_transform_data[2].shape[0]]
        while True:
            cv2.imshow('registered background',overlaid_arenas)
            number_clicked_points = [update_transform_data[1].shape[0], update_transform_data[2].shape[0]]

            k = cv2.waitKey(10)
            # If a left and right point are clicked:
            if number_clicked_points[0]>initial_number_clicked_points[0] and number_clicked_points[1]>initial_number_clicked_points[1]:
                initial_number_clicked_points = number_clicked_points
                # update transform and overlay images
                # M = cv2.findHomography(update_transform_data[1], update_transform_data[2])
                try:
                    M = cv2.estimateRigidTransform(update_transform_data[1], update_transform_data[2],True)

                    update_transform_data[3] = M
                    # registered_background = cv2.warpPerspective(background_copy, M, background.shape)
                    registered_background = cv2.warpAffine(background_copy, M, background.shape)
                    registered_background_color = (cv2.cvtColor(registered_background, cv2.COLOR_GRAY2RGB)
                                               * np.squeeze(color_array[:, :, :, 0])).astype(np.uint8)
                    overlaid_arenas = cv2.addWeighted(registered_background_color, alpha, arena_color, 1 - alpha, 0)
                    update_transform_data[0] = overlaid_arenas
                except:
                    continue

            elif  k == ord('r'):
                print('Transformation erased')
                update_transform_data[1] = np.array(([],[])).T
                update_transform_data[2] = np.array(([],[])).T
                initial_number_clicked_points = [3,3]
            elif k == ord('q'):
                print('Registration completed')
                break

        np.save(str(file_num+1)+'_transform',[M, update_transform_data[1], update_transform_data[2]])

    return [M, update_transform_data[1], update_transform_data[2]]


# mouse callback function I
def select_transform_points(event,x,y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:

        data[0] = cv2.circle(data[0], (x, y), 3, 255, -1)
        data[0] = cv2.circle(data[0], (x, y), 4, 0, 1)

        clicks = np.reshape(np.array([x, y]),(1,2))
        data[1] = np.concatenate((data[1], clicks))

# mouse callback function II
def additional_transform_points(event,x,y, flags, data):
    if event == cv2.EVENT_RBUTTONDOWN:

        data[0] = cv2.circle(data[0], (x, y), 2, (200,0,0), -1)
        data[0] = cv2.circle(data[0], (x, y), 3, 0, 1)

        M_inverse = cv2.invertAffineTransform(data[3])
        transformed_clicks = np.matmul(np.append(M_inverse,np.zeros((1,3)),0), [x, y, 1])

        # data[4] = cv2.circle(data[4], (int(transformed_clicks[0]), int(transformed_clicks[1])), 2, (0, 0, 200), -1)
        # data[4] = cv2.circle(data[4], (int(transformed_clicks[0]), int(transformed_clicks[1])), 3, 0, 1)
        # cv2.imshow('sup',data[4])
        # print(x,y)
        # print(transformed_clicks)

        clicks = np.reshape(transformed_clicks[0:2],(1,2))
        data[1] = np.concatenate((data[1], clicks))
    if event == cv2.EVENT_LBUTTONDOWN:

        data[0] = cv2.circle(data[0], (x, y), 2, (0,200,200), -1)
        data[0] = cv2.circle(data[0], (x, y), 3, 0, 1)

        clicks = np.reshape(np.array([x, y]),(1,2))
        data[2] = np.concatenate((data[2], clicks))



def make_color_array(colors, image_size):
    color_array = np.zeros((image_size[0],image_size[1], 3, len(colors)))  # create coloring arrays
    for c in range(len(colors)):
        for i in range(3):  # B, G, R
            color_array[:, :, i, c] = np.ones((image_size[0],image_size[1])) * colors[c][i] / sum(
                colors[c])
    return color_array


def cut_crop_video(vidpath = '', videoname = '', savepath = '', start_frame=0., end_frame=100., stim_frame = 0,
                   registration = 0, fps=False, save_clip = False, display_clip = False, counter = True, make_flight_image = True):
    # GET BEAHVIOUR VIDEO - ######################################
    vid = cv2.VideoCapture(vidpath)
    if not fps: fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # SETUP VIDEO CLIP SAVING - ######################################
    # file_already_exists = os.path.isfile(os.path.join(savepath,videoname+'.avi'))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # LJPG for lossless, XVID for compressed
    border_size = 20
    if save_clip:
        video_clip = cv2.VideoWriter(os.path.join(savepath,videoname+'.avi'), fourcc, fps, (width+2*border_size, height+2*border_size), counter)
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    pre_stim_color = [255, 120, 120]
    post_stim_color = [120, 120, 255]
    prev_frame = stim_frame

    # RUN SAVING AND ANALYSIS OVER EACH FRAME - ######################################
    while True: #and not file_already_exists:
        ret, frame = vid.read()  # get the frame
        if not not [registration]:
            frame = cv2.cvtColor(cv2.warpAffine(frame[:,:,0], registration, frame[:,:,0].shape),cv2.COLOR_GRAY2RGB)
        if ret:
            frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES) # frame number

            # MAKE ESCAPE TRAJECTORY IMAGE - #######################################
            if make_flight_image:
                # at stimulus onset, take this frame to lay all the superimposed mice on top of
                if frame_num == stim_frame:
                    flight_image_by_distance = frame[:,:,0].copy()
                    flight_image_by_time = frame[:, :, 0].copy()

                # in subsequent frames, see if frame is different enough from previous image to merit joining the image
                elif frame_num > stim_frame:
                    # get the number of pixels that are darker than the flight image
                    difference_from_previous_image = ((frame[:,:,0]+.001) / (flight_image_by_distance+.001))<.55 #.5 original parameter
                    number_of_darker_pixels = np.sum(difference_from_previous_image)

                    # if that number is high enough, add mouse to image
                    if number_of_darker_pixels > 950: # 850 original parameter
                        # add mouse where pixels are darker
                        flight_image_by_distance[difference_from_previous_image] = frame[difference_from_previous_image,0]

                    # try doing it by time
                    if (frame_num - prev_frame) >= 7:
                        difference_from_previous_image = ((frame[:, :, 0] + .001) / (flight_image_by_time + .001)) < .6
                        flight_image_by_time[difference_from_previous_image] = frame[difference_from_previous_image, 0]
                        prev_frame = frame_num

            # SHOW BOUNDARY AND TIME COUNTER - #######################################
            if counter and (display_clip or save_clip):
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
                cv2.putText(frame, videoname, (20, 40), 0, .55, (180, 180, 180), thickness=1)

                # report time relative to stimulus onset
                frame_time = (frame_num - stim_frame) / fps
                frame_time = str(round(.2*round(frame_time/.2), 1))+ '0'*(abs(frame_time)<10)
                cv2.putText(frame, sign + str(frame_time) + 's', (width-110, height+10), 0, 1,(180,180,180), thickness=2)

            else:
                frame = frame[:,:,0] # or use 2D grayscale image instead

            # SHOW AND SAVE FRAME - #######################################
            if display_clip:
                cv2.imshow('Trial Clip', frame)
            if save_clip:
                video_clip.write(frame)
            if display_clip:
                if cv2.waitKey(1) & 0xFF == ord('q'):
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
        flight_image_by_distance = cv2.copyMakeBorder(flight_image_by_distance, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0)
        flight_image_by_time = cv2.copyMakeBorder(flight_image_by_time, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0)
        cv2.putText(flight_image_by_distance, videoname, (border_size, border_size-5), 0, .55, (180, 180, 180), thickness=1)
        cv2.putText(flight_image_by_time, videoname, (border_size, border_size - 5), 0, .55, (180, 180, 180), thickness=1)
        cv2.imshow('Flight image by distance', flight_image_by_distance)
        cv2.imshow('Flight image by time', flight_image_by_time)
        cv2.waitKey(10)
        scipy.misc.imsave(os.path.join(savepath, videoname + '_by_distance.tif'), flight_image_by_distance)
        scipy.misc.imsave(os.path.join(savepath, videoname + '_by_time.tif'), flight_image_by_time)
    if save_clip:
        video_clip.release()
    # cv2.destroyAllWindows()





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
