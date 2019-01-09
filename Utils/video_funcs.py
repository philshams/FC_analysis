from Utils.imports import *
from Config import track_options

if track_options['track whole session']:
    from deeplabcut.pose_estimation_tensorflow import analyze_videos
    # from deeplabcut.utils import create_labeled_video

import cv2
import numpy as np
import scipy.misc
from termcolor import colored
from tqdm import tqdm
import glob
from Config import y_offset, x_offset


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
        cv2.circle(model_arena,(x_center,y_center),25,150,-1)

    model_arena = cv2.resize(model_arena,size)

    points = np.array(([500,500+460-75],[500-460+75,500],[500,500-460+75],[500+460-75,500]))* [size[0]/1000,size[1]/1000]
                      # , [500 - 504 / 2,500],[500 + 504 / 2,500]))

    return model_arena, points

# =================================================================================
#              IMAGE REGISTRATION GUI
# =================================================================================
def register_arena(background, fisheye_map_location):
    """ extract background: first frame of first video of a session
    Allow user to specify ROIs on the background image """

    # create model arena and background
    arena, arena_points = model_arena(background.shape)

    # load the fisheye correction
    try:
        maps = np.load(fisheye_map_location)
        map1 = maps[:, :, 0:2]
        map2 = maps[:, :, 2]*0

        background_copy = cv2.copyMakeBorder(background, y_offset, int((map1.shape[0] - background.shape[0]) - y_offset),
                                             x_offset, int((map1.shape[1] - background.shape[1]) - x_offset), cv2.BORDER_CONSTANT, value=0)

        background_copy = cv2.remap(background_copy, map1, map2, interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        background_copy = background_copy[y_offset:-int((map1.shape[0] - background.shape[0]) - y_offset),
                          x_offset:-int((map1.shape[1] - background.shape[1]) - x_offset)]
    except:
        background_copy = background.copy()
        fisheye_map_location = ''
        print('fisheye correction not available')

    # initialize clicked points
    blank_arena = arena.copy()
    background_data = [background_copy, np.array(([], [])).T]
    arena_data = [[], np.array(([], [])).T]
    cv2.namedWindow('registered background')
    alpha = .7
    colors = [[150, 0, 150], [0, 255, 0]]
    color_array = make_color_array(colors, background.shape)
    use_loaded_transform = False
    make_new_transform_immediately = False
    use_loaded_points = False

    # LOOP OVER TRANSFORM FILES
    file_num = -1;
    transform_files = glob.glob('.\\transforms\\*transform.npy')
    for file_num, transform_file in enumerate(transform_files[::-1]):

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
        print('Does transform ' + str(file_num+1) + ' / ' + str(len(transform_files)) + ' match this session?')
        print('\'y\' - yes! \'n\' - no. \'q\' - skip examining loaded transforms. \'p\' - update current transform')
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
            elif k == ord('p'):
                use_loaded_points = True
                break
        if use_loaded_transform:
            update_transform_data = [overlaid_arenas,background_data[1], arena_data[1], M]
            break
        elif make_new_transform_immediately or use_loaded_points:
            file_num = len(glob.glob('*transform.npy'))-1
            break

    if not use_loaded_transform:
        if not use_loaded_points:
            print('\nSelect reference points on the experimental background image in the indicated order')

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
        M = cv2.estimateRigidTransform(background_data[1], arena_data[1], False)


        # REGISTER BACKGROUND, BE IT WITH LOADED OR CREATED TRANSFORM
        # registered_background = cv2.warpPerspective(background_copy,M[0],background.shape)
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
        print('\nLeft click model arena // Right click model background')
        print('Purple within arena and green along the boundary represent the model arena')
        print('Advanced users: use arrow keys and \'wasd\' to fine-tune translation and scale')
        print('Crème de la crème: use \'tfgh\' to fine-tune shear\n')
        update_transform_data = [overlaid_arenas,background_data[1], arena_data[1], M, background_copy]

        # create functions to react to additional clicked points
        cv2.setMouseCallback('registered background', additional_transform_points, update_transform_data)

        # take in clicked points until 'q' is pressed
        initial_number_clicked_points = [update_transform_data[1].shape[0], update_transform_data[2].shape[0]]
        M_initial = M
        M_indices = [(0,2),(1,2),(0,0),(1,1),(0,1),(1,0),(2,0),(2,2)]
        # M_indices_meanings = ['x-translate','y-translate','x-scale','y-scale','x->y shear','y->x shear','x perspective','y perspective']
        M_mod_keys = [2424832, 2555904, 2490368, 2621440, ord('w'), ord('a'), ord('s'), ord('d'), ord('f'), ord('t'),
                      ord('g'), ord('h'), ord('j'), ord('i'), ord('k'), ord('l')]
        while True:
            cv2.imshow('registered background',overlaid_arenas)
            cv2.imshow('background', registered_background)
            number_clicked_points = [update_transform_data[1].shape[0], update_transform_data[2].shape[0]]
            update_transform = False
            k = cv2.waitKey(10)
            # If a left and right point are clicked:
            if number_clicked_points[0]>initial_number_clicked_points[0] and number_clicked_points[1]>initial_number_clicked_points[1]:
                initial_number_clicked_points = number_clicked_points
                # update transform and overlay images
                try:
                    # M = cv2.findHomography(update_transform_data[1], update_transform_data[2])
                    M = cv2.estimateRigidTransform(update_transform_data[1], update_transform_data[2],False) #True ~ full transform
                    update_transform = True
                except:
                    continue
            elif k in M_mod_keys: # if an arrow key if pressed
                if k == 2424832: # left arrow - x translate
                    M[M_indices[0]] = M[M_indices[0]] - abs(M_initial[M_indices[0]]) * .005
                elif k == 2555904: # right arrow - x translate
                    M[M_indices[0]] = M[M_indices[0]] + abs(M_initial[M_indices[0]]) * .005
                elif k == 2490368: # up arrow - y translate
                    M[M_indices[1]] = M[M_indices[1]] - abs(M_initial[M_indices[1]]) * .005
                elif k == 2621440: # down arrow - y translate
                    M[M_indices[1]] = M[M_indices[1]] + abs(M_initial[M_indices[1]]) * .005
                elif k == ord('a'): # down arrow - x scale
                    M[M_indices[2]] = M[M_indices[2]] + abs(M_initial[M_indices[2]]) * .005
                elif k == ord('d'): # down arrow - x scale
                    M[M_indices[2]] = M[M_indices[2]] - abs(M_initial[M_indices[2]]) * .005
                elif k == ord('s'): # down arrow - y scale
                    M[M_indices[3]] = M[M_indices[3]] + abs(M_initial[M_indices[3]]) * .005
                elif k == ord('w'): # down arrow - y scale
                    M[M_indices[3]] = M[M_indices[3]] - abs(M_initial[M_indices[3]]) * .005
                elif k == ord('f'): # down arrow - x-y shear
                    M[M_indices[4]] = M[M_indices[4]] - abs(M_initial[M_indices[4]]) * .005
                elif k == ord('h'): # down arrow - x-y shear
                    M[M_indices[4]] = M[M_indices[4]] + abs(M_initial[M_indices[4]]) * .005
                elif k == ord('t'): # down arrow - y-x shear
                    M[M_indices[5]] = M[M_indices[5]] - abs(M_initial[M_indices[5]]) * .005
                elif k == ord('g'): # down arrow - y-x shear
                    M[M_indices[5]] = M[M_indices[5]] + abs(M_initial[M_indices[5]]) * .005

                update_transform = True

            elif  k == ord('r'):
                print('Transformation erased')
                update_transform_data[1] = np.array(([],[])).T
                update_transform_data[2] = np.array(([],[])).T
                initial_number_clicked_points = [3,3]
            elif k == ord('q') or k == ord('y'):
                print('Registration completed')
                break

            if update_transform:
                update_transform_data[3] = M
                # registered_background = cv2.warpPerspective(background_copy, M, background.shape)
                registered_background = cv2.warpAffine(background_copy, M, background.shape)
                registered_background_color = (cv2.cvtColor(registered_background, cv2.COLOR_GRAY2RGB)
                                               * np.squeeze(color_array[:, :, :, 0])).astype(np.uint8)
                overlaid_arenas = cv2.addWeighted(registered_background_color, alpha, arena_color, 1 - alpha, 0)
                update_transform_data[0] = overlaid_arenas

        np.save('.\\transforms\\str(file_num+1)' + '_transform',[M, update_transform_data[1], update_transform_data[2], fisheye_map_location])

    cv2.destroyAllWindows()
    return [M, update_transform_data[1], update_transform_data[2], fisheye_map_location]


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
        # M_inverse = np.linalg.inv(data[3])
        # M_inverse = cv2.findHomography(data[2][:len(data[1])], data[1])
        # transformed_clicks = np.matmul(M_inverse[0], [x, y, 1])

        data[4] = cv2.circle(data[4], (int(transformed_clicks[0]), int(transformed_clicks[1])), 2, (0, 0, 200), -1)
        data[4] = cv2.circle(data[4], (int(transformed_clicks[0]), int(transformed_clicks[1])), 3, 0, 1)

        clicks = np.reshape(transformed_clicks[0:2],(1,2))
        data[1] = np.concatenate((data[1], clicks))
    elif event == cv2.EVENT_LBUTTONDOWN:

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



# # =================================================================================
# #              GENERATE PERI-STIMULUS VIDEO CLIPS and FLIGHT IMAGE
# # =================================================================================
# def peri_stimulus_video_clip(vidpath = '', videoname = '', savepath = '', start_frame=0., end_frame=100., stim_frame = 0,
#                    registration = 0, fps=False, save_clip = False, display_clip = False, counter = True, make_flight_image = True):
#     # GET BEAHVIOUR VIDEO - ######################################
#     vid = cv2.VideoCapture(vidpath)
#     if not fps: fps = vid.get(cv2.CAP_PROP_FPS)
#     width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     # SETUP VIDEO CLIP SAVING - ######################################
#     # file_already_exists = os.path.isfile(os.path.join(savepath,videoname+'.avi'))
#     fourcc = cv2.VideoWriter_fourcc(*"XVID")  # LJPG for lossless, XVID for compressed
#     border_size = 20
#     if save_clip:
#         if not os.path.isdir(savepath):
#             os.makedirs(savepath)
#         video_clip = cv2.VideoWriter(os.path.join(savepath,videoname+'.avi'), fourcc, fps, (width+2*border_size*counter, height+2*border_size*counter), counter)
#     vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
#
#     pre_stim_color = [255, 120, 120]
#     post_stim_color = [120, 120, 255]
#
#     if registration[3]: # setup fisheye correction
#         maps = np.load(registration[3])
#         map1 = maps[:, :, 0:2]
#         map2 = maps[:, :, 2] * 0
#     else:
#         print(colored('Fisheye correction unavailable', 'green'))
#     # RUN SAVING AND ANALYSIS OVER EACH FRAME - ######################################
#     while True: #and not file_already_exists:
#         ret, frame = vid.read()  # get the frame
#         if ret:
#             frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES)
#             if [registration]:
#                 # load the fisheye correction
#                 frame_register = frame[:, :, 0]
#                 if registration[3]:
#                     frame_register = cv2.copyMakeBorder(frame_register, y_offset, int((map1.shape[0] - frame.shape[0]) - y_offset),
#                                                         x_offset, int((map1.shape[1] - frame.shape[1]) - x_offset), cv2.BORDER_CONSTANT, value=0)
#                     frame_register = cv2.remap(frame_register, map1, map2, interpolation=cv2.INTER_LINEAR,
#                                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
#                     frame_register = frame_register[y_offset:-int((map1.shape[0] - frame.shape[0]) - y_offset),
#                                      x_offset:-int((map1.shape[1] - frame.shape[1]) - x_offset)]
#                 frame = cv2.cvtColor(cv2.warpAffine(frame_register, registration[0], frame.shape[0:2]),cv2.COLOR_GRAY2RGB)
#
#             # MAKE ESCAPE TRAJECTORY IMAGE - #######################################
#             if make_flight_image:
#                 # at stimulus onset, take this frame to lay all the superimposed mice on top of
#                 if frame_num == stim_frame:
#                     flight_image_by_distance = frame[:,:,0].copy()
#
#                 # in subsequent frames, see if frame is different enough from previous image to merit joining the image
#                 elif frame_num > stim_frame and (frame_num - stim_frame) < 30*10:
#                     # get the number of pixels that are darker than the flight image
#                     difference_from_previous_image = ((frame[:,:,0]+.001) / (flight_image_by_distance+.001))<.55 #.5 original parameter
#                     number_of_darker_pixels = np.sum(difference_from_previous_image)
#
#                     # if that number is high enough, add mouse to image
#                     if number_of_darker_pixels > 1050: # 850 original parameter
#                         # add mouse where pixels are darker
#                         flight_image_by_distance[difference_from_previous_image] = frame[difference_from_previous_image,0]
#
#             # SHOW BOUNDARY AND TIME COUNTER - #######################################
#             if counter and (display_clip or save_clip):
#                 # cv2.rectangle(frame, (0, height), (150, height - 60), (150,150,150), -1)
#                 if frame_num < stim_frame:
#                     cur_color = tuple([x * ((frame_num - start_frame) / (stim_frame - start_frame)) for x in pre_stim_color])
#                     sign = ''
#                 else:
#                     cur_color = tuple([x * (1 - (frame_num - stim_frame) / (end_frame-stim_frame))  for x in post_stim_color])
#                     sign = '+'
#
#                 # border and colored rectangle around frame
#                 frame = cv2.copyMakeBorder(frame, border_size, border_size, border_size, border_size,cv2.BORDER_CONSTANT, value=cur_color)
#
#                 # report video details
#                 cv2.putText(frame, videoname, (20, 40), 0, .55, (180, 180, 180), thickness=1)
#
#                 # report time relative to stimulus onset
#                 frame_time = (frame_num - stim_frame) / fps
#                 frame_time = str(round(.1*round(frame_time/.1), 1))+ '0'*(abs(frame_time)<10)
#                 cv2.putText(frame, sign + str(frame_time) + 's', (width-110, height+10), 0, 1,(180,180,180), thickness=2)
#
#             else:
#                 frame = frame[:,:,0] # or use 2D grayscale image instead
#
#             # SHOW AND SAVE FRAME - #######################################
#             if display_clip:
#                 cv2.imshow('Trial Clip', frame)
#             if save_clip:
#                 video_clip.write(frame)
#             if display_clip:
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#             if frame_num >= end_frame:
#                 break
#         else:
#             print('Problem with movie playback')
#             cv2.waitKey(1000)
#             break
#
#     # wrap up
#     vid.release()
#     if make_flight_image:
#         flight_image_by_distance = cv2.copyMakeBorder(flight_image_by_distance, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0)
#         cv2.putText(flight_image_by_distance, videoname, (border_size, border_size-5), 0, .55, (180, 180, 180), thickness=1)
#         cv2.imshow('Flight image', flight_image_by_distance)
#         cv2.waitKey(10)
#         scipy.misc.imsave(os.path.join(savepath, videoname + '.tif'), flight_image_by_distance)
#     if save_clip:
#         video_clip.release()
#     # cv2.destroyAllWindows()
#


# =================================================================================
#        GENERATE PERI-STIMULUS VIDEO CLIPS and FLIGHT IMAGE ***with wall***
# =================================================================================
def peri_stimulus_video_clip(vidpath = '', videoname = '', savepath = '', start_frame=0., end_frame=100., stim_frame = 0,
                   registration = 0, fps=False, analyze_wall = True, save_clip = False, display_clip = False, counter = True, make_flight_image = True):
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
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        video_clip = cv2.VideoWriter(os.path.join(savepath,videoname+'.avi'), fourcc, fps, (width+2*border_size*counter, height+2*border_size*counter), counter)

    # set of border colors
    pre_stim_color = [255, 120, 120]
    post_stim_color = [120, 120, 255]

    # setup fisheye correction
    if registration[3]:
        maps = np.load(registration[3])
        map1 = maps[:, :, 0:2]
        map2 = maps[:, :, 2] * 0
    else:
        print(colored('Fisheye correction unavailable', 'green'))

    # generate model arena and wall ROIs
    if analyze_wall:
        # arena = model_arena((width,height))
        x_wall_up_ROI_left = [int(x*width/1000) for x in [223-10,249+10]]
        x_wall_up_ROI_right = [int(x*width/1000) for x in [752-10,777+10]]
        y_wall_up_ROI = [int(x*height/1000) for x in [494 - 10,504 + 10]]

        #check state of wall on various frames
        frames_to_check = [start_frame, stim_frame-1, stim_frame + 13, end_frame]
        wall_darkness = np.zeros((len(frames_to_check), 2))
        for i, frame_to_check in enumerate(frames_to_check):
            vid.set(cv2.CAP_PROP_POS_FRAMES, frame_to_check)
            ret, frame = vid.read()
            frame_register = frame[:, :, 0]
            if registration[3]:
                frame_register = cv2.copyMakeBorder(frame_register, y_offset,
                                                    int((map1.shape[0] - frame.shape[0]) - y_offset),
                                                    x_offset, int((map1.shape[1] - frame.shape[1]) - x_offset),
                                                    cv2.BORDER_CONSTANT, value=0)
                frame_register = cv2.remap(frame_register, map1, map2, interpolation=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                frame_register = frame_register[y_offset:-int((map1.shape[0] - frame.shape[0]) - y_offset),
                                 x_offset:-int((map1.shape[1] - frame.shape[1]) - x_offset)]
            frame = cv2.cvtColor(cv2.warpAffine(frame_register, registration[0], frame.shape[0:2]), cv2.COLOR_GRAY2RGB)
            wall_darkness[i, 0] = sum(
                sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1], x_wall_up_ROI_left[0]:x_wall_up_ROI_left[1], 0] < 200))
            wall_darkness[i, 1] = sum(
                sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1], x_wall_up_ROI_right[0]:x_wall_up_ROI_right[1], 0] < 200))

        wall_darkness_pre = np.min(wall_darkness[0:int(len(frames_to_check)/2),0:2])
        wall_darkness_post = np.min(wall_darkness[int(len(frames_to_check)/2):len(frames_to_check),0:2])

        # use these darkness levels to detect whether wall is up, down, rising, or falling:
        wall_mouse_already_shown = False
        wall_mouse_show = False
        finished_clip = False
        if (wall_darkness_pre - wall_darkness_post) < -30:
            print(colored('Wall rising trial detected!', 'green'))
            wall_height_timecourse = [0]
            trial_type = 1
        elif (wall_darkness_pre - wall_darkness_post) > 30:
            print(colored('Wall falling trial detected!', 'green'))
            wall_height_timecourse = [1]
            trial_type = -1
        elif (wall_darkness_pre > 85) and (wall_darkness_post > 85):
            print(colored('Wall trial detected', 'green'))
            trial_type = 0
            wall_height_timecourse = 1 #[1 for x in list(range(int(fps * .5)))]
        elif (wall_darkness_pre < 85) and (wall_darkness_post < 85):
            print(colored('No Wall trial detected', 'green'))
            trial_type = 0
            wall_height_timecourse = 0 # [0 for x in list(range(int(fps * .5)))]
        else:
            print('Uh-oh -- not sure what kind of trial!')
    else:
        trial_type = 0
    # print(wall_darkness)
    # if not trial_type:
    #     analyze = False
    # else:
    #     analyze = True

    # RUN SAVING AND ANALYSIS OVER EACH FRAME - ######################################
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if display_clip:
        cv2.namedWindow('Trial Clip')
        cv2.moveWindow('Trial Clip',100,100)
    while True: # and analyze: #and not file_already_exists:
        ret, frame = vid.read()  # get the frame
        if ret:
            frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES)
            if [registration]:
                # load the fisheye correction
                frame_register = frame[:, :, 0]
                if registration[3]:
                    frame_register = cv2.copyMakeBorder(frame_register, y_offset, int((map1.shape[0] - frame.shape[0]) - y_offset),
                                                        x_offset, int((map1.shape[1] - frame.shape[1]) - x_offset), cv2.BORDER_CONSTANT, value=0)
                    frame_register = cv2.remap(frame_register, map1, map2, interpolation=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    frame_register = frame_register[y_offset:-int((map1.shape[0] - frame.shape[0]) - y_offset),
                                     x_offset:-int((map1.shape[1] - frame.shape[1]) - x_offset)]
                frame = cv2.cvtColor(cv2.warpAffine(frame_register, registration[0], frame.shape[0:2]),cv2.COLOR_GRAY2RGB)

            # MAKE ESCAPE TRAJECTORY IMAGE - #######################################
            if make_flight_image:
                # at stimulus onset, take this frame to lay all the superimposed mice on top of
                if frame_num == stim_frame:
                    flight_image_by_distance = frame[:,:,0].copy()

                # in subsequent frames, see if frame is different enough from previous image to merit joining the image
                if frame_num >= stim_frame and (frame_num - stim_frame) < fps*10:

                    # get the number of pixels that are darker than the flight image
                    difference_from_previous_image = ((frame[:,:,0]+.001) / (flight_image_by_distance+.001))<.55 #.5 original parameter
                    number_of_darker_pixels = np.sum(difference_from_previous_image)

                    # check on wall progress for .5 sec
                    if trial_type and (frame_num - stim_frame) < fps * .5:
                        left_wall_darkness = sum(sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1],
                                                             x_wall_up_ROI_left[0]:x_wall_up_ROI_left[1], 0] < 200))
                        right_wall_darkness = sum(sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1],
                                                             x_wall_up_ROI_right[0]:x_wall_up_ROI_right[1], 0] < 200))

                        # calculate overall change
                        left_wall_darkness_change_overall = left_wall_darkness - wall_darkness[1,0]
                        right_wall_darkness_change_overall = right_wall_darkness - wall_darkness[1,1]

                        # show wall up timecourse
                        wall_mouse_show = False
                        if trial_type == -1:
                            wall_height = round(1 - np.mean(
                                [left_wall_darkness_change_overall / (wall_darkness[2,0] - wall_darkness[1,0]),
                                 right_wall_darkness_change_overall / (wall_darkness[2,1] - wall_darkness[1,1])]),1)
                            wall_height_timecourse.append(wall_height)

                            # show mouse when wall is down
                            if wall_height <= .1 and not wall_mouse_already_shown:
                                wall_mouse_show = True

                        if trial_type == 1:
                            wall_height = round(np.mean(
                                [left_wall_darkness_change_overall / (wall_darkness[2,0] - wall_darkness[1,0]),
                                 right_wall_darkness_change_overall / (wall_darkness[2,1] - wall_darkness[1,1])]),1)
                            wall_height_timecourse.append(wall_height)

                            # show mouse when wall is up
                            if wall_height >= .6 and not wall_mouse_already_shown:
                                wall_mouse_show = True

                    # if that number is high enough, add mouse to image
                    if number_of_darker_pixels > 950: # 850 original parameter
                        # add mouse where pixels are darker
                        flight_image_by_distance[difference_from_previous_image] = frame[difference_from_previous_image,0]



            # SHOW BOUNDARY AND TIME COUNTER - #######################################
            if counter and (display_clip or save_clip or wall_mouse_show):
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
                frame_time = str(round(.1*round(frame_time/.1), 1))+ '0'*(abs(float(frame_time))<10)
                cv2.putText(frame, sign + str(frame_time) + 's', (width-110, height+10), 0, 1,(180,180,180), thickness=2)

            else:
                frame = frame[:,:,0] # or use 2D grayscale image instead

            # SHOW AND SAVE FRAME - #######################################
            if display_clip:
                cv2.imshow('Trial Clip', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if wall_mouse_show:
                cv2.imshow('Probe moment', frame)
                cv2.waitKey(10)
                scipy.misc.imsave(os.path.join(savepath, videoname + '-probe trial.tif'), frame[:,:,::-1])
                wall_mouse_already_shown = True
            if save_clip:
                video_clip.write(frame)
            if frame_num >= end_frame:
                finished_clip = True
                break
        else:
            print('Problem with movie playback')
            cv2.waitKey(1000)
            break

    # wrap up
    print(wall_height_timecourse)
    vid.release()
    if make_flight_image and finished_clip:
        flight_image_by_distance = cv2.copyMakeBorder(flight_image_by_distance, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0)
        cv2.putText(flight_image_by_distance, videoname, (border_size, border_size-5), 0, .55, (180, 180, 180), thickness=1)
        cv2.imshow('Flight image', flight_image_by_distance)
        cv2.moveWindow('Flight image',1000,100)
        cv2.waitKey(10)
        scipy.misc.imsave(os.path.join(savepath, videoname + '.tif'), flight_image_by_distance)
    if save_clip:
        video_clip.release()
    # cv2.destroyAllWindows()




# =========================================================================================
#        GENERATE PERI-STIMULUS VIDEO CLIPS and FLIGHT IMAGE ***with wall, using DLC***
# =========================================================================================
def peri_stimulus_analysis(coordinates, vidpath = '', videoname = '', savepath = '', start_frame=0., end_frame=100., stim_frame = 0,
                   registration = 0, fps=False, save_clip = False, analyze_wall=True, display_clip = False, counter = True, make_flight_image = True):

    # GET BEHAVIOUR VIDEO - ######################################
    vid = cv2.VideoCapture(vidpath)
    if not fps: fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # SET UP VIDEO CLIP SAVING - ####################################
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # LJPG for lossless, XVID for compressed
    border_size = 20
    if save_clip:
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        video_clip = cv2.VideoWriter(os.path.join(savepath, videoname + '.avi'), fourcc,
                                     fps,(width + 2 * border_size * counter, height + 2 * border_size * counter),counter)

    # setup fisheye correction
    if registration[3]:
        maps = np.load(registration[3])
        map1 = maps[:, :, 0:2]
        map2 = maps[:, :, 2] * 0
    else:
        print(colored('Fisheye correction unavailable', 'green'))

    # set up border colors
    pre_stim_color = [255, 120, 120]
    post_stim_color = [120, 120, 255]

    # generate model arena and wall ROIs
    if analyze_wall:
        wall_height_timecourse, trial_type = initialize_wall_analysis(analyze_wall, stim_frame,  start_frame, end_frame, registration, x_offset, y_offset, map1, map2, vid, width, height)
    wall_mouse_show = False

    # RUN SAVING AND ANALYSIS OVER EACH FRAME - ######################################
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if display_clip:
        cv2.namedWindow('Trial Clip')
        cv2.moveWindow('Trial Clip', 100, 100)
    while True:  # and analyze: #and not file_already_exists:
        ret, frame = vid.read()  # get the frame
        if ret:
            frame_num = int(vid.get(cv2.CAP_PROP_POS_FRAMES))
            if [registration]:
                # load the fisheye correction
                frame = register_frame(frame, x_offset, y_offset, registration, map1, map2)

            # MAKE ESCAPE TRAJECTORY IMAGE - #######################################
            if make_flight_image:
                # at stimulus onset, take this frame to lay all the superimposed mice on top of
                if frame_num == stim_frame:
                    flight_image_by_distance = frame[:, :, 0].copy()
                    arena, _ = model_arena(flight_image_by_distance.shape)

                # in subsequent frames, see if frame is different enough from previous image to merit joining the image
                if frame_num >= stim_frame and (frame_num - stim_frame) < fps * 10:

                    # get the number of pixels that are darker than the flight image
                    difference_from_previous_image = ((frame[:, :, 0] + .001) / (flight_image_by_distance + .001)) < .55  # .5 original parameter
                    number_of_darker_pixels = np.sum(difference_from_previous_image)

                    # check on wall progress for .5 sec
                    if anaylze_wall:
                        wall_mouse_show, wall_height_timecourse = do_wall_analysis(trial_type, frame_num, stim_frame, fps, x_wall_up_ROI_left,
                                         x_wall_up_ROI_right, y_wall_up_ROI, wall_darkness, wall_height_timecourse, wall_mouse_already_shown)

                    # if that number is high enough, add mouse to image
                    if number_of_darker_pixels > 950:  # 850 original parameter
                        # add mouse where pixels are darker
                        flight_image_by_distance[difference_from_previous_image] = frame[
                            difference_from_previous_image, 0]

            # SHOW BOUNDARY AND TIME COUNTER - #######################################
            if counter and (display_clip or save_clip or wall_mouse_show):
                # cv2.rectangle(frame, (0, height), (150, height - 60), (150,150,150), -1)
                if frame_num < stim_frame:
                    cur_color = tuple(
                        [x * ((frame_num - start_frame) / (stim_frame - start_frame)) for x in pre_stim_color])
                    sign = ''
                else:
                    cur_color = tuple(
                        [x * (1 - (frame_num - stim_frame) / (end_frame - stim_frame)) for x in post_stim_color])
                    sign = '+'

                # border and colored rectangle around frame
                frame = cv2.copyMakeBorder(frame, border_size, border_size, border_size, border_size,
                                           cv2.BORDER_CONSTANT, value=cur_color)

                # report video details
                cv2.putText(frame, videoname, (20, 40), 0, .55, (180, 180, 180), thickness=1)

                # report time relative to stimulus onset
                frame_time = (frame_num - stim_frame) / fps
                frame_time = str(round(.1 * round(frame_time / .1), 1)) + '0' * (abs(float(frame_time)) < 10)
                cv2.putText(frame, sign + str(frame_time) + 's', (width - 110, height + 10), 0, 1, (180, 180, 180),
                            thickness=2)

            else:
                frame = frame[:, :, 0]  # or use 2D grayscale image instead

            # SHOW AND SAVE FRAME - #######################################
            if display_clip:
                cv2.imshow('Trial Clip', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if wall_mouse_show:
                cv2.imshow('Probe moment', frame)
                cv2.waitKey(10)
                scipy.misc.imsave(os.path.join(savepath, videoname + '-probe trial.tif'), frame[:, :, ::-1])
                wall_mouse_already_shown = True
            if save_clip:
                video_clip.write(frame)
            if frame_num >= end_frame:
                finished_clip = True
                break
        else:
            print('Problem with movie playback')
            cv2.waitKey(1000)
            break

    # wrap up
    print(wall_height_timecourse)
    vid.release()
    if make_flight_image and finished_clip:
        flight_image_by_distance = cv2.copyMakeBorder(flight_image_by_distance, border_size, border_size,
                                                      border_size, border_size, cv2.BORDER_CONSTANT, value=0)
        cv2.putText(flight_image_by_distance, videoname, (border_size, border_size - 5), 0, .55, (180, 180, 180),
                    thickness=1)
        cv2.imshow('Flight image', flight_image_by_distance)
        cv2.moveWindow('Flight image', 1000, 100)
        cv2.waitKey(10)
        scipy.misc.imsave(os.path.join(savepath, videoname + '.tif'), flight_image_by_distance)
    if save_clip:
        video_clip.release()

def register_frame(frame, x_offset, y_offset, registration, map1, map2):
    '''Go from raw frame to registered frame'''
    frame_register = frame[:, :, 0]

    frame_register = cv2.copyMakeBorder(frame_register, y_offset,
                                        int((map1.shape[0] - frame.shape[0]) - y_offset),
                                        x_offset, int((map1.shape[1] - frame.shape[1]) - x_offset),
                                        cv2.BORDER_CONSTANT, value=0)
    frame_register = cv2.remap(frame_register, map1, map2, interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    frame_register = frame_register[y_offset:-int((map1.shape[0] - frame.shape[0]) - y_offset),
                     x_offset:-int((map1.shape[1] - frame.shape[1]) - x_offset)]



    frame = cv2.cvtColor(cv2.warpAffine(frame_register, registration[0], frame.shape[0:2]),cv2.COLOR_GRAY2RGB)

    return frame


def initialize_wall_analysis(analyze_wall, stim_frame, start_frame, end_frame, registration, x_offset, y_offset, map1, map2, vid, width, height):
    '''determine whether this is a wall up, wall down, wall, or no wall trial'''
    if analyze_wall:
        # arena = model_arena((width,height))
        x_wall_up_ROI_left = [int(x * width / 1000) for x in [223 - 10, 249 + 10]]
        x_wall_up_ROI_right = [int(x * width / 1000) for x in [752 - 10, 777 + 10]]
        y_wall_up_ROI = [int(x * height / 1000) for x in [494 - 10, 504 + 10]]

        # check state of wall on various frames
        frames_to_check = [start_frame, stim_frame - 1, stim_frame + 13, end_frame]
        wall_darkness = np.zeros((len(frames_to_check), 2))
        for i, frame_to_check in enumerate(frames_to_check):
            vid.set(cv2.CAP_PROP_POS_FRAMES, frame_to_check)
            ret, frame = vid.read()
            frame_register = frame[:, :, 0]
            if registration[3]:
                frame_register = cv2.copyMakeBorder(frame_register, y_offset,
                                                    int((map1.shape[0] - frame.shape[0]) - y_offset),
                                                    x_offset, int((map1.shape[1] - frame.shape[1]) - x_offset),
                                                    cv2.BORDER_CONSTANT, value=0)
                frame_register = cv2.remap(frame_register, map1, map2, interpolation=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                frame_register = frame_register[y_offset:-int((map1.shape[0] - frame.shape[0]) - y_offset),
                                 x_offset:-int((map1.shape[1] - frame.shape[1]) - x_offset)]
            frame = cv2.cvtColor(cv2.warpAffine(frame_register, registration[0], frame.shape[0:2]),
                                 cv2.COLOR_GRAY2RGB)
            wall_darkness[i, 0] = sum(
                sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1], x_wall_up_ROI_left[0]:x_wall_up_ROI_left[1], 0] < 200))
            wall_darkness[i, 1] = sum(
                sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1], x_wall_up_ROI_right[0]:x_wall_up_ROI_right[1],
                    0] < 200))

        wall_darkness_pre = np.min(wall_darkness[0:int(len(frames_to_check) / 2), 0:2])
        wall_darkness_post = np.min(wall_darkness[int(len(frames_to_check) / 2):len(frames_to_check), 0:2])

        # use these darkness levels to detect whether wall is up, down, rising, or falling:
        wall_mouse_already_shown = False
        wall_mouse_show = False
        finished_clip = False
        if (wall_darkness_pre - wall_darkness_post) < -30:
            print(colored('Wall rising trial detected!', 'green'))
            wall_height_timecourse = [0]
            trial_type = 1
        elif (wall_darkness_pre - wall_darkness_post) > 30:
            print(colored('Wall falling trial detected!', 'green'))
            wall_height_timecourse = [1]
            trial_type = -1
        elif (wall_darkness_pre > 85) and (wall_darkness_post > 85):
            print(colored('Wall trial detected', 'green'))
            trial_type = 0
            wall_height_timecourse = 1  # [1 for x in list(range(int(fps * .5)))]
        elif (wall_darkness_pre < 85) and (wall_darkness_post < 85):
            print(colored('No Wall trial detected', 'green'))
            trial_type = 0
            wall_height_timecourse = 0  # [0 for x in list(range(int(fps * .5)))]
        else:
            print('Uh-oh -- not sure what kind of trial!')
    else:
        trial_type = 0
        wall_height_timecourse = None

# print(wall_darkness)
# if not trial_type:
#     analyze = False
# else:
#     analyze = True

    return wall_height_timecourse, trial_type


def do_wall_analysis(trial_type, frame_num, stim_frame, fps, x_wall_up_ROI_left, x_wall_up_ROI_right, y_wall_up_ROI, wall_darkness, wall_height_timecourse, wall_mouse_already_shown):

    if trial_type and (frame_num - stim_frame) < fps * .5:
        left_wall_darkness = sum(sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1],
                                     x_wall_up_ROI_left[0]:x_wall_up_ROI_left[1], 0] < 200))
        right_wall_darkness = sum(sum(frame[y_wall_up_ROI[0]:y_wall_up_ROI[1],
                                      x_wall_up_ROI_right[0]:x_wall_up_ROI_right[1], 0] < 200))

        # calculate overall change
        left_wall_darkness_change_overall = left_wall_darkness - wall_darkness[1, 0]
        right_wall_darkness_change_overall = right_wall_darkness - wall_darkness[1, 1]

        # show wall up timecourse
        wall_mouse_show = False
        if trial_type == -1:
            wall_height = round(1 - np.mean(
                [left_wall_darkness_change_overall / (wall_darkness[2, 0] - wall_darkness[1, 0]),
                 right_wall_darkness_change_overall / (wall_darkness[2, 1] - wall_darkness[1, 1])]),
                                1)
            wall_height_timecourse.append(wall_height)

            # show mouse when wall is down
            if wall_height <= .1 and not wall_mouse_already_shown:
                wall_mouse_show = True

        if trial_type == 1:
            wall_height = round(np.mean(
                [left_wall_darkness_change_overall / (wall_darkness[2, 0] - wall_darkness[1, 0]),
                 right_wall_darkness_change_overall / (wall_darkness[2, 1] - wall_darkness[1, 1])]),1)
            wall_height_timecourse.append(wall_height)

            # show mouse when wall is up
            if wall_height >= .6 and not wall_mouse_already_shown:
                wall_mouse_show = True

    return wall_mouse_show, wall_height_timecourse

def invert_fisheye_map(registration, inverse_fisheye_map_location):
    '''Go from a normal opencv fisheye map to an inverted one, so coordinates can be transform'''

    if len(registration) == 5:
        pass
    elif os.path.isfile(inverse_fisheye_map_location):
        registration.append(inverse_fisheye_map_location)
    elif len(registration) == 4:  # setup fisheye correction
        print('creating inverse fisheye map')
        inverse_maps = np.load(registration[3])
        # invert maps
        inverse_maps[inverse_maps < 0] = 0

        maps_x_orig = inverse_maps[:, :, 0]
        maps_x_orig[maps_x_orig > 1279] = 1279
        maps_y_orig = inverse_maps[:, :, 1]
        maps_y_orig[maps_y_orig > 1023] = 1023

        map_x = np.ones(inverse_maps.shape[0:2]) * np.nan
        map_y = np.ones(inverse_maps.shape[0:2]) * np.nan
        for x in range(inverse_maps.shape[1]):
            for y in range(inverse_maps.shape[0]):
                map_x[maps_y_orig[y, x], maps_x_orig[y, x]] = x
                map_y[maps_y_orig[y, x], maps_x_orig[y, x]] = y

        grid_x, grid_y = np.mgrid[0:inverse_maps.shape[0], 0:inverse_maps.shape[1]]
        valid_values_x = np.ma.masked_invalid(map_x)
        valid_values_y = np.ma.masked_invalid(map_y)

        valid_idx_x_map_x = grid_x[~valid_values_x.mask]
        valid_idx_y_map_x = grid_y[~valid_values_x.mask]

        valid_idx_x_map_y = grid_x[~valid_values_y.mask]
        valid_idx_y_map_y = grid_y[~valid_values_y.mask]

        map_x_interp = interpolate.griddata((valid_idx_x_map_x, valid_idx_y_map_x), map_x[~valid_values_x.mask],
                                            (grid_x, grid_y), method='linear').astype(np.uint16)
        map_y_interp = interpolate.griddata((valid_idx_x_map_y, valid_idx_y_map_y), map_y[~valid_values_y.mask],
                                            (grid_x, grid_y), method='linear').astype(np.uint16)

        fisheye_maps_interp = np.zeros((map_x_interp.shape[0], map_x_interp.shape[1], 2)).astype(np.uint16)
        fisheye_maps_interp[:, :, 0] = map_x_interp
        fisheye_maps_interp[:, :, 1] = map_y_interp

        np.save('C:\\Drive\\DLC\\transforms\\inverse_fisheye_maps.npy', fisheye_maps_interp)

    return registration


def extract_coordinates_with_dlc(dlc_config_settings, video, registration):
    '''extract coordinates for each frame, given a video and DLC network'''

    analyze_videos(dlc_config_settings['config_file'], video)
    # create_labeled_video(dlc_config_settings['config_file'], video)

    # read the freshly saved coordinates file
    coordinates_file = glob.glob(os.path.dirname(video[0]) + '\\*.h5')[0]
    DLC_network = os.path.basename(coordinates_file)
    DLC_network = DLC_network[DLC_network.find('Deep'):-3]
    body_parts = dlc_config_settings['body parts']

    DLC_dataframe = pd.read_hdf(coordinates_file)

    # plot body part positions over time
    import matplotlib.pyplot as plt
    position_figure = plt.figure('position over time')
    ax = position_figure.add_subplot(111)
    plt.title('DLC positions')
    plt.xlabel('frame number')
    plt.ylabel('position (pixels)')
    coordinates = {}
    for body_part in body_parts:
        # initialize coordinates
        coordinates[body_part] = np.zeros((2, len(DLC_dataframe[DLC_network][body_part]['x'].values)))

        # extract coordinates
        for i, axis in enumerate(['x', 'y']):
            coordinates[body_part][i] = DLC_dataframe[DLC_network][body_part][axis].values
            coordinates[body_part][i] = DLC_dataframe[DLC_network][body_part][axis].values

            likelihood = DLC_dataframe[DLC_network][body_part]['likelihood'].values

        # remove coordinates with low confidence
        coordinates[body_part][0][likelihood < .999999] = np.nan
        coordinates[body_part][1][likelihood < .999999] = np.nan

        # lineraly interporlate the low-confidence time points
        coordinates[body_part][0] = np.array(pd.Series(coordinates[body_part][0]).interpolate())
        coordinates[body_part][0][0:np.argmin(np.isnan(coordinates[body_part][0]))] = coordinates[body_part][0][
            np.argmin(np.isnan(coordinates[body_part][0]))]

        coordinates[body_part][1] = np.array(pd.Series(coordinates[body_part][1]).interpolate())
        coordinates[body_part][1][0:np.argmin(np.isnan(coordinates[body_part][1]))] = coordinates[body_part][1][
            np.argmin(np.isnan(coordinates[body_part][1]))]

        # fisheye correct the coordinates
        registration = invert_fisheye_map(registration, dlc_config_settings['inverse_fisheye_map_location'])
        inverse_fisheye_maps = np.load(registration[4])

        # convert original coordinates to registered coordinates
        coordinates[body_part][0] = inverse_fisheye_maps[
                                        coordinates[body_part][1].astype(np.uint16) + y_offset, coordinates[body_part][
                                            0].astype(np.uint16) + x_offset, 0] - x_offset
        coordinates[body_part][1] = inverse_fisheye_maps[
                                        coordinates[body_part][1].astype(np.uint16) + y_offset, coordinates[body_part][
                                            0].astype(np.uint16) + x_offset, 1] - y_offset

        # affine transform to match model arena
        transformed_points = np.matmul(np.append(registration[0], np.zeros((1, 3)), 0),
                                       np.concatenate((coordinates[body_part][0:1], coordinates[body_part][1:2],
                                                       np.ones((1, len(coordinates[body_part][0])))), 0))

        # plot the coordinates
        ax.plot(np.sqrt((transformed_points[0] - 500 * 720 / 1000) ** 2 + (transformed_points[1] - 885 * 720 / 1000) ** 2))

    return coordinates, registration

########################################################################################################################
if __name__ == "__main__":
    peri_stimulus_video_clip(vidpath='', videoname='', savepath='', start_frame=0., end_frame=100., stim_frame=0,
                             registration=0, fps=False, save_clip=False, display_clip=False, counter=True,
                             make_flight_image=True)