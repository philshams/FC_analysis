import os.path
import sys

subfolder = 'C:\\Users\\Federico\\Documents\\GitHub\\DeepLabCut'

# add parent directory: (where nnet & config are!)
sys.path.append(os.path.join(subfolder, "pose-tensorflow"))
sys.path.append(os.path.join(subfolder, "Generating_a_Training_Set"))

from Tracking.dlc_analysis_config import cropping, Task, date, \
    trainingsFraction, resnet, snapshotindex, shuffle,x1, x2, y1, y2, videotype, storedata_as_csv

# Deep-cut dependencies
from Tracking.dlc_config import load_config
from nnet import predict
from dataset.pose_dataset import data_to_input

from Tracking.Tracking_config import dlc_config_settings


# Dependencies for video:
import pickle
# import matplotlib.pyplot as plt
import imageio
imageio.plugins.ffmpeg.download()
from skimage.util import img_as_ubyte
from moviepy.editor import VideoFileClip
import skimage
import skimage.color
import time
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


def getpose(sess, inputs, image, cfg, outputs, outall=False):
    ''' Adapted from DeeperCut, see pose-tensorflow folder'''
    image_batch = data_to_input(skimage.color.gray2rgb(image))
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref = predict.extract_cnn_output(outputs_np, cfg)
    pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
    if outall:
        return scmap, locref, pose
    else:
        return pose


def analyse():
    ####################################################
    # Loading data, and defining model folder
    ####################################################
    cfg = load_config(dlc_config_settings['dlc_network_path'])
    cfg['init_weights'] = dlc_config_settings['dlc_network_snapshot']

    ##################################################
    # Compute predictions over images
    ##################################################
    scorer = dlc_config_settings['scorer']
    sess, inputs, outputs = predict.setup_pose_prediction(cfg)
    pdindex = pd.MultiIndex.from_product(
        [[scorer], cfg['all_joints_names'], ['x', 'y', 'likelihood']],
        names=['scorer', 'bodyparts', 'coords'])

    videofolder = os.path.join(subfolder, 'videos')
    frame_buffer = 10

    os.chdir(videofolder)
    videos = np.sort([fn for fn in os.listdir(os.curdir) if (videotype in fn)])
    print("Starting ", videofolder, videos)
    for video in videos:
        dataname = video.split('.')[0] + scorer + '.h5'
        try:
            # Attempt to load data...
            pd.read_hdf(dataname)
            print("Video already analyzed!", dataname)
        except FileNotFoundError:
            print("Loading ", video)
            clip = VideoFileClip(video)
            ny, nx = clip.size  # dimensions of frame (height, width)
            fps = clip.fps
            # nframes = np.sum(1 for j in clip.iter_frames()) #this is slow (but accurate)
            nframes_approx = int(np.ceil(clip.duration * clip.fps) + frame_buffer)
            # this will overestimage number of frames (see https://github.com/AlexEMG/DeepLabCut/issues/9) This is especially a problem
            # for high frame rates and long durations due to rounding errors (as Rich Warren found). Later we crop the result (line 187)

            if cropping:
                clip = clip.crop(
                    y1=y1, y2=y2, x1=x1, x2=x2)  # one might want to adjust

            print("Duration of video [s]: ", clip.duration, ", recorded with ", fps,
                  "fps!")
            print("Overall # of frames: ", nframes_approx, "with cropped frame dimensions: ", clip.size)

            start = time.time()
            PredicteData = np.zeros((nframes_approx, 3 * len(cfg['all_joints_names'])))

            temp_image = img_as_ubyte(clip.get_frame(0))
            scmap, locref, pose = getpose(sess, inputs, temp_image, cfg, outputs, outall=True)
            PredictedScmap = np.zeros((nframes_approx, scmap.shape[0], scmap.shape[1], len(cfg['all_joints_names'])))

            print("Starting to extract posture")
            for index in tqdm(range(nframes_approx)):
                # image = img_as_ubyte(clip.get_frame(index * 1. / fps))
                image = img_as_ubyte(clip.reader.read_frame())
                # Thanks to Rick Warren for the  following snipplet:
                # if close to end of video, start checking whether two adjacent frames are identical
                # this should only happen when moviepy has reached the final frame
                # if two adjacent frames are identical, terminate the loop
                if index == int(nframes_approx - frame_buffer * 2):
                    last_image = image
                elif index > int(nframes_approx - frame_buffer * 2):
                    if (image == last_image).all():
                        nframes = index
                        print("Detected frames: ", nframes)
                        break
                    else:
                        last_image = image
                try:
                    pose = getpose(sess, inputs,image, cfg, outputs, outall=True)
                    PredicteData[index, :] = pose.flatten()
                except:
                    scmap, locref, pose = getpose(sess, inputs, image, cfg, outputs, outall=True)
                    PredicteData[index, :] = pose.flatten()
                    PredictedScmap[index, :, :, :] = scmap

            stop = time.time()

            dictionary = {
                "start": start,
                "stop": stop,
                "run_duration": stop - start,
                "Scorer": scorer,
                "config file": cfg,
                "fps": fps,
                "frame_dimensions": (ny, nx),
                "nframes": nframes
            }
            metadata = {'data': dictionary}

            print("Saving results...")
            DataMachine = pd.DataFrame(PredicteData[:nframes, :], columns=pdindex,
                                       index=range(nframes))  # slice pose data to have same # as # of frames.
            DataMachine.to_hdf(dataname, 'df_with_missing', format='table', mode='w')

            np.save(dataname.split('.')[0] + 'scmap', PredictedScmap)

            if storedata_as_csv:
                DataMachine.to_csv(video.split('.')[0] + scorer + '.csv')

            with open(dataname.split('.')[0] + 'includingmetadata.pickle',
                      'wb') as f:
                pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)


