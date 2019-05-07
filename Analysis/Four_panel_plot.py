import cv2
import imageio
import numpy as np
import os
import glob
from Config import dlc_options
dlc_config_settings = dlc_options()


'''
Mine the extant plots to create a new Super Plot
'''

def four_panel_plot(session):

    # Find the old folder, named after the experiment and the mouse
    original_save_folder = os.path.join(dlc_config_settings['clips_folder'], session['Metadata'].experiment,
                               str(session['Metadata'].mouse_id))

    # Save to a new folder named after the experiment and the mouse with the word 'summary'
    new_save_folder = os.path.join(dlc_config_settings['clips_folder'], session['Metadata'].experiment,
                               str(session['Metadata'].mouse_id) + '_summary')
    if not os.path.isdir(new_save_folder):
        os.makedirs(new_save_folder)

    # List the dlc_history, spont_homings, state_action, and expl_recent contents of the old folder
    dlc_history_files = glob.glob(original_save_folder + '\\*history.tif')
    spont_homings_files = glob.glob(original_save_folder + '\\*spont_homings.tif')
    state_action_files = glob.glob(original_save_folder + '\\*procedural_learning.tif')
    exploration_files = glob.glob(original_save_folder + '\\*exploration_recent.tif')

    # make the summary plot for each trial
    for trial in range(len(dlc_history_files)):

        # parameters
        border_size = 40

        videoname = os.path.basename(dlc_history_files[trial]).split("')_")[0] + "')"

        # open the images
        dlc_history_image = cv2.imread(dlc_history_files[trial])
        spont_homings_image = cv2.imread(spont_homings_files[trial])
        state_action_image = cv2.imread(state_action_files[trial])
        exploration_image = cv2.imread(exploration_files[trial])

        # create a new, super image
        images_shape = dlc_history_image.shape
        super_image = np.zeros((int(images_shape[0] * 1.5), int(images_shape[1] * 1.5), 3)).astype(np.uint8)

        # add the dlc image
        super_image[int(1.5 * border_size):int(.5 * border_size) + images_shape[0], 0:images_shape[1] - border_size ] = \
            dlc_history_image[border_size:, :-border_size]

        # add the auxiliary images
        super_image[ int(1.5 * border_size + (images_shape[0]-border_size)/4) : int(.5 * border_size + images_shape[0] - (images_shape[0]-border_size)/4),
         int(images_shape[1] - border_size) : -int(1.5 * border_size) ] = \
            cv2.resize( spont_homings_image[border_size:, :-border_size], (int( (images_shape[0]-border_size)/2), int((images_shape[1]-border_size)/2)), cv2.INTER_CUBIC)

        super_image[ int(.5 * border_size + images_shape[0]) : int(1.5 * images_shape[0]),
         int((images_shape[1]-border_size)/4) : int( (images_shape[1]-border_size) *3/4) ] = \
            cv2.resize( state_action_image[border_size:, :-border_size], (int((images_shape[0]-border_size)/2), int((images_shape[1]-border_size)/2)), cv2.INTER_CUBIC)

        super_image[ int(.5 * border_size + images_shape[0] - (images_shape[0]-border_size)/6) : int(1.5 * images_shape[0] - (images_shape[0]-border_size)/6),
         int(images_shape[1] - border_size - (images_shape[0]-border_size)/6) : -int(1.5 * border_size + (images_shape[0]-border_size)/6) ] = \
            cv2.resize( exploration_image[border_size:, :-border_size], (int((images_shape[0]-border_size)/2), int((images_shape[1]-border_size)/2)), cv2.INTER_CUBIC)

        # add the title and border
        super_image[:int(1.5 * border_size), :-int(1.5 * border_size)] = \
            cv2.resize(dlc_history_image[:border_size, :-border_size], (super_image.shape[1] - int(1.5 * border_size), int(1.5 * border_size)), cv2.INTER_CUBIC)
        super_image[:, -int(1.5 * border_size):] = \
            cv2.resize(dlc_history_image[:, -border_size:], (int(1.5 * border_size), super_image.shape[0]), cv2.INTER_CUBIC)

        # make it a bit smaller
        cv2.imshow('super image', super_image)
        cv2.waitKey(100)

        # recolor and save image
        super_image = cv2.cvtColor(super_image, cv2.COLOR_BGR2RGB)
        imageio.imwrite(os.path.join(new_save_folder, videoname + '.tif'), super_image)

    print('Experiment summary plots saved.')



def four_panel_plot_simulate(session):

    # Find the old folder, named after the experiment and the mouse
    original_save_folder = os.path.join(dlc_config_settings['clips_folder'], session['Metadata'].experiment,
                               str(session['Metadata'].mouse_id))

    # Find the old folder, named after the experiment and the mouse with simulate
    simulate_save_folder = os.path.join(dlc_config_settings['clips_folder'], session['Metadata'].experiment,
                                        str(session['Metadata'].mouse_id) + '_simulate')

    # Save to a new folder named after the experiment and the mouse with the word 'summary'
    new_save_folder = os.path.join(dlc_config_settings['clips_folder'], session['Metadata'].experiment,
                               str(session['Metadata'].mouse_id) + '_simulate_summary')
    if not os.path.isdir(new_save_folder):
        os.makedirs(new_save_folder)

    # List the dlc_history, homing_vector, path_repetition, and expl_recent contents of the old folder
    dlc_history_files = glob.glob(original_save_folder + '\\*history.tif')
    homing_vector_files = glob.glob(simulate_save_folder + '\\*experience_model.tif')
    path_repetition_files = glob.glob(simulate_save_folder + '\\*path_repetition.tif')
    target_repetition_files = glob.glob(simulate_save_folder + '\\*target_repetition.tif')

    # make the summary plot for each trial
    for trial in range(len(dlc_history_files)):

        # parameters
        border_size = 40

        videoname = os.path.basename(dlc_history_files[trial]).split("')_")[0] + "')"

        # open the images
        dlc_history_image = cv2.imread(dlc_history_files[trial])
        homing_vector_image = cv2.imread(homing_vector_files[trial])
        path_repetition_image = cv2.imread(path_repetition_files[trial])
        target_repetition_image = cv2.imread(target_repetition_files[trial])

        # create a new, super image
        images_shape = dlc_history_image.shape
        super_image = np.zeros((int(images_shape[0] * 1.5), int(images_shape[1] * 1.5), 3)).astype(np.uint8)

        # add the dlc image
        super_image[int(1.5 * border_size):int(.5 * border_size) + images_shape[0], 0:images_shape[1] - border_size ] = \
            dlc_history_image[border_size:, :-border_size]

        # add the auxiliary images
        super_image[ int(1.5 * border_size + (images_shape[0]-border_size)/4) : int(.5 * border_size + images_shape[0] - (images_shape[0]-border_size)/4),
         int(images_shape[1] - border_size) : -int(1.5 * border_size) ] = \
            cv2.resize( homing_vector_image[border_size:, :-border_size], (int( (images_shape[0]-border_size)/2), int((images_shape[1]-border_size)/2)), cv2.INTER_CUBIC)

        super_image[ int(.5 * border_size + images_shape[0]) : int(1.5 * images_shape[0]),
         int((images_shape[1]-border_size)/4) : int( (images_shape[1]-border_size) *3/4) ] = \
            cv2.resize( path_repetition_image[border_size:, :-border_size], (int((images_shape[0]-border_size)/2), int((images_shape[1]-border_size)/2)), cv2.INTER_CUBIC)

        super_image[ int(.5 * border_size + images_shape[0] - (images_shape[0]-border_size)/6) : int(1.5 * images_shape[0] - (images_shape[0]-border_size)/6),
         int(images_shape[1] - border_size - (images_shape[0]-border_size)/6) : -int(1.5 * border_size + (images_shape[0]-border_size)/6) ] = \
            cv2.resize( target_repetition_image[border_size:, :-border_size], (int((images_shape[0]-border_size)/2), int((images_shape[1]-border_size)/2)), cv2.INTER_CUBIC)

        # add the title and border
        super_image[:int(1.5 * border_size), :-int(1.5 * border_size)] = \
            cv2.resize(dlc_history_image[:border_size, :-border_size], (super_image.shape[1] - int(1.5 * border_size), int(1.5 * border_size)), cv2.INTER_CUBIC)
        super_image[:, -int(1.5 * border_size):] = \
            cv2.resize(dlc_history_image[:, -border_size:], (int(1.5 * border_size), super_image.shape[0]), cv2.INTER_CUBIC)

        # add text labels
        cv2.putText(super_image, 'Experience Model', (int(images_shape[1] * 17/16), int(border_size + images_shape[0]*17/64)), 0, .75, (255, 255, 255), thickness=2)
        cv2.putText(super_image, 'Target Repetition', (int(images_shape[1] * 29/32), int(border_size*2 + images_shape[0] * 49/64)), 0, .75, (255, 255, 255), thickness=2)
        cv2.putText(super_image, 'Path Repetition', (int(images_shape[1] * 23/64), int(border_size * 2 + images_shape[0] * 119/128)), 0, .75, (255, 255, 255), thickness=2)

        # show the image
        cv2.imshow('super image', super_image)
        cv2.waitKey(100)

        # recolor and save image
        super_image = cv2.cvtColor(super_image, cv2.COLOR_BGR2RGB)
        imageio.imwrite(os.path.join(new_save_folder, videoname + '_experience.tif'), super_image)

    print('Simulation summary plots saved.')




def three_panel_plot_simulate(session):

    # Find the old folder, named after the experiment and the mouse
    original_save_folder = os.path.join(dlc_config_settings['clips_folder'], session['Metadata'].experiment,
                               str(session['Metadata'].mouse_id))

    # Find the old folder, named after the experiment and the mouse with simulate
    simulate_save_folder = os.path.join(dlc_config_settings['clips_folder'], session['Metadata'].experiment,
                                        str(session['Metadata'].mouse_id) + '_simulate')

    # Save to a new folder named after the experiment and the mouse with the word 'summary'
    new_save_folder = os.path.join(dlc_config_settings['clips_folder'], session['Metadata'].experiment,
                               str(session['Metadata'].mouse_id) + '_simulate_summary')
    if not os.path.isdir(new_save_folder):
        os.makedirs(new_save_folder)

    # List the dlc_history, homing_vector, geodesic_model, and expl_recent contents of the old folder
    dlc_history_files = glob.glob(original_save_folder + '\\*history.tif')
    homing_vector_files = glob.glob(simulate_save_folder + '\\*homing_vector.tif')
    geodesic_model_files = glob.glob(simulate_save_folder + '\\*geodesic_model.tif')


    # make the summary plot for each trial
    for trial in range(len(dlc_history_files)):

        # parameters
        border_size = 40

        videoname = os.path.basename(dlc_history_files[trial]).split("')_")[0] + "')"

        # open the images
        dlc_history_image = cv2.imread(dlc_history_files[trial])
        homing_vector_image = cv2.imread(homing_vector_files[trial])
        geodesic_model_image = cv2.imread(geodesic_model_files[trial])

        # create a new, super image
        images_shape = dlc_history_image.shape
        super_image = np.zeros((int(images_shape[0] * 1.5), int(images_shape[1] * 1.5), 3)).astype(np.uint8)

        # add the dlc image
        super_image[int(1.5 * border_size):int(.5 * border_size) + images_shape[0], 0:images_shape[1] - border_size ] = \
            dlc_history_image[border_size:, :-border_size]

        # add the auxiliary images
        super_image[ int(1.5 * border_size + (images_shape[0]-border_size)*2/4) : int(.5 * border_size + images_shape[0] + (images_shape[0]-border_size)*0/4),
         int(images_shape[1] - border_size) : -int(1.5 * border_size) ] = \
            cv2.resize( homing_vector_image[border_size:, :-border_size], (int( (images_shape[0]-border_size)/2), int((images_shape[1]-border_size)/2)), cv2.INTER_CUBIC)

        super_image[ int(.5 * border_size + images_shape[0]) : int(1.5 * images_shape[0]),
         int((images_shape[1]-border_size)*5/8) : int( (images_shape[1]-border_size) *9/8) ] = \
            cv2.resize( geodesic_model_image[border_size:, :-border_size], (int((images_shape[0]-border_size)/2), int((images_shape[1]-border_size)/2)), cv2.INTER_CUBIC)

        # add the title and border
        super_image[:int(1.5 * border_size), :-int(1.5 * border_size)] = \
            cv2.resize(dlc_history_image[:border_size, :-border_size], (super_image.shape[1] - int(1.5 * border_size), int(1.5 * border_size)), cv2.INTER_CUBIC)
        super_image[:, -int(1.5 * border_size):] = \
            cv2.resize(dlc_history_image[:, -border_size:], (int(1.5 * border_size), super_image.shape[0]), cv2.INTER_CUBIC)

        # add text labels
        cv2.putText(super_image, 'Homing Vector', (int(images_shape[1] * 17/16), int(border_size + images_shape[0]*32/64)), 0, .75, (255, 255, 255), thickness=2)
        cv2.putText(super_image, 'Optimal Path', (int(images_shape[1] * 23/32), int(border_size * 2 + images_shape[0] * 59/64)), 0, .75, (255, 255, 255), thickness=2)

        # show the image
        cv2.imshow('super image', super_image)
        cv2.waitKey(100)

        # recolor and save image
        super_image = cv2.cvtColor(super_image, cv2.COLOR_BGR2RGB)
        imageio.imwrite(os.path.join(new_save_folder, videoname + '.tif'), super_image)

    print('Simulation summary plots saved.')