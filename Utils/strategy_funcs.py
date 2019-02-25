
# #####################################
# compute and display GOALNESS DURING EXPLORATION
# go through each frame, adding the mouse silhouette
# ########################################

scale = int(frame.shape[0 ] /10)
goal_arena, _, _ = model_arena(frame.shape[0:2], trial_type, False, False, obstacle_type)
speed_map = np.zeros((scale, scale))
occ_map = np.zeros((scale, scale))

goal_map = np.zeros((scale, scale))

# stim_erase_idx = np.arange(len(coordinates['center_location'][0][:stim_frame]))
# stim_erase_idx = [np.min(abs(x - stims)) for x in stim_erase_idx]
# stim_erase_idx = [x > 300 for x in stim_erase_idx]

# filter_sequence = np.concatenate( (np.ones(15)*-np.percentile(coordinates['speed'],99.5), np.zeros(10)) )
filter_sequence = np.ones(20) * -np.percentile(coordinates['speed'], 99.5)
print(colored(' Calculating goalness...', 'green'))
for x_loc in tqdm(range(occ_map.shape[0])):
    for y_loc in range(occ_map.shape[1]):
        curr_dist = np.sqrt((coordinates['center_location'][0][:stim_frame] - ((720 / scale) * (x_loc + 1 / 2))) ** 2 +
                            (coordinates['center_location'][1][:stim_frame] - ((720 / scale) * (y_loc + 1 / 2))) ** 2)
        occ_map[x_loc, y_loc] = np.mean(curr_dist < (2 * 720 / scale))
        curr_speed = np.concatenate(([0], np.diff(curr_dist))) # * np.array(stim_erase_idx)  # * (coordinates['center_location'][1] < 360) #
        speed_map[x_loc, y_loc] = abs(np.mean(curr_speed < -np.percentile(coordinates['speed'], 99.5)))

        goal_map[x_loc, y_loc] = np.percentile(np.concatenate((np.zeros(len(filter_sequence) - 1),
                                                               np.convolve(curr_speed, filter_sequence, mode='valid'))) * (curr_dist < 60), 99.8)  # 98

goal_map_plot = goal_map.T * (occ_map.T > 0)

goal_image = goal_map_plot.copy()

goal_image = goal_image * 255 / np.percentile(goal_map_plot, 99)
goal_threshold = int(np.percentile(goal_map_plot, 90) * 255 / np.percentile(goal_map_plot, 99))

goal_image[goal_image > 255] = 255
goal_image = cv2.resize(goal_image.astype(np.uint8), frame.shape[0:2])

goal_image[goal_image <= int(goal_threshold * 1 / 5) * (goal_image > 1)] = int(goal_threshold * 1 / 10)
goal_image[(goal_image <= goal_threshold * 2 / 5) * (goal_image > int(goal_threshold * 1 / 5))] = int(goal_threshold * 2 / 10)
goal_image[(goal_image <= goal_threshold * 3 / 5) * (goal_image > int(goal_threshold * 2 / 5))] = int(goal_threshold * 3 / 10)
goal_image[(goal_image <= goal_threshold * 4 / 5) * (goal_image > int(goal_threshold * 3 / 5))] = int(goal_threshold * 4 / 10)
goal_image[(goal_image <= goal_threshold) * (goal_image > int(goal_threshold * 4 / 5))] = int(goal_threshold * 6 / 10)
goal_image[(goal_image < 255) * (goal_image > goal_threshold)] = int(goal_threshold)

# goal_image[(arena_fresh[:,:,0] > 0) * (goal_image == 0)] = int(goal_threshold * 1 / 5)
goal_image[(arena_fresh[:, :, 0] < 100)] = 0

goal_image = cv2.copyMakeBorder(goal_image, border_size, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
textsize = cv2.getTextSize(videoname, 0, .55, 1)[0]
textX = int((width - textsize[0]) / 2)
cv2.putText(goal_image, videoname, (textX, int(border_size * 3 / 4)), 0, .55, (255, 255, 255), thickness=1)
scipy.misc.imsave(os.path.join(savepath, videoname + '_goalness.tif'), goal_image)

cv2.imshow('goal', goal_image)
cv2.waitKey(1)







# #####################################
# compute and display PLANNESS DURING EXPLORATION
# go through each frame, adding the mouse silhouette
# ########################################
scale = int(frame.shape[0] / 10)
goal_arena, _, _ = model_arena(frame.shape[0:2], trial_type, False, False, obstacle_type)
speed_map = np.zeros((scale, scale))
occ_map = np.zeros((scale, scale))

plan_map = np.zeros((scale, scale))

# stim_erase_idx = np.arange(len(coordinates['center_location'][0][:stim_frame]))
# stim_erase_idx = [np.min(abs(x - stims)) for x in stim_erase_idx]
# stim_erase_idx = [x > 300 for x in stim_erase_idx]

# filter_sequence = np.concatenate( (np.ones(15)*-np.percentile(coordinates['speed'],99.5), np.zeros(10)) )
# stim_frame = 30*60*28

filter_sequence = np.ones(20) * -np.percentile(coordinates['speed'], 99.5)
print(colored(' Calculating planness...', 'green'))
speed_toward_shelter = np.convolve(coordinates['speed_toward_shelter'][:stim_frame], filter_sequence, mode='valid')
distance_from_shelter = coordinates['distance_from_shelter'][:stim_frame]
# arrival_in_shelter = coordinates['distance_from_shelter'][:stim_frame] < 100
# future_arrival_in_shelter = np.concatenate( (arrival_in_shelter[:-30], np.zeros(30) ) )
for x_loc in tqdm(range(occ_map.shape[0])):
    for y_loc in range(occ_map.shape[1]):
        curr_dist = np.sqrt((coordinates['center_location'][0][:stim_frame] - ((720 / scale) * (x_loc + 1 / 2))) ** 2 +
                            (coordinates['center_location'][1][:stim_frame] - ((720 / scale) * (y_loc + 1 / 2))) ** 2)
        occ_map[x_loc, y_loc] = np.mean(curr_dist < (2 * 720 / scale))
        # curr_speed = np.concatenate(([0], np.diff(curr_dist)))  # * np.array(stim_erase_idx)  # * (coordinates['center_location'][1] < 360) #
        # speed_map[x_loc, y_loc] = abs(np.mean(curr_speed < -np.percentile(coordinates['speed'], 99.5)))

        plan_map[x_loc, y_loc] = np.percentile(
            np.concatenate((speed_toward_shelter, np.zeros(len(filter_sequence) - 1))) * (curr_dist < 50) * (distance_from_shelter > 175), 99.2)  # 98

plan_map_plot = plan_map.T * (occ_map.T > 0)

plan_image = plan_map_plot.copy()

plan_image = plan_image * 255 / np.percentile(plan_map_plot, 99.9)
try:
    plan_threshold = int(np.percentile(plan_map_plot, 99) * 255 / np.percentile(plan_map_plot, 99.9))
except:
    plan_threshold = 200

plan_image[plan_image > 255] = 255
plan_image = cv2.resize(plan_image.astype(np.uint8), frame.shape[0:2])

plan_image[plan_image <= int(plan_threshold * 1 / 5) * (plan_image > 1)] = int(plan_threshold * 1 / 5)
plan_image[(plan_image <= plan_threshold * 2 / 5) * (plan_image > int(plan_threshold * 1 / 5))] = int(plan_threshold * 3 / 5)
plan_image[(plan_image <= plan_threshold * 3 / 5) * (plan_image > int(plan_threshold * 2 / 5))] = int(plan_threshold * 3 / 5)
plan_image[(plan_image <= plan_threshold * 4 / 5) * (plan_image > int(plan_threshold * 3 / 5))] = int(plan_threshold * 4 / 5)
plan_image[(plan_image <= plan_threshold) * (plan_image > int(plan_threshold * 4 / 5))] = plan_threshold
plan_image[(plan_image < 255) * (plan_image > plan_threshold)] = int((plan_threshold + 255) / 2)

# plan_image[(arena_fresh[:,:,0] > 0) * (plan_image == 0)] = int(plan_threshold * 3 / 10)
plan_image[(arena_fresh[:, :, 0] < 100)] = 0

plan_image = cv2.copyMakeBorder(plan_image, border_size, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
textsize = cv2.getTextSize(videoname, 0, .55, 1)[0]
textX = int((width - textsize[0]) / 2)
cv2.putText(plan_image, videoname, (textX, int(border_size * 3 / 4)), 0, .55, (255, 255, 255), thickness=1)
scipy.misc.imsave(os.path.join(savepath, videoname + '_planness.tif'), plan_image)

cv2.imshow('plan', plan_image)
cv2.waitKey(1)











# #####################################
# compute and display EXPLORATION
# go through each frame, adding the mouse silhouette
# ########################################
# exploration_arena_in = arena_fresh.copy()
# slow_color = np.array([155, 245, 245])  # brown-yellow
# exploration_arena_in_cum = arena_fresh.copy()
high_speed = np.percentile(coordinates['speed_toward_shelter'], 99)
# print(high_speed)
# vid_EXPLORE = cv2.VideoWriter(os.path.join(savepath, videoname + '_dlc_spont_homings.avi'), fourcc,
#                                  fps, (width, height), True)

for frame_num in range(previous_stim_frame + 300, stim_frame + 300):

    # speed_toward_shelter = coordinates['speed_toward_shelter'][frame_num - 1]
    speed_toward_shelter = np.median(coordinates['speed_toward_shelter'][frame_num - 1 - 8:frame_num + 8])
    speed_toward_shelter_future = np.median(coordinates['speed_toward_shelter'][frame_num - 1:frame_num + 30])
    speed_toward_shelter_far_future = np.median(coordinates['speed_toward_shelter'][frame_num + 15:frame_num + 60])
    speed_toward_shelter_past = np.median(coordinates['speed_toward_shelter'][frame_num - 30:frame_num - 1])
    speed = coordinates['speed'][frame_num - 1]

    if (speed_toward_shelter < -5) or (speed_toward_shelter_future < -4.5) or (speed_toward_shelter_far_future < -5 and speed_toward_shelter < 1) or \
            (speed_toward_shelter_past < -4.5) or (frame_num > stim_frame and speed > 3 and speed_toward_shelter < 3) or frame_num == stim_frame:

        multiplier = 1
        if (speed_toward_shelter_far_future < -5):
            # print('far future')
            multiplier = .8
        if (speed_toward_shelter_future < -4.5):
            # print('future')
            multiplier = .9
        if (speed_toward_shelter_past < -4.5):
            # print('past')
            multiplier = 1
        if (speed_toward_shelter < -5):
            # print('speed')
            multiplier = 1

        body_angle = coordinates['body_angle'][frame_num - 1]
        shoulder_angle = coordinates['shoulder_angle'][frame_num - 1]
        head_angle = coordinates['head_angle'][frame_num - 1]

        head_location = tuple(coordinates['front_location'][:, frame_num - 1].astype(np.uint16))
        butt_location = tuple(coordinates['butt_location'][:, frame_num - 1].astype(np.uint16))
        back_location = tuple(coordinates['back_location'][:, frame_num - 1].astype(np.uint16))
        center_body_location = tuple(coordinates['center_body_location'][:, frame_num - 1].astype(np.uint16))
        center_location = tuple(coordinates['center_location'][:, frame_num - 1].astype(np.uint16))

        model_mouse_mask = cv2.ellipse(model_mouse_mask_initial.copy(), head_location, (8, 4), 180 - head_angle, 0, 360, 100, thickness=-1)
        model_mouse_mask = cv2.ellipse(model_mouse_mask, center_location, (12, 6), 180 - shoulder_angle, 0, 360, 100, thickness=-1)
        model_mouse_mask = cv2.ellipse(model_mouse_mask, center_body_location, (13, 7), 180 - body_angle, 0, 360, 100, thickness=-1)

        speed_toward_shelter = abs(speed_toward_shelter)
        speed_toward_shelter_past = abs(speed_toward_shelter_past)
        speed_toward_shelter_future = abs(speed_toward_shelter_future)

        if not speed_toward_shelter:
            speed_color = np.array([245, 245, 245])
        elif speed_toward_shelter < high_speed:  # 5:
            speed_color = np.array([255, 254, 253.9])  # blue
            if frame_num > stim_frame:
                speed_color = np.array([200, 105, 200])  # purple
                multiplier = 1.7
        elif speed_toward_shelter < high_speed * 2:
            speed_color = np.array([220, 220, 200])
            if frame_num > stim_frame:
                speed_color = np.array([185, 175, 240])  # purple
                multiplier = .7
        else:
            speed_color = np.array([152, 222, 152])  # green
            if frame_num > stim_frame:
                speed_color = np.array([150, 150, 250])  # red

        speed_color = 1 - (1 - speed_color / [255, 255, 255]) / (np.mean(1 - speed_color / [255, 255, 255]) /
                                                                 (.0035 / 2 * (speed_toward_shelter + speed_toward_shelter_future + 1) * multiplier))

        if not np.isnan(speed_color[0]):
            exploration_arena_in_cum[model_mouse_mask.astype(bool)] = exploration_arena_in_cum[model_mouse_mask.astype(bool)] * speed_color

        cv2.imshow('explore_in', exploration_arena_in_cum)
        # vid_EXPLORE.write(exploration_arena_in_cum)

        if frame_num == stim_frame:
            _, contours, _ = cv2.findContours(model_mouse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# vid_EXPLORE.release()
try:
    exploration_arena_in_cum_save = cv2.drawContours(exploration_arena_in_cum.copy(), contours, 0, (0, 0, 0), 2)
    exploration_arena_in_cum_save = cv2.drawContours(exploration_arena_in_cum_save, contours, 0, (255, 255, 255), 1)
    scipy.misc.imsave(os.path.join(savepath, videoname + '_spont_homings.tif'), cv2.cvtColor(exploration_arena_in_cum_save, cv2.COLOR_BGR2RGB))
except:
    print('repeat stimulus trial')
# Make a position heat map as well
scale = 1
H, x_bins, y_bins = np.histogram2d(coordinates['back_location'][0, 0:stim_frame], coordinates['back_location'][1, 0:stim_frame],
                                   [np.arange(0, width + 1, scale), np.arange(0, height + 1, scale)], normed=True)
H = H.T

H = cv2.GaussianBlur(H, ksize=(5, 5), sigmaX=1, sigmaY=1)
H[H > np.percentile(H, 98)] = np.percentile(H, 98)

H_image = (H * 255 / np.max(H)).astype(np.uint8)
H_image[(H_image < 25) * (H_image > 0)] = 25
H_image[(arena > 0) * (H_image == 0)] = 9
H_image = cv2.copyMakeBorder(H_image, border_size, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
textsize = cv2.getTextSize(videoname, 0, .55, 1)[0]
textX = int((width - textsize[0]) / 2)
cv2.putText(H_image, videoname, (textX, int(border_size * 3 / 4)), 0, .55, (255, 255, 255), thickness=1)

cv2.imshow('heat map', H_image)
cv2.waitKey(1)
scipy.misc.imsave(os.path.join(savepath, videoname + '_exploration.tif'), H_image)