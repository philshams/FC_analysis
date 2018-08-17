import cv2


def process_background(background, track_options):
    print('     ... extracting background')

    cv2.startWindowThread()
    # cv2.namedWindow('Raw background')
    # cv2.namedWindow('edges')

    if len(background.shape) == 3:
        gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    else:
        gray = background

    # Blur
    blur = cv2.blur(gray, (15, 15))

    # Get maze edges
    edges = cv2.Canny(blur, 25, 30)

    if track_options['bg get rois']:
        # Get user to define Shelter ROI, Threat area ROI and the line between them or arena ROI
        print('\n\nPlease mark NEST ROI')
        shelter_roi = cv2.selectROI(gray, fromCenter=False)
        print('\n\nPlease mark THREAT ROI')
        threat_roi = cv2.selectROI(gray, fromCenter=False)

        print('\n\nPlease mark ARENA FLOOR ROI or SMALL BRIDGE ROI')
        var_roi = cv2.selectROI(gray, fromCenter=False)

        rois = {'Shelter':shelter_roi, 'Threat':threat_roi, 'Task':var_roi}
    else:
        rois = {'Shelter':None, 'Threat':None, 'Task':None}
    return edges, rois


