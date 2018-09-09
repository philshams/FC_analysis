import cv2


def process_background(background, track_options):
    """ extract background: first frame of first video of a session
    Allow user to specify ROIs on the background image """
    print('     ... extracting background')

    cv2.startWindowThread()
    if len(background.shape) == 3:
        gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    else:
        gray = background

    blur = cv2.blur(gray, (15, 15))
    edges = cv2.Canny(blur, 25, 30)

    rois = {'Shelter': None, 'Threat': None, 'Task': None}
    if track_options['bg get rois']:          # Get user to define Shelter ROI
        for rname in rois.keys():
            print('\n\nPlease mark {}'.format(rname))
            rois[rname] = cv2.selectROI(gray, fromCenter=False)

    return edges, rois


