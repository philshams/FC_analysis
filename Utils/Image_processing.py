import cv2
from termcolor import colored

def process_background(background, track_options):
    """ extract background: first frame of first video of a session
    Allow user to specify ROIs on the background image """
    print(colored('Extracting ROIs:','green'))

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