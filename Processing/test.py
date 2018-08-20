def testfunc(session):
    dlc_tracking = session['Tracking']['62-visual_0-0'].dlc_tracking
    keys = dlc_tracking['Posture'].keys()[0][0]
    dlc_tracking = dlc_tracking['Posture'][keys]
    a = 1
