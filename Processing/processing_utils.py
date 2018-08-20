def extract_dlc_data(session):
    """
    Extract the posture data from the DLC tracking into a more meaningful way.
    The behaviour of this function is defined in processing_cfg.yml

    :param session: database entry for the session being processed
    :return:
    """
    for tr_name, tr_data in session['Tracking'].items():
        dlc_tracking = tr_data.dlc_tracking

        try:
            keys  = dlc_tracking['Posture'].keys()[0][0]
        except:
            # This trial doesnt have dlc tracking data
            continue

        # Get bodyparts data
        dlc_tracking = dlc_tracking['Posture'][keys]
        bparts = dlc_tracking.columns.levels[0]

