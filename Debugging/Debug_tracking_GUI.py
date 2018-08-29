import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget




class DebugTracking():
    """
    The aim of this class is to create a GUI that allows the user to load a video file and inspect the tracking data
    that correspond to it.

    It needs to:
        * Present a list of all the videos for the session being debugged
        * Allow user to load one video
        * Plot all the frames of the video at high speed
        * Give option to check either DLC or STD tracking by overimposing stuff over the video
        * Things that cannot be overimposed (e.g. orientation relative to shelter..) will have to be displayed a part


    """

    app = QApplication(sys.argv)

    w = QWidget()
    w.resize(250, 150)
    w.move(300, 300)
    w.setWindowTitle('Simple')
    w.show()

    sys.exit(app.exec_())




    