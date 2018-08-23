from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import pyqtgraph as pg
from moviepy.video.io.VideoFileClip import VideoFileClip
import time
import numpy as np
import traceback, sys

from Debug.DebugTrack_funcs import *


class DebugUI(QWidget):
    def __init__(self, session):
        self.session = session
        super().__init__()

        self.create_widgets()
        self.define_layout()

    def create_widgets(self):
        # Take care of the layout
        self.grid = QGridLayout()
        self.grid.setSpacing(25)

        self.sessname = QLabel('Sess. Name')

        self.vfile_list_tag = QLabel('Video Files')

        self.framenumber = QLabel('Frame: ')

        self.enterframe =  QLineEdit('Enter frame')

        self.goframe = QPushButton(text='Go to Frame')

        self.vidfiles_list = QListWidget()
        self.vidfiles_list.itemDoubleClicked.connect(self.load_selected_video)

        self.loaded_video = QPushButton('Loaded Video file: ')

        self.loaded_tdms = QPushButton('Loaded TDMS file: ')

        self.start_video_btn = QPushButton(text='Start video')
        self.start_video_btn.clicked.connect(self.start_video)

        self.graphicsView = pg.PlotWidget()

    def define_layout(self):
        # Initialise Widgets
        self.grid.addWidget(self.sessname, 1, 1, 1, 3)

        self.grid.addWidget(self.vfile_list_tag, 2, 0)

        self.grid.addWidget(self.framenumber,1, 3, 1, 1)

        self.grid.addWidget(self.enterframe,1, 4, 1, 1)

        self.grid.addWidget(self.goframe,1, 5, 1, 1)

        self.grid.addWidget(self.vidfiles_list, 2, 1, 2, 1)

        self.grid.addWidget(self.loaded_video, 4, 1, 2, 1)

        self.grid.addWidget(self.loaded_tdms, 5, 1, 2, 1)

        self.grid.addWidget(self.start_video_btn, 7, 1, 2, 1)

        self.grid.addWidget(self.graphicsView, 2, 3, 5, 5)

        self.setLayout(self.grid)
        self.setGeometry(100, 100, 2500, 1800)
        self.setWindowTitle('Review')
        self.show()

        # Start threading
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        # Load files
        self.handle_files()

    def handle_files(self):
        # get sessions video files
        self.videos = get_session_videofiles(self.session)

        for filename in self.videos.keys():
            self.vidfiles_list.addItem(filename)

    def load_selected_video(self, selected):
        self.clip = VideoFileClip(self.videos[selected.text()])
        self.loaded_video.setText('Videoclip loaded: {}'.format(selected.text()))
        return selected.text()

    def start_video(self, playbackspeed=1.5):
        if self.loaded_video.text().split(' ')[0] == 'Videoclip':
            print('Playing {}'.format(self.loaded_video.text()))
            # Get the tracking data for the selected session
            clip_name = self.loaded_video.text().split(' ')[-1].split('.')[0]
            tracking_dates = list(self.session['Tracking'].keys())
            if not clip_name in tracking_dates:
                print('Couldnt find tracking data')
            else:
                self.tracking_data = self.session['Tracking'][clip_name]


            # Prep video playback
            self.fps = int(self.clip.fps * 1.5)
            self.sperframe = 1 / self.fps
            self.t = 1

            # Start timer and playback
            self.timer = QTimer()
            self.timer.setInterval(10)
            self.timer.timeout.connect(self.video_player)
            self.timer.start()

    def video_player(self):
        # clean up graphicsView
        self.graphicsView.clear()

        # Plot tracking data
        self.framen = int(self.t*self.fps)
        print(self.framen)

        try:
            std_body_centre = (int(self.tracking_data.std_tracking['x'][self.framen]-50),
                               int(self.tracking_data.std_tracking['y'][self.framen]))
        except:
            std_body_centre = (1, 1)

        # ADD TRACKING DATA FROM DLC

        # Plot a new frame
        frame = np.rot90(self.clip.get_frame(self.t), 1)
        img = pg.ImageItem(frame)
        self.graphicsView.addItem(img)

        # Plot tracking data
        # self.graphicsView.plot([std_body_centre[0]], [std_body_centre[1]], pen=None, symbol='o')

        pen = pg.mkPen('r', width=3, style=Qt.DashLine)
        self.graphicsView.plot(self.tracking_data.std_tracking['x'].values, self.tracking_data.std_tracking['y'].values,
                               pen=pen, symbol='o')
        self.graphicsView.plot([std_body_centre[0]], [std_body_centre[1]], pen=None, symbol='o')

        self.t += self.sperframe


    def plot_tracking(self):
        a = 1


def start_gui(session):
    app = QApplication(sys.argv)
    ex = DebugUI(session)
    sys.exit(app.exec_())

if __name__ == '__main__':
    session = None
    app = QApplication(sys.argv)
    ex = DebugUI(session)
    sys.exit(app.exec_())


