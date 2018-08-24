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

        self.framen = 0
        self.playing = False

        self.create_widgets()
        self.define_layout()

        # Display silent exceptions that cause the gui to crash
        sys._excepthook = sys.excepthook

        def exception_hook(exctype, value, traceback):
            print(exctype, value, traceback)
            sys._excepthook(exctype, value, traceback)
            sys.exit(1)

        sys.excepthook = exception_hook

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

        self.resume_video_btn = QPushButton(text='Stop video')
        self.resume_video_btn.clicked.connect(self.resume_stop_video)

        self.stop_video_btn = QPushButton(text='Stop video')
        self.stop_video_btn.clicked.connect(self.resume_stop_video)

        self.videoView = pg.PlotWidget()

        self.plotView = pg.PlotWidget()

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

        self.grid.addWidget(self.start_video_btn, 6, 1, 2, 1)

        self.grid.addWidget(self.resume_video_btn, 7, 1, 2, 1)

        self.grid.addWidget(self.stop_video_btn, 8, 1, 2, 1)

        self.grid.addWidget(self.videoView, 2, 3, 5, 5)

        self.grid.addWidget(self.plotView, 7, 3, 2, 5)

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

        for filename in sorted(self.videos.keys()):
            self.vidfiles_list.addItem(filename)

    def load_selected_video(self, selected):
        self.clip = VideoFileClip(self.videos[selected.text()])
        self.loaded_video.setText('Videoclip loaded: {}'.format(selected.text()))
        return selected.text()

    def start_video(self, playbackspeed=1.5):
        if self.loaded_video.text().split(' ')[0] == 'Videoclip':
            self.playing = True

            print('Playing {}'.format(self.loaded_video.text()))
            # Get the tracking data for the selected session
            clip_name = self.loaded_video.text().split(' ')[-1].split('.')[0]
            tracking_dates = list(self.session['Tracking'].keys())
            if not clip_name in tracking_dates:
                print('Couldnt find tracking data')
            else:
                self.tracking_data = self.session['Tracking'][clip_name]

            # Prep video playback
            self.fps = int(self.clip.fps)  # * 1.5)
            self.sperframe = 1 / self.fps
            self.t = 1
            self.start_playback_time = time.clock()
            self.stimframe = int(len(self.tracking_data.std_tracking)/2)
            self.stimduration = 0.75*self.fps

            # Start timer and playback
            self.timer = QTimer()
            self.timer.setInterval(10)
            self.timer.timeout.connect(self.video_player)
            self.timer.start()

    def video_player(self):
        if self.playing:
            # clean up plots and set axes
            self.videoView.clear()
            self.plotView.clear()

            self.plotView.setRange(xRange=[0, 120], yRange=[-1, 1])
            marker = pg.InfiniteLine(60)
            self.plotView.addItem(marker)

            # Plot tracking data
            elapsed = time.clock() - self.start_playback_time
            print(elapsed, self.framen)

            std_body_centre, std_velocity, std_orientation, std_velocity_smooth\
                = get_std_to_plot(self.tracking_data.std_tracking, elapsed, self.fps)

            dlc_body_parts, dlc_velocity, dlc_velocity_smooth =\
                get_dlc_to_plot(self.tracking_data.dlc_tracking, elapsed, self.fps)

            self.framen += 1

            # Plot a new frame
            frame = np.rot90(self.clip.get_frame(elapsed), 1)
            frame = np.flip(frame, 0)
            img = pg.ImageItem(frame)
            self.videoView.addItem(img)

            # Plot tracking data
            rpen = pg.mkPen('r', width=3)
            gpen = pg.mkPen('g', width=3)

            symbs = ['t', 't1', 't2']
            self.videoView.plot([std_body_centre[0]], [std_body_centre[1]], pen=None, symbol='o', symbolPen='r')
            for idx, bp in enumerate(dlc_body_parts):
                self.videoView.plot([bp[0]], [bp[1]], pen=None, symbol=symbs[idx], symbolPen='g')

            self.plotView.plot(std_velocity_smooth, pen=rpen)
            self.plotView.plot(dlc_velocity_smooth, pen=gpen)

            # Plot stimulus when LOOM is on
            if self.stimframe<=self.framen<=self.stimframe+self.stimduration:
                r = pg.QtGui.QGraphicsRectItem(10, 10, 10, 10)
                r.setPen(pg.mkPen((0, 0, 0, 100)))
                r.setBrush(pg.mkBrush((50, 50, 200)))
                self.videoView.addItem(r)

            self.t += self.sperframe

    def resume_stop_video(self):
        if self.playing:
            self.playing = False
        else:
            self.playing = True


def start_gui(session):
    app = QApplication(sys.argv)
    ex = DebugUI(session)
    sys.exit(app.exec_())


if __name__ == '__main__':
    session = None
    app = QApplication(sys.argv)
    ex = DebugUI(session)
    sys.exit(app.exec_())


