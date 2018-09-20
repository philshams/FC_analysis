from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os
import pyqtgraph as pg
import sys
import time
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import cv2


class App(QtGui.QMainWindow):
    def __init__(self, session, parent=None):
        super(App, self).__init__(parent)

        self.colors = dict(head=(100, 220, 100), body=(220, 100, 100), tail=(100, 100, 220))
        self.bodyparts = dict(head=['snout', 'Lear', 'Rear', 'neck'], body=['body'], tail=['tail'])
        self.bodyparts_plotdata = {}

        """  Creates the user interface of the  debugger app """
        self.session = session

        self.define_layout()

        self.get_session_data()

        self.counter = 0
        self.fps = 0.
        self.lastupdate = time.time()
        self.data_loaded = False

        app = QtGui.QApplication(sys.argv)
        self.show()
        app.exec_()

    def get_session_data(self):
        if self.session is None:
            return

        self.videos = self.session.Metadata.video_file_paths

        self.trials = [t for t in self.session.Tracking.keys() if '-' in t]

        [self.trials_listw.addItem(tr) for tr in self.trials]

    def define_layout(self):
        def define_style_sheet():
            # Main window color
            self.setAutoFillBackground(True)
            p = self.palette()
            p.setColor(self.backgroundRole(), QColor(40, 40, 40, 255))
            self.setPalette(p)

            # Widgets style sheet
            self.setStyleSheet("""
             QPushButton {
                            color: #ffffff;
                            background-color: #565656;
                            border: 2px solid #8f8f91;
                            border-radius: 6px;
                            min-width: 120px;
                            min-height: 60px;
                        }
                        
             QLabel {
                            color: #ffffff;
    
                            font-size: 16pt;
    
                            min-width: 80px;
                            min-height: 40px;
                        }          
                       
                       
             QLineEdit {
                            color: #202223;
                            font-size: 10pt;
                            background-color: #c1c1c1;
                            border-radius: 4px;

                            min-width: 80px;
                            min-height: 40px;
                        }
                        
             QListWidget {
                            font-size: 14pt;
                            background-color: #c1c1c1;
                            border-radius: 4px;

                        }
                        
            """)

        define_style_sheet()

        """ defines the layout of the previously created widgets  """
        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtGui.QGridLayout())

        self.progres_bar_canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.progres_bar_canvas, 16, 0, 2, 15)


        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas, 0, 0, 12, 15)

        self.framerate_label = QtGui.QLabel()
        self.framerate_label.setText('Framerate')
        self.mainbox.layout().addWidget(self.framerate_label, 1, 16, 1, 4)

        self.playspeed_label = QtGui.QLabel()
        self.playspeed_label.setText('Play speed')
        self.mainbox.layout().addWidget(self.playspeed_label, 2, 16, 1, 2)

        self.playspeed_entry = QLineEdit('1x')
        self.mainbox.layout().addWidget(self.playspeed_entry, 2, 18, 1, 2)

        self.current_frame =  QtGui.QLabel()
        self.current_frame.setText('Frame 0')
        self.mainbox.layout().addWidget(self.current_frame, 3, 16, 1, 4)

        self.sessname = QtGui.QLabel()
        if not self.session is None:
            self.sessname.setText('Session {}'.format(self.session.Metadata['name']))
        self.mainbox.layout().addWidget(self.sessname, 0, 16, 1, 2)

        self.trname = QtGui.QLabel()
        self.trname.setText('Trial Name')
        self.mainbox.layout().addWidget(self.trname, 0, 18, 1, 2)

        self.view = self.canvas.addViewBox()
        self.view.setAspectLocked(False)

        #  image plot
        self.img = pg.ImageItem(border='w')
        self.view.addItem(self.img)

        #  Pose Reconstruction
        self.poseplot = self.canvas.addPlot()
        self.poseplot.invertY(True)
        self.pose_body = self.poseplot.plot(pen='g')
        self.canvas.nextRow()

        #  Velocity
        self.velplot = self.canvas.addPlot()
        self.vel_line = self.velplot.plot(pen=pg.mkPen('r', width=5))

        # Ang vels
        self.angvelplot = self.canvas.addPlot()
        self.ang_vel_line = self.angvelplot.plot(pen=pg.mkPen('r', width=5))

        # Progress
        self.progress_plot = self.progres_bar_canvas.addPlot(colspan=2)
        self.progress_line = self.velplot.plot(pen=pg.mkPen('r', width=5))

        # buttons
        self.launch_btn = QPushButton(text='Launch')
        self.launch_btn.clicked.connect(lambda: self.update_by_frame(self))
        self.mainbox.layout().addWidget(self.launch_btn, 6, 16, 2, 2)

        self.stop_btn = QPushButton(text='Stop')
        self.stop_btn.clicked.connect(lambda: self.update_by_frame(self))
        self.mainbox.layout().addWidget(self.stop_btn, 6, 20, 2, 2)

        self.pause_btn = QPushButton(text='Pause')
        self.pause_btn.clicked.connect(lambda: self.pause_playback(self))
        self.mainbox.layout().addWidget(self.pause_btn, 4, 16, 4, 2)

        self.resume_btn = QPushButton(text='Resume')
        self.resume_btn.clicked.connect(lambda: self.update_by_frame(self))
        self.mainbox.layout().addWidget(self.resume_btn, 4, 20, 4, 2)

        # List of trials widgets
        self.trlistlabel = QtGui.QLabel()
        self.trlistlabel.setText('Trials')
        self.mainbox.layout().addWidget(self.trlistlabel, 8, 16)


        self.trials_listw = QListWidget()
        self.mainbox.layout().addWidget(self.trials_listw, 9, 16, 3, 6)
        self.trials_listw.itemDoubleClicked.connect(self.load_trial_data)

        # Define window geometry
        self.setGeometry(50, 50, 3000, 2000)

    def load_trial_data(self, trial_name):
        # Get data
        trial_name = trial_name.text()
        print(trial_name)
        videonum = int(trial_name.split('_')[1].split('-')[0])
        self.video = self.videos[videonum][0]
        self.video_grabber = cv2.VideoCapture(self.video)
        self.start_frame = self.session.Tracking[trial_name].metadata['Start frame']
        self.tracking_data = self.session.Tracking[trial_name].dlc_tracking['Posture']

        # Plot the first frame
        self.video_grabber.set(1, self.start_frame)
        _, self.frame = self.video_grabber.read()
        self.frame = self.frame[:, :, 0]
        self.img.setImage(np.rot90(self.frame, 3))

        self.data_loaded = True

    def plot_pose(self, framen):
        if framen == 0:
            self.bodyparts_plotdata = {}

        for bp, data in self.tracking_data.items():
            for key,parts in self.bodyparts.items():
                if bp in parts:
                    if bp == 'body':
                        centre = data.loc[framen].x, data.loc[framen].y
                    col = self.colors[key]
                    if framen == 0:
                        self.bodyparts_plotdata[bp] = self.poseplot.plot([data.loc[framen].x],
                                           [data.loc[framen].y],
                                           pen=col, symbolBrush=col, symbolPen='w', symbol='o', symbolSize=30)
                    else:
                        self.bodyparts_plotdata[bp].setData([data.loc[framen].x], [data.loc[framen].y])
                    break

        self.poseplot.setRange(xRange=[centre[0]-50, centre[0]+50], yRange=[centre[1]+50, centre[1]-50])

    def plot_tracking_data(self, framen):
        vel = self.tracking_data['body']['Velocity'].values
        self.vel_line.setData(np.linspace(0, 100, 100), vel[framen:framen+100])
        self.velplot.setRange(yRange=[0, max(vel)+(max(vel)/10)])

    def update_by_frame(self, event):
        if not self.data_loaded:
            return

        f = 0
        self.video_grabber.set(1, self.start_frame)

        while True:
            now = time.time()
            dt = (now - self.lastupdate)
            if dt <= 0:
                dt = 0.000000000001
            fps2 = 1.0 / dt
            self.lastupdate = now
            self.fps = self.fps * 0.9 + fps2 * 0.1
            tx = 'Mean Frame Rate:  {fps:.3f} FPS'.format(fps=self.fps)
            self.framerate_label.setText(tx)

            ret, self.frame = self.video_grabber.read()
            if not ret:
                break
            self.frame = self.frame[:, :, 0]
            self.img.setImage(np.rot90(self.frame, 3))
            self.plot_pose(f)
            self.plot_tracking_data(f)

            f += 1
            pg.QtGui.QApplication.processEvents()

    def pause_playback(self):
        self.data_loaded = False


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    thisapp = App(None)
    thisapp.show()
    app.exec_()


















