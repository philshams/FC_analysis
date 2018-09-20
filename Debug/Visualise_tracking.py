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

        self.colors = dict(head=(.3, .8, .3), body=(.8, .3, .3), tail=(.3, .3, .8))
        self.bodyparts = dict(head=['snout', 'Lear', 'Rear', 'neck'], body=['body'], tail=['tail'])


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
        self.videos = self.session.Metadata.video_file_paths

        self.trials = [t for t in self.session.Tracking.keys() if '-' in t]

        [self.trials_listw.addItem(tr) for tr in self.trials]

    def define_layout(self):
        """ defines the layout of the previously created widgets  """
        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtGui.QVBoxLayout())

        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas)

        self.label = QtGui.QLabel()
        self.mainbox.layout().addWidget(self.label)

        self.view = self.canvas.addViewBox()
        self.view.setAspectLocked(True)
        self.view.setRange(QtCore.QRectF(0, 0, 600, 600))

        #  image plot
        self.img = pg.ImageItem(border='w')
        self.view.addItem(self.img)

        #  Pose Reconstruction
        self.poseplot = self.canvas.addPlot()
        self.pose_body = self.poseplot.plot(pen='g')

        self.canvas.nextRow()

        #  Velocity
        self.velplot = self.canvas.addPlot()

        # buttons
        self.canvas.nextRow()
        self.launch_btn = QPushButton(text='Launch')
        self.launch_btn.clicked.connect(lambda: self.update_by_frame(self))
        self.mainbox.layout().addWidget(self.launch_btn)

        # List of trials widgets
        self.trials_listw = QListWidget()
        self.mainbox.layout().addWidget(self.trials_listw)
        self.trials_listw.itemDoubleClicked.connect(self.load_trial_data)

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

        self.plot_pose(0)

        self.data_loaded = False

    def plot_pose(self, framen):
        for bp, data in self.tracking_data.items():
            for key,parts in self.bodyparts.items():
                if bp in parts:
                    col = self.colors[key]
                    self.poseplot.plot([data.loc[framen].x],
                                       [data.loc[framen].y],
                                       pen=col, symbolBrush=col, symbolPen='w', symbol='o', symbolSize=30)
                    break

    def update_by_frame(self):
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
            self.label.setText(tx)


            _, self.frame = self.video_grabber.read()
            self.frame = self.frame[:, :, 0]
            self.img.setImage(np.rot90(self.frame, 3))
            self.plot_pose(f)
            f += 1

    def update(self):
        self.data = np.sin(self.X/3.+self.counter/9.)*np.cos(self.Y/3.+self.counter/9.)
        self.ydata = np.sin(self.x/3.+ self.counter/9.)
        self.img.setImage(self.data)
        self.h2.setData(self.ydata)
        now = time.time()
        dt = (now-self.lastupdate)
        if dt <= 0:
            dt = 0.000000000001
        fps2 = 1.0 / dt
        self.lastupdate = now
        self.fps = self.fps * 0.9 + fps2 * 0.1
        tx = 'Mean Frame Rate:  {fps:.3f} FPS'.format(fps=self.fps )
        self.label.setText(tx)
        QtCore.QTimer.singleShot(1, self._update)
        self.counter += 1


if __name__ == '__main__':
    thisapp = App(None)
    thisapp.show()


















