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

        self.wait_ms = 0

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

        [self.trials_listw.addItem(tr) for tr in sorted(self.trials)]

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
                            font-size: 18pt;
                            background-color: #565656;
                            border: 2px solid #8f8f91;
                            border-radius: 6px;
                            min-width: 250px;
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
                            font-size: 14pt;
                            background-color: #c1c1c1;
                            border-radius: 4px;
                            min-height: 40px;
                            min-width: 20px;
                        }
                        
             QListWidget {
                            font-size: 14pt;
                            background-color: #c1c1c1;
                            border-radius: 4px;

                        }
                        
             QPushButton#LaunchBtn {
                            background-color: #006600;
                        }   
             QPushButton#PauseBtn {
                background-color: #d7c832;
                        }
             QPushButton#StopBtn {
                background-color: #a32020;
                        }  
             QPushButton#ResumeBtn {
                background-color: #73a120;
                        }  
            """)

        define_style_sheet()

        """ defines the layout of the previously created widgets  """
        self.mainbox = QtGui.QWidget()
        self.mainbox.showFullScreen()
        self.setCentralWidget(self.mainbox)
        grid = QtGui.QGridLayout()
        grid.setSpacing(10)
        self.mainbox.setLayout(grid)

        self.progres_bar_canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.progres_bar_canvas, 16, 0, 2, 15)


        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas, 0, 0, 12, 15)

        self.framerate_label = QtGui.QLabel()
        self.framerate_label.setText('Framerate')
        self.mainbox.layout().addWidget(self.framerate_label, 16, 16, 1, 4)

        self.current_frame = QtGui.QLabel()
        self.current_frame.setText('Frame 0')
        self.mainbox.layout().addWidget(self.current_frame, 2, 16, 1, 4)

        self.goto_frame_label = QtGui.QLabel()
        self.goto_frame_label.setText('Go to frame')
        self.mainbox.layout().addWidget(self.goto_frame_label, 3, 16, 1, 1)

        self.goto_frame_edit = QtGui.QLineEdit(' Enter frame number ')
        self.mainbox.layout().addWidget(self.goto_frame_edit, 3, 17, 1, 2)

        self.sessname = QtGui.QLabel()
        if not self.session is None:
            name = """ 
                    Session ID:    {},
                    Experiment:    {},
                    Date:          {},
                    Mouse ID:      {}.
            """.format(self.session.Metadata.session_id, self.session.Metadata.experiment,
                       self.session.Metadata.date, self.session.Metadata.mouse_id)
            self.sessname.setText('Session Metadata \n {}'.format(name))
        self.mainbox.layout().addWidget(self.sessname, 0, 16, 1, 2)

        self.trname = QtGui.QLabel()
        self.trname.setText('Trial Name')
        self.mainbox.layout().addWidget(self.trname, 1, 16, 1, 2)

        self.view = self.canvas.addViewBox()
        self.view.setAspectLocked(False)

        #  image plot
        self.img = pg.ImageItem(border='w')
        self.view.addItem(self.img)

        #  Pose Reconstruction
        self.poseplot = self.canvas.addPlot(title='Pose reconstruction')
        self.poseplot.invertY(True)
        self.pose_body = self.poseplot.plot(pen='g')
        self.canvas.nextRow()

        #  Velocity
        self.velplot = self.canvas.addPlot(title='Velocity and bodylength')
        self.vel_line = self.velplot.plot(pen=pg.mkPen('r', width=5))
        self.blength_line = self.velplot.plot(pen=pg.mkPen((150, 100, 220), width=3))

        # Ang vels
        self.angvelplot = self.canvas.addPlot(title='Head and Body ang. vel')
        self.head_ang_vel_line = self.angvelplot.plot(pen=pg.mkPen((100, 100, 25), width=5))
        self.body_ang_vel_line = self.angvelplot.plot(pen=pg.mkPen((150, 50, 75), width=5))

        # Progress
        self.progress_plot = self.progres_bar_canvas.addPlot(title='Progress bar', colspan=2)

        # buttons
        self.launch_btn = QPushButton(text='Launch')
        self.launch_btn.setObjectName('LaunchBtn')
        self.launch_btn.clicked.connect(lambda: self.update_by_frame(self))
        self.mainbox.layout().addWidget(self.launch_btn, 5, 16, 2, 2)

        self.stop_btn = QPushButton(text='Stop')
        self.stop_btn.setObjectName('StopBtn')
        self.stop_btn.clicked.connect(lambda: self.update_by_frame(self))
        self.mainbox.layout().addWidget(self.stop_btn, 6, 16, 2, 2)

        self.pause_btn = QPushButton(text='Pause')
        self.pause_btn.setObjectName('PauseBtn')
        self.pause_btn.clicked.connect(lambda: self.pause_playback(self))
        self.mainbox.layout().addWidget(self.pause_btn, 7, 16, 4, 2)

        self.resume_btn = QPushButton(text='Resume')
        self.resume_btn.setObjectName('ResumeBtn')
        self.resume_btn.clicked.connect(lambda: self.resume_playback(self))
        self.mainbox.layout().addWidget(self.resume_btn, 6, 16, 4, 2)

        self.faster_btn = QPushButton(text='Faster')
        self.faster_btn.clicked.connect(lambda: self.increase_speed(self))
        self.mainbox.layout().addWidget(self.faster_btn, 17, 17)

        self.slower_btn = QPushButton(text='Slower')
        self.slower_btn.clicked.connect(lambda: self.decrease_speed(self))
        self.mainbox.layout().addWidget(self.slower_btn, 17, 20)

        self.gotoframe_btn = QPushButton(text='Go to frame')
        self.gotoframe_btn.clicked.connect(lambda: self.change_frame(self))
        self.mainbox.layout().addWidget(self.gotoframe_btn, 4, 18)

        # List of trials widgets
        self.trlistlabel = QtGui.QLabel()
        self.trlistlabel.setText('Trials')
        self.mainbox.layout().addWidget(self.trlistlabel, 6, 19)


        self.trials_listw = QListWidget()
        self.mainbox.layout().addWidget(self.trials_listw, 7, 18, 2, 3)
        self.trials_listw.itemDoubleClicked.connect(self.load_trial_data)

        # Print tracking variables
        self.tracking_vars_label = QtGui.QLabel()
        self.tracking_vars_label.setText('Tracking data')
        self.mainbox.layout().addWidget(self.tracking_vars_label, 10, 17, 3, 2)

        # Define window geometry
        self.setGeometry(50, 50, 3600, 2000)

    def load_trial_data(self, trial_name):
        # Clear up a previously running trials
        self.data_loaded = False
        # self.velplot.cla()
        # self.poseplot.cla()
        # self.angvelplot.cla()

        # Get data
        trial_name = trial_name.text()
        self.trname.setText('Trial: {}'.format(trial_name))

        videonum = int(trial_name.split('_')[1].split('-')[0])
        self.video = self.videos[videonum][0]
        self.video_grabber = cv2.VideoCapture(self.video)
        self.num_frames = int(self.video_grabber.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.video_grabber.get(cv2.CAP_PROP_FPS)
        self.start_frame = self.session.Tracking[trial_name].metadata['Start frame']
        self.tracking_data = self.session.Tracking[trial_name].dlc_tracking['Posture']

        self.num_frames_trial = len(self.tracking_data['body'])

        # Plot the first frame
        self.video_grabber.set(1, self.start_frame)
        _, self.frame = self.video_grabber.read()
        self.frame = self.frame[:, :, 0]
        self.img.setImage(np.rot90(self.frame, 3))

        self.progress_bg = pg.QtGui.QGraphicsRectItem(0, 0, self.num_frames_trial, 1)
        self.progress_bg.setPen(pg.mkPen((180, 180, 180)))
        self.progress_bg.setBrush(pg.mkBrush((180, 180, 180)))
        self.progress_plot.addItem(self.progress_bg)

        stim_dur = 9
        self.stim_bg = pg.QtGui.QGraphicsRectItem(self.num_frames_trial/2, 0, stim_dur*self.video_fps, 1)
        self.stim_bg.setPen(pg.mkPen((100, 100, 200)))
        self.stim_bg.setBrush(pg.mkBrush((100, 100, 200)))
        self.progress_plot.addItem(self.stim_bg)
        self.progress_plot.setRange(xRange=[0, self.num_frames_trial], yRange=[0, 1])
        self.progress_line = self.progress_plot.plot(pen=pg.mkPen('r', width=5))


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
        x, y = self.tracking_data['body']['x'].values, self.tracking_data['body']['y'].values
        vel = self.tracking_data['body']['Velocity'].values
        blength = self.tracking_data['body']['Body length'].values
        blength = np.divide(blength, max(blength))
        head_ang_vel = self.tracking_data['body']['Head ang vel'].values
        body_ang_vel = self.tracking_data['body']['Body ang vel'].values
        bori = self.tracking_data['body']['Orientation'].values
        hori = self.tracking_data['body']['Head angle'].values

        bor = bori[framen]
        while bor>360:
            bor -= 360
        hor = hori[framen]
        while hor>360:
            hor -= 360

        self.tracking_vars_label.setText("""
        
        
            Position:                 {}, {}
            Velocity:                 {}
            Orientation [body]:       {}
            Orientation [head]:       {}
            Ang. Vel. [body]:         {}
            Ang. vel. [head]:         {}
        
        """.format(round(x[framen]), round(y[framen]), round(vel[framen],2), round(bor,2),
                   round(hor, 2), round(body_ang_vel[framen],2), round(head_ang_vel[framen],2)))


        xx = np.linspace(0, 100, 100)
        self.vel_line.setData(xx, vel[framen:framen+100])
        self.blength_line.setData(xx, blength[framen:framen+100])
        self.head_ang_vel_line.setData(xx, head_ang_vel[framen:framen+100])
        self.body_ang_vel_line.setData(xx, body_ang_vel[framen:framen+100])

        self.velplot.setRange(yRange=[0, max(vel)+(max(vel)/10)])
        max_ori = max(abs(head_ang_vel))+(max(abs(head_ang_vel))/10)
        self.angvelplot.setRange(yRange=[-max_ori, max_ori])

    def update_by_frame(self, event, start_frame=0):
        if not self.data_loaded:
            return

        f =start_frame
        self.curr_frame = f
        self.video_grabber.set(1, self.start_frame+start_frame)

        while True:
                now = time.time()
                dt = (now - self.lastupdate)
                if dt <= 0:
                    dt = 0.000000000001
                fps2 = 1.0 / dt
                self.lastupdate = now
                self.fps = self.fps * 0.9 + fps2 * 0.1
                tx = 'Plotting Frame Rate:  {fps:.3f} FPS'.format(fps=self.fps)
                self.framerate_label.setText(tx)

                ret, self.frame = self.video_grabber.read()
                if not ret:
                    print('finished video')
                    break

                self.frame = self.frame[:, :, 0]
                self.img.setImage(np.rot90(self.frame, 3))
                self.plot_pose(f)
                self.plot_tracking_data(f)

                self.progress_line.setData([f, f], [0, 1])

                self.current_frame.setText('Frame: {}'.format(f))
                f += 1
                pg.QtGui.QApplication.processEvents()

                if not self.data_loaded:
                    self.curr_frame = f
                    break

                if self.wait_ms:
                    time.sleep(self.wait_ms/1000)

    def pause_playback(self, event):
        self.data_loaded = False

    def resume_playback(self, event):
        self.data_loaded = True
        self.update_by_frame(None, start_frame=self.curr_frame)

    def decrease_speed(self, event):
        self.wait_ms += 50

    def increase_speed(self, event):
        if self.wait_ms > 0:
            self.wait_ms -= 50

    def change_frame(self, event):
        self.data_loaded = False
        try:
            target_frame = int(self.goto_frame_edit.text())
            self.data_loaded = True
            self.update_by_frame(None, target_frame)
        except:
            return

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    thisapp = App(None)
    thisapp.show()
    app.exec_()


















