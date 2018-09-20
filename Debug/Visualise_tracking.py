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
    """ Display frames from behavioural videos and tracking data for processed sessions.
     Only works for Tracked and Processed sessions """
    def __init__(self, session, parent=None):
        """ Set up class, initialise variables """
        super(App, self).__init__(parent)

        # Useful vars
        self.wait_ms = 0  # change speed of figure refresh rate
        self.colors = dict(head=(100, 220, 100), body=(220, 100, 100), tail=(100, 100, 220))  # Stuff to color bparts
        self.bodyparts = dict(head=['snout', 'Lear', 'Rear', 'neck'], body=['body'], tail=['tail'])
        self.bodyparts_plotdata = {}
        self.session = session  # session
        self.lastupdate = time.time()  # vars to keep track of plot refresh speed
        self.ready_to_plot = False   # flag to control plotting behaviour
        self.plot_wnd = 100

        # Create GUI
        self.define_layout()

        # Get session Data
        self.get_session_data()

        app = QtGui.QApplication(sys.argv)
        self.show()
        app.exec_()

    ####################################################################################################################
    def define_style_sheet(self):
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

    def create_label(self, txt, pos):
        obj = QtGui.QLabel()
        obj.setText(txt)
        if len(pos) == 4:
            self.mainbox.layout().addWidget(obj, pos[0], pos[1], pos[2], pos[3])
        elif len(pos) == 2:
            self.mainbox.layout().addWidget(obj, pos[0], pos[1])
        else:
            print('Cannot create label widget, wrong position parameter: {}'.format(pos))
        return obj

    def create_btn(self, txt, pos, name=None, func=None):
        obj = QPushButton(text=txt)
        if len(pos) == 4:
            self.mainbox.layout().addWidget(obj, pos[0], pos[1], pos[2], pos[3])
        elif len(pos) == 2:
            self.mainbox.layout().addWidget(obj, pos[0], pos[1])
        else:
            print('Cannot create label widget, wrong position parameter: {}'.format(pos))
        if name is not None:
            obj.setObjectName(name)
        if func is not None:
            obj.connect(lambda: func(self))
        return obj

    def define_layout(self):
        """ Create the layout of the figure"""
        self.define_style_sheet()

        # Main figure layout
        self.mainbox = QtGui.QWidget()
        self.mainbox.showFullScreen()
        self.setCentralWidget(self.mainbox)
        grid = QtGui.QGridLayout()
        grid.setSpacing(10)
        self.mainbox.setLayout(grid)

        # Create Plotting Canvases
        self.progres_bar_canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.progres_bar_canvas, 16, 0, 2, 15)

        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas, 0, 0, 12, 15)

        self.view = self.canvas.addViewBox()
        self.view.setAspectLocked(False)

        # Create Plotting Items
        #  Frame
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

        # Create text labels and lineedits
        self.framerate_label = self.create_label('Framerate',  (16, 16, 1, 4))

        self.current_frame = self.create_label('Frame 0', (2, 16, 1, 4))

        self.goto_frame_label = self.create_label('Goto frame', ( 3, 16, 1, 1))

        self.goto_frame_edit = QtGui.QLineEdit(' Enter frame number ')
        self.mainbox.layout().addWidget(self.goto_frame_edit, 3, 17, 1, 2)

        self.trname = self.create_label('Trial Name', (1, 16, 1, 2))

        self.tracking_vars_label = self.create_label('Tracking data', (10, 17, 3, 2))

        self.trlistlabel = self.create_label('Trials', (6, 19))

        if not self.session is None:
            name = """ 
                    Session ID:    {},
                    Experiment:    {},
                    Date:          {},
                    Mouse ID:      {}.
            """.format(self.session.Metadata.session_id, self.session.Metadata.experiment,
                       self.session.Metadata.date, self.session.Metadata.mouse_id)
            self.sessname = self.create_label('Session Metadata \n {}'.format(name), (0, 16, 1, 2))

        # Create buttons
        self.launch_btn = self.create_btn('Launch', (5, 16, 2, 2), name='LaunchBtn', func=self.update_by_frame)

        self.stop_btn = self.create_btn('Stop', (6, 16, 2, 2), name='StopBtn', func=self.update_by_frame)

        self.pause_btn = self.create_btn('Pause', (7, 16, 4, 2), name='PauseBtn', func=self.pause_playback)

        self.resume_btn = self.create_btn('Resume', (6, 16, 4, 2), name='ResumeBtn', func=self.resume_playback)

        self.faster_btn = self.create_btn('Faster', (17, 17), func=self.increase_speed)

        self.slower_btn = self.create_btn('Slower', (17, 20), func=self.decrease_speed)

        self.gotoframe_btn = self.create_btn('Go to frame', (4, 18), func=self.change_frame)

        # List widgets
        self.trials_listw = QListWidget()
        self.mainbox.layout().addWidget(self.trials_listw, 7, 18, 2, 3)
        self.trials_listw.itemDoubleClicked.connect(self.load_trial_data)

        # Define window geometry
        self.setGeometry(50, 50, 3600, 2000)

    ####################################################################################################################
    def get_session_data(self):
        """ Get paths of videofiles and names of tracking data + add these names to list widget"""
        if self.session is None:
            return
        self.videos = self.session.Metadata.video_file_paths
        self.trials = [t for t in self.session.Tracking.keys() if '-' in t]
        [self.trials_listw.addItem(tr) for tr in sorted(self.trials)]

    def load_trial_data(self, trial_name):
        """  get data from trial and initialise plots """
        # Clear up a previously running trials
        self.ready_to_plot = False  # TODO clean up plots when a new one is crated
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
        self.frame = self.prep_frame()
        self.img.setImage(self.frame)

        # Prep progress bar
        stim_dur = 9
        self.progress_bg = pg.QtGui.QGraphicsRectItem(0, 0, self.num_frames_trial, 1)
        self.progress_bg.setPen(pg.mkPen((180, 180, 180)))
        self.progress_bg.setBrush(pg.mkBrush((180, 180, 180)))
        self.progress_plot.addItem(self.progress_bg)
        self.stim_bg = pg.QtGui.QGraphicsRectItem(self.num_frames_trial/2, 0, stim_dur*self.video_fps, 1)
        self.stim_bg.setPen(pg.mkPen((100, 100, 200)))
        self.stim_bg.setBrush(pg.mkBrush((100, 100, 200)))
        self.progress_plot.addItem(self.stim_bg)
        self.progress_plot.setRange(xRange=[0, self.num_frames_trial], yRange=[0, 1])
        self.progress_line = self.progress_plot.plot(pen=pg.mkPen('r', width=5))

        # Change flag value
        self.ready_to_plot = True

    def prep_frame(self):
        _, frame = self.video_grabber.read()
        frame = frame[:, :, 0]
        return np.rot90(frame, 3)

    ####################################################################################################################
    def plot_pose(self, framen):
        # TODO add lines to posture plot
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
        # Get vars and prep them
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

        # Update plots
        xx = np.linspace(0, self.plot_wnd, self.plot_wnd)
        self.vel_line.setData(xx, vel[framen:framen+self.plot_wnd])
        self.blength_line.setData(xx, blength[framen:framen+self.plot_wnd])
        self.head_ang_vel_line.setData(xx, head_ang_vel[framen:framen+self.plot_wnd])
        self.body_ang_vel_line.setData(xx, body_ang_vel[framen:framen+self.plot_wnd])

        self.velplot.setRange(yRange=[0, max(vel)+(max(vel)/10)])
        max_ori = max(abs(head_ang_vel))+(max(abs(head_ang_vel))/10)
        self.angvelplot.setRange(yRange=[-max_ori, max_ori])

        # Display data
        self.tracking_vars_label.setText("""

            Position:                 {}, {}
            Velocity:                 {}
            Orientation [body]:       {}
            Orientation [head]:       {}
            Ang. Vel. [body]:         {}
            Ang. vel. [head]:         {}

        """.format(round(x[framen]), round(y[framen]), round(vel[framen], 2), round(bor, 2),
                   round(hor, 2), round(body_ang_vel[framen], 2), round(head_ang_vel[framen], 2)))

    def update_by_frame(self, event, start_frame=0):
        def get_plotting_fps(self):
            self.current_frame.setText('Frame: {}'.format(f))

            now = time.time()
            dt = (now - self.lastupdate)
            tx = 'Plotting Frame Rate:  {} FPS'.format(1/(dt*6000))
            self.framerate_label.setText(tx)

        """ update each plot and relevant widget frame by frame starting from start_frame """
        if not self.ready_to_plot:
            return

        # Set up start time
        f = start_frame
        self.curr_frame = f
        self.video_grabber.set(1, self.start_frame+start_frame)

        # Keep looping unless something goes wrong
        while True:
                get_plotting_fps(self)

                # Plot
                frame = self.prep_frame()
                self.img.setImage(frame)
                self.plot_pose(f)
                self.plot_tracking_data(f)
                self.progress_line.setData([f, f], [0, 1])

                f += 1
                pg.QtGui.QApplication.processEvents()

                if not self.ready_to_plot:
                    self.curr_frame = f
                    break

                if self.wait_ms:
                    time.sleep(self.wait_ms/1000)

    ####################################################################################################################
    def pause_playback(self, event):
        self.ready_to_plot = False

    def resume_playback(self, event):
        self.ready_to_plot = True
        self.update_by_frame(None, start_frame=self.curr_frame)

    def decrease_speed(self, event):
        self.wait_ms += 50

    def increase_speed(self, event):
        if self.wait_ms > 0:
            self.wait_ms -= 50

    def change_frame(self, event):
        self.ready_to_plot = False
        try:
            target_frame = int(self.goto_frame_edit.text())
            self.ready_to_plot = True
            self.update_by_frame(None, target_frame)
        except:
            return


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    thisapp = App(None)
    thisapp.show()
    app.exec_()


















