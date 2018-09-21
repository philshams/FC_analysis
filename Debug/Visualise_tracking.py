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
from scipy import misc


class App(QtGui.QMainWindow):
    # TODO add possibility to invert flow of time
    """ Display frames from behavioural videos and tracking data for processed sessions.
     Only works for Tracked and Processed sessions """
    def __init__(self, sessions, parent=None):
        """ Set up class, initialise variables """
        super(App, self).__init__(parent)

        # Useful vars
        self.wait_ms = 0  # change speed of figure refresh rate
        self.colors = dict(head=(100, 220, 100), body=(220, 100, 100), tail=(100, 100, 220))  # Stuff to color bparts
        self.bodyparts = dict(head=['snout', 'Lear', 'Rear', 'neck'], body=['body'], tail=['tail'])
        self.bodyparts_plotdata = {}
        self.sessions = sessions  # sessions
        self.session = None  # place holder for current working session
        self.ready_to_plot = False   # flag to control plotting behaviour
        self.plot_wnd = 100
        self.plot_items = []
        self.counter = 0  # vars to check plotting framerate
        self.fps = 0.
        self.lastupdate = time.time()
        self.start_frame = 1200


        # Create GUI
        self.define_layout()

        # Create second window to display trial images
        self.previews = ImgsViwer()

        # Get session Data
        self.get_session_data(None)

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
                    
         QPlainTextEdit {
                        color: #ffffff;
                        font-size: 14pt;
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
                    
         QPushButton#GotoBtn {
                        font-size: 16pt;
                        min-width: 200px;
                        min-height: 40px;
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
            obj.clicked.connect(lambda: func(self))
        return obj

    def define_layout(self):
        # TODO make legends
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
        self.mainbox.layout().addWidget(self.progres_bar_canvas, 14, 0, 3, 15)

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
        self.canvas.nextRow()

        #  Velocity
        self.velplot = self.canvas.addPlot(title='Velocity and bodylength')

        # Ang vels
        self.angvelplot = self.canvas.addPlot(title='Head and Body ang. vel')

        # Progress
        self.progress_plot = self.progres_bar_canvas.addPlot(title='Progress bar', colspan=2)

        # Create text labels and lineedits
        self.framerate_label = self.create_label('Framerate',  (16, 18, 1, 2))
        self.current_frame = self.create_label('Frame 0', (14, 18, 1, 2))
        self.goto_frame_edit = QtGui.QLineEdit('Go to frame number')
        self.mainbox.layout().addWidget(self.goto_frame_edit, 15, 18, 1, 2)
        self.trname = self.create_label('Trial Name', (1, 16, 1, 2))
        self.tracking_vars_label = QtGui.QPlainTextEdit()
        self.tracking_vars_label.insertPlainText('Tracking data')
        self.mainbox.layout().addWidget(self.tracking_vars_label, 14, 16, 2, 2)
        self.trlistlabel = self.create_label('Trials', (8, 19))
        self.sesslistlabel = self.create_label('Sessions', (5, 19))
        if not self.sessions is None:
            name = """ 
                    Session ID:    ,
                    Experiment:    ,
                    Date:          ,
                    Mouse ID:      .
            """
            self.sessname = self.create_label('Session Metadata \n {}'.format(name), (0, 16, 1, 2))
        else:
            self.sessname = self.create_label('No session found', (0, 16, 1, 2))

        # Create buttons
        self.launch_btn = self.create_btn('Launch', (6, 16, 2, 2), name='LaunchBtn', func=self.update_by_frame)
        self.stop_btn = self.create_btn('Stop', (7, 16, 2, 2), name='StopBtn', func=self.stop_playback)
        self.pause_btn = self.create_btn('Pause', (8, 16, 4, 2), name='PauseBtn', func=self.pause_playback)
        self.resume_btn = self.create_btn('Resume', (7, 16, 4, 2), name='ResumeBtn', func=self.resume_playback)
        self.faster_btn = self.create_btn('Faster', (16, 16), func=self.increase_speed)
        self.slower_btn = self.create_btn('Slower', (16, 17), func=self.decrease_speed)
        self.gotoframe_btn = self.create_btn('Go to frame', (15, 20), func=self.change_frame, name='GotoBtn')

        # List widgets
        self.trials_listw = QListWidget()
        self.mainbox.layout().addWidget(self.trials_listw, 9, 18, 2, 3)
        self.trials_listw.itemDoubleClicked.connect(self.load_trial_data)

        self.sessions_listw = QListWidget()
        self.mainbox.layout().addWidget(self.sessions_listw, 6, 18, 2, 3)
        self.sessions_listw.itemDoubleClicked.connect(self.get_session_data)

        # Define window geometry
        self.setGeometry(50, 50, 3600, 2000)

        # Create plot items
        self.define_plot_items()

    def define_plot_items(self):
        self.vel_line = self.velplot.plot(pen=pg.mkPen('r', width=5))
        self.blength_line = self.velplot.plot(pen=pg.mkPen((150, 100, 220), width=3))
        self.plot_items.append(self.vel_line)
        self.plot_items.append(self.blength_line)

        self.head_ang_vel_line = self.angvelplot.plot(pen=pg.mkPen((100, 100, 25), width=3))
        self.body_ang_vel_line = self.angvelplot.plot(pen=pg.mkPen((150, 50, 75), width=3))
        self.head_body_angle_diff_line = self.angvelplot.plot(pen=pg.mkPen((150, 200, 150), width=5))
        self.plot_items.append(self.head_ang_vel_line)
        self.plot_items.append(self.body_ang_vel_line)
        self.plot_items.append(self.head_body_angle_diff_line)

    ####################################################################################################################
    def get_session_data(self, event):
        """ Get paths of videofiles and names of tracking data + add these names to list widget"""
        if self.sessions is None:
            return

        if event is None:   # function not called by widget click
            [self.sessions_listw.addItem(sess) for sess in sorted(list(self.sessions.keys()))]
            session_name = sorted(list(self.sessions.keys()))[0]
        else:  # Clean up trials list widget and laod data
            for i in range(self.trials_listw.count()):
                self.trials_listw.model().removeRow(0)

            session_name = event.text()

        session = self.sessions[session_name]
        self.session = session
        self.videos = session.Metadata.video_file_paths
        self.trials = [t for t in session.Tracking.keys() if '-' in t]
        [self.trials_listw.addItem(tr) for tr in sorted(self.trials)]

        name = """ 
                   Session ID:    {},
                   Experiment:    {},
                   Date:          {},
                   Mouse ID:      {}.
                   """.format(session.Metadata.session_id, session.Metadata.experiment,
                              session.Metadata.date, session.Metadata.mouse_id)
        self.sessname.setText('Session Metadata \n {}'.format(name))

        # Load images in secondary window
        self.previews.get_images(sorted(self.trials))

    def load_trial_data(self, trial_name):
        """  get data from trial and initialise plots """
        # Clear up a previously running trials
        self.ready_to_plot = False
        [p.setData([], []) for p in self.plot_items]
        self.curr_frame = 0

        # Get data
        trial_name = trial_name.text()
        self.previews.set_img(trial_name)
        self.trname.setText('Trial: {}'.format(trial_name))

        videonum = int(trial_name.split('_')[1].split('-')[0])
        self.video = self.videos[videonum][0]
        self.video_grabber = cv2.VideoCapture(self.video)
        self.num_frames = int(self.video_grabber.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.video_grabber.get(cv2.CAP_PROP_FPS)
        self.trial_start_frame = self.session.Tracking[trial_name].metadata['Start frame']

        self.tracking_data = self.session.Tracking[trial_name].dlc_tracking['Posture']
        self.num_frames_trial = len(self.tracking_data['body'])

        # Plot the first frame
        self.video_grabber.set(1, self.trial_start_frame)
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

    def prep_frame(self):
        _, frame = self.video_grabber.read()
        frame = frame[:, :, 0]
        return np.rot90(frame, 3)

    ####################################################################################################################
    def plot_pose(self, framen):
        # TODO add lines to posture plot
        # TODO fix timing issues
        if framen == self.start_frame:
            self.bodyparts_plotdata = {}

        for bp, data in self.tracking_data.items():
            for key,parts in self.bodyparts.items():
                if bp in parts:
                    if bp == 'body':
                        centre = data.loc[framen].x, data.loc[framen].y
                    col = self.colors[key]
                    if framen == self.start_frame:
                        self.bodyparts_plotdata[bp] = self.poseplot.plot([data.loc[framen].x],
                                           [data.loc[framen].y],
                                           pen=col, symbolBrush=col, symbolPen='w', symbol='o', symbolSize=30)
                        self.plot_items.append(self.bodyparts_plotdata[bp])
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
        hb_ori_diff = np.subtract(hori, bori)

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
        self.head_body_angle_diff_line.setData(xx, hb_ori_diff[framen:framen+self.plot_wnd])

        self.velplot.setRange(yRange=[0, max(vel)+(max(vel)/10)])
        max_ori = max(abs(head_ang_vel))+(max(abs(head_ang_vel))/10)
        self.angvelplot.setRange(yRange=[-max_ori, max_ori])

        # Display data
        self.tracking_vars_label.setPlainText("""
        Tracking data
        Position:                 {}, {}
        Velocity:                 {}
        Orientation [body]:       {}
        Orientation [head]:       {}
        Ang. Vel. [body]:         {}
        Ang. vel. [head]:         {}

        """.format(round(x[framen]), round(y[framen]), round(vel[framen], 2), round(bor, 2),
                   round(hor, 2), round(body_ang_vel[framen], 2), round(head_ang_vel[framen], 2)))

    def update_by_frame(self, event, start_frame=1200):
        def get_plotting_fps(self):
            self.current_frame.setText('Frame: {}'.format(f))

            now = time.time()
            dt = (now - self.lastupdate)
            if dt <= 0:
                dt = 0.000000000001
            fps2 = 1.0 / dt
            self.lastupdate = now
            self.fps = self.fps * 0.9 + fps2 * 0.1
            tx = 'Plotting Frame Rate:  {} FPS'.format(round(self.fps, 0))
            self.framerate_label.setText(tx)

        """ update each plot and relevant widget frame by frame starting from start_frame """
        if event is not None:
            self.ready_to_plot = True

        if not self.ready_to_plot:
            return

        # Set up start time

        self.start_frame = start_frame
        f = start_frame
        self.curr_frame = f
        self.video_grabber.set(1, self.trial_start_frame+start_frame)

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

    def stop_playback(self, event):
        self.ready_to_plot = False
        [p.setData([], []) for p in self.plot_items]


class ImgsViwer(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(ImgsViwer, self).__init__(parent)

        self.define_layout()
        self.show()

        self.images_flds = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\data\\z_TrialImages'
        self.images = {}
        self.curr_img = None

    def define_layout(self):
        # Main window color
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
                    }          """)

        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QColor(40, 40, 40, 255))
        self.setPalette(p)

        self.mainbox = QtGui.QWidget()
        self.mainbox.showFullScreen()
        self.setCentralWidget(self.mainbox)
        grid = QtGui.QGridLayout()
        grid.setSpacing(10)
        self.mainbox.setLayout(grid)

        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas, 0, 0, 6, 6)

        self.view = self.canvas.addViewBox()
        self.view.setAspectLocked(False)
        self.img = pg.ImageItem(border='w')
        self.view.addItem(self.img)

        self.curr_img_label = App.create_label(self, 'Current Image', (7, 1))
        self.prev_btn = App.create_btn(self, 'Prev', (8, 1), func=self.prev_img)
        self.next_btn = App.create_btn(self, 'Next', (8, 3), func=self.next_img)

        self.setGeometry(3835, 40, 1450, 1400)

    def get_images(self, trials):
        if len(self.images):
            self.discard_images()

        for tr_name in trials:
            self.images[tr_name] = os.path.join(self.images_flds, tr_name + '.png')

        # self.set_img(trials[0])

    def discard_images(self):
        self.images = {}

    def set_img(self, trial):
        self.curr_img_label.setText(trial)
        img = np.rot90(misc.imread(self.images[trial]), 3)
        self.img.setImage(img)
        self.curr_img = list(self.images.keys()).index(trial)

    def next_img(self, event):
        self.curr_img += 1
        if self.curr_img < len(list(self.images.keys())):
            trial = list(self.images.keys())[self.curr_img]
            self.set_img(trial)

    def prev_img(self, event):
        self.curr_img -= 1
        if self.curr_img > 0:
            trial = list(self.images.keys())[self.curr_img]
            self.set_img(trial)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    thisapp = App(None)
    thisapp.show()
    app.exec_()


















