from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('axes', edgecolor=[0.8, 0.8, 0.8])
matplotlib.rcParams['text.color'] = [0.8, 0.8, 0.8]
matplotlib.rcParams['axes.labelcolor'] = [0.8, 0.8, 0.8]
matplotlib.rcParams['axes.labelcolor'] = [0.8, 0.8, 0.8]
matplotlib.rcParams['xtick.color'] = [0.8, 0.8, 0.8]
matplotlib.rcParams['ytick.color'] = [0.8, 0.8, 0.8]
params = {'legend.fontsize': 16,
          'legend.handlelength': 2}
plt.rcParams.update(params)

from Utils.maths import calc_distance_2d


class PoseReconstructor:
    def __init__(self, data):
        posture_ = namedtuple('posture', 'Lear snout Rear neck body tail')
        self.posture = posture_(data['Lear'], data['snout'], data['Rear'], data['neck'], data['body'], data['tail'])

        self.axes = dict(head_main=[self.posture.neck, self.posture.snout],
                         head_minor=[self.posture.Lear, self.posture.Rear],
                         body=[self.posture.body, self.posture.neck],
                         tail=[self.posture.tail, self.posture.body],
                         side1=[self.posture.Lear, self.posture.snout],
                         side2=[self.posture.snout, self.posture.Rear],
                         side3=[self.posture.Rear, self.posture.neck],
                         side4=[self.posture.neck, self.posture.Lear])

        self.loop_postures()

    def loop_postures(self):
        self.colors = dict(head_main=[.8, .2, .2], head_minor=[.5, .1, .1], body=[.2, .6, .2], tail=[.4, .8, .4],
                      side=[.6, .4, .4])
        plt.ion()
        f, self.ax = plt.subplots(facecolor=[.2, .2, .2])
        self.ax.set(facecolor=[.2, .2, .2], xlim=[200, 400])

        for idx in range(len(self.posture.body)):
            self.get_lengths(idx)
            self.update_body_plot(idx)

            self.ax.cla()

    def get_lengths(self, idx):
        head_main_axis = calc_distance_2d([self.axes['head_main'][0], self.axes['head_main'][1]])
        a = 1

    def update_body_plot(self, idx):
        centre = self.posture.body.loc[idx]
        self.ax.set(xlim=[centre.x - 100, centre.x + 100], ylim=[centre.y - 100, centre.y + 100])

        [self.plot_body_line(axis, idx) for axis in self.axes.keys()]

        plt.draw()
        plt.pause(0.1)

    def plot_body_line(self, axisname, idx):
        bp1 = self.axes[axisname][0]
        bp2 = self.axes[axisname][1]

        if not 'side' in axisname:
            self.ax.plot([bp1.x[idx], bp2.x[idx]], [bp1.y[idx], bp2.y[idx]], color=self.colors[axisname], linewidth=3)
        else:
            self.ax.plot([bp1.x[idx], bp2.x[idx]], [bp1.y[idx], bp2.y[idx]], color=self.colors['side'],
                         linewidth=3, alpha=0.5)

