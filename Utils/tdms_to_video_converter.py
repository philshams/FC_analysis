import numpy as np
import os
from tempfile import mkdtemp
from nptdms import TdmsFile
import psutil
import gc
import time
from multiprocessing.dummy import Pool as ThreadPool
import matplotlib.pyplot as plt
from functools import partial

import cv2


class TDMs_to_Video():
    def __init__(self):
        # Specify path to TDMS file and temp folder where to store data

        # self.tempdir = mkdtemp(dir='D:\\')
        self.start_time = time.clock()

        self.tempdir = 'D:\\temp'
        filefld = 'Z:\\rig_bigrig\\cameratest'
        filename = 'Prot18-24-default-119418055-video.tdms'

        self.filepath = os.path.join(self.tempdir, filename)

        # HARDCODED variabels about the video recorded
        # TODO extract these from tdms header
        skip_data_points = 4094
        self.real_width = 1936
        self.width = self.real_width + 48

        self.height = 1216
        frame_size = self.width * self.height
        self.real_frame_size = self.real_width * self.height
        self.f_size = os.path.getsize(self.filepath)  # size in bytes
        self.tot_frames = int((self.f_size - skip_data_points) / frame_size)  # num frames

        self.iscolor = False  # is the video RGB or greyscale
        print('Total number of frames {}'.format(self.tot_frames))

        # Number of parallel processes for faster writing to video
        self.num_processes = 3

        # Call MAIN to get the work done
        self.main()

    def main(self):
        # load TDMS data
        self.get_data()

        # write to video
        self.write_to_video()

        # Print how long it took

        print('It took {}s to process a file of {} bytes'.format(time.clock() - self.start_time, self.f_size))

    ####################################################################################################

    def get_data(self):
        print('Opening binary')
        bfile = open(self.filepath, 'rb')
        self.show_mem_stats()

        print('Opening mmap tdms')
        tdms = TdmsFile(bfile, memmap_dir=self.tempdir)  # open tdms binary file as a memmapped object
        self.show_mem_stats()

        #  show data
        # plt.figure()
        # plt.plot(tdms.__dict__['objects']["/'cam0'/'data'"].data[0:10000])

        print('Extracting data')
        tdms = tdms.__dict__['objects']["/'cam0'/'data'"].data.reshape((self.tot_frames, self.height, self.width),
                                                                       order='C')
        self.show_mem_stats()

        print('Got data, cleaning up cached memory')
        gc.collect()

        print('Cleaning up data')
        self.tdms = tdms[:, :, :self.real_width]

    def write_to_video(self):
        if self.num_processes == 1:
            self.write_clip([0, self.tot_frames])
        else:
            steps = np.linspace(0, self.tot_frames, self.num_processes + 1).astype(int)
            step = steps[1]
            steps2 = np.asarray([x + step for x in steps])
            limits = [s for s in zip(steps, steps2)][:-1]

            pool = ThreadPool(self.num_processes)
            _ = pool.map(self.write_clip, limits)

    @staticmethod
    def show_mem_stats():
            giga = 1073741824
            stats = psutil.virtual_memory()
            print("""Total memory:           {} GB
              available:    {} GB
                   free:    {} GB
                   used:    {} GB
                percent:    {}%
            """.format(round(stats.total/giga, 2), round(stats.available/giga, 2),
                       round(stats.free/giga, 2), round(stats.used/giga, 2), stats.percent))
            return stats.available


    def write_clip(self, limits):
        vidname = 'output_{}.mp4'.format(limits[0])
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 0X00000021
        videowriter = cv2.VideoWriter(os.path.join(self.tempdir, vidname), fourcc,
                                      120, (self.real_width, self.height), self.iscolor)

        for framen in range(limits[0], limits[1]):
            print(framen)
            videowriter.write(self.tdms[framen])
        videowriter.release()






if __name__=="__main__":
    converter = TDMs_to_Video()