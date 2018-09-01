import numpy as np
import time
from moviepy.editor import *
import os
from tempfile import mkdtemp
from nptdms import TdmsFile
import psutil
import gc
import sys

class Converter():
    def __init__(self):

        self.start_next_thread, self.start_saving, self.save_wait = True, False, False

        # Variables
        # Path to the file location
        self.filepath = 'D:\\Camera_LDR-default-10160362-video.tdms'
        # Location of memmapped files
        self.memmap_path = mkdtemp(dir='D:\\')

        # Vars specific to the video to be converted
        self.skip_data_points = 4094
        self.width = 1100 + 52
        self.real_width = 1100
        self.height = 550
        self.frame_size = self.width * self.height
        self.real_frame_size = self.real_width * self.height

        self.open_memmapped_tdms()
        self.main()

    def main(self):
        chunk_size = 1000
        self.extractors = []
        self.frames = {}
        self.active_threads = 0
        processed_chunks = 0
        num_chunks = np.ceil(self.tot_frames/chunk_size)

        tmp = mkdtemp(dir='D:\\')

        import matplotlib.pyplot as plt

        self.tdms = self.tdms.__dict__['objects']["/'cam0'/'data'"].data.reshape(
            (self.width, self.height, self.tot_frames), order='F')


        plt.figure()
        plt.imshow(self.tdms[:, :, 30000], cmap='Greys')



        for framen in range(self.tot_frames+1):
            if not framen % 100:
                print(framen)
                if not framen%1000:
                    self.show_mem_stats()

            linear_frame = line[self.frame_size *framen:self.frame_size * (framen + 1)]

            square_frame = linear_frame.reshape(self.height, self.width)
            square_frame = square_frame[:self.height, :self.real_width]

            frames[:, :, framen] = square_frame

        a = 1

    def print_all_vars(self):
        def sizeof_fmt(num, suffix='B'):
            ''' By Fred Cirera, after https://stackoverflow.com/a/1094933/1870254'''
            for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
                if abs(num) < 1024.0:
                    return "%3.1f%s%s" % (num, unit, suffix)
                num /= 1024.0
            return "%.1f%s%s" % (num, 'Yi', suffix)





    def make_small_video(self, frames, tag):
        clip = ImageSequenceClip(frames, fps=30, with_mask=False)
        return clip

    def nofunc(self):
        pass

    def show_mem_stats(self):
        giga = 1073741824
        stats = psutil.virtual_memory()
        print(""" \n 
                Total memory:   {} GB
                  available:    {} GB
                       free:    {} GB
                       used:    {} GB
                    percent:    {}%\n
        """.format(round(stats.total/giga, 2), round(stats.available/giga, 2),
                   round(stats.free/giga, 2), round(stats.used/giga, 2), stats.percent))

    def open_memmapped_tdms(self):
        start = time.clock()  # keep track of how much time it takes
        print('opening tdms file')
        f = open(self.filepath, 'rb')
        self.f_size = os.path.getsize(self.filepath)  # size in bytes

        self.tot_frames = int((self.f_size - self.skip_data_points) / self.frame_size)  # num frames in the vieo

        self.tdms = TdmsFile(f, memmap_dir=self.memmap_path)  # open tdms binary file as a memmapped object
        print('File size: {}, loaded in {}s'.format(self.f_size, time.clock() - start))
        self.show_mem_stats()

    def stitch_small_videos(self):
        clips_l = [f for f in os.listdir(self.memmap_path) if 'mp4' in f]
        clips = [VideoFileClip(c) for c in clips_l]
        final = concatenate_videoclips(clips)
        final.write_videofile('{}test.mp4'.format(self.memmap_path))

    def delete_temp_data(self):
        os.rmdir(self.memmap_path)


if __name__ == "__main__":
    Converter()