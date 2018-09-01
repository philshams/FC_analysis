import numpy as np
import time
from moviepy.editor import *
import os
from tempfile import mkdtemp
from nptdms import TdmsFile
import psutil
from multiprocessing.dummy import Pool


class Converter():
    def __init__(self):

        self.start_next_thread, self.start_saving, self.save_wait = True, False, False

        # Variables
        # Path to the file location
        self.filepath = 'C:\\Users\\Federico\\Documents\\GitHub\\FC_analysis\\Utils\\Camera_LDR-default-10160362-video.tdms'
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
        chunk_size = 40*60
        self.extractors = []
        self.frames = {}
        self.active_threads = 0
        processed_chunks = 0


        while processed_chunks < 4:
            chunk_start = time.clock()

            print('Chunk {}'.format(processed_chunks))
            self.frames_ready = []
            n_loader_threads = 6
            step_size = 480
            steps = np.linspace(chunk_size * processed_chunks, chunk_size * (processed_chunks + 1),
                                n_loader_threads).astype(np.int32)

            steps2 = [x+step_size for x in steps]

            steps_tuples = [x for x in zip(steps, steps2)][0:-1]

            pool = Pool(len(steps_tuples))

            results = pool.map(self.get_frames_chunk, steps_tuples)
            self.frames_ready = [item for sublist in results for item in sublist]

            self.make_small_video(processed_chunks+1)

            pool.close()
            pool.join()
            del pool

            processed_chunks += 1
            print('This cunk took: {}'.format(time.clock()-chunk_start))

    def sort_chunks(self, data):
        self.frames_ready = []
        keys = [i.keys()[0] for i in data]
        print('Sorting chunks {}'.format(sorted(keys)))
        for key in sorted(keys):
            self.frames_ready.append(self.frames[str(key)])

        self.frames_ready = [item for sublist in self.frames_ready for item in sublist]

    def get_frames_chunk(self, frames):
        start_f, end_f = frames[0], frames[1]
        print('Started extracting frames {} to {} - in a separate thread'.format(start_f, end_f))

        frames_list = []
        for framen in range(start_f, end_f):
            if framen/(end_f-start_f) in [0.0, 0.5, 0.25, 0.75]:
                print('Progress: {}%'.format((framen/(end_f-start_f))*100))

            linear_frame = self.tdms.__dict__['objects']["/'cam0'/'data'"].data[self.frame_size *
                                                                        framen:self.frame_size * (framen + 1)]

            square_frame = linear_frame.reshape(self.height, self.width)
            square_frame = square_frame[:self.height, :self.real_width]

            frame = np.zeros((self.height, self.real_width, 3))
            frame[:] = np.nan
            frame[:, :, 0] = square_frame
            frame[:, :, 1] = square_frame
            frame[:, :, 2] = square_frame

            frames_list.append(frame)
        return frames_list

    def make_small_video(self, tag):
        print('Making video with {} frames'.format(len(self.frames_ready)))
        clip = ImageSequenceClip(self.frames_ready, fps=30, with_mask=False)
        self.frames_ready = []
        clip.write_videofile(os.path.join(self.memmap_path, "movie{}.mp4".format(tag)))
        self.show_mem_stats()

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
        f_size = os.path.getsize(self.filepath)  # size in bytes

        self.tot_frames = int((f_size - self.skip_data_points) / self.frame_size)  # num frames in the vieo

        self.tdms = TdmsFile(f, memmap_dir=self.memmap_path)  # open tdms binary file as a memmapped object
        print('File size: {}, loaded in {}s'.format(f_size, time.clock() - start))
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