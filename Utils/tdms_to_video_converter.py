from Utils.imports import *


class TDMs_to_Video():
    """  current implementation: takes one .tdms video and saves it into as a number of .mp4 videos in a temp foldre"""
    # TODO extract video parametrs from .tdms
    # TODO deal with batch processing
    # TODO Stitch .mp4s together
    # TODO convert mp4 to avi
    # TODO easier handling of saving destination
    def __init__(self):
        self.start_time = time.clock()

        # Specify path to TDMS file and temp folder where to store data
        # self.tempdir = mkdtemp(dir='D:\\')
        self.tempdir = 'D:\\temp'
        filefld = 'Z:\\rig_bigrig\\cameratest'
        filename = 'Prot18-24-default-119418055-video.tdms'
        self.filepath = os.path.join(self.tempdir, filename)

        # HARDCODED variables about the video recorded
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

        # load TDMS data
        self.get_data()

        # write to video
        self.write_to_video()

        # Print how long it took
        print('It took {}s to process a file of {} bytes'.format(time.clock() - self.start_time, self.f_size))

    ####################################################################################################

    def get_data(self):
        """ loads data from the .tdms file """
        print('Opening binary')  # necessary, otherwise TdmsFile breaks. doesnt slow down process
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
        self.show_mem_stats()

        # reshape data
        self.tdms = tdms[:, :, :self.real_width]

    def write_to_video(self):
        """ writes frames data from self.tdms to .mp4 videos. Pooled for faster execution"""
        if self.num_processes == 1:
            self.write_clip([0, self.tot_frames])
        else:
            # Get frames range for each video writer that will run in parallel
            steps = np.linspace(0, self.tot_frames, self.num_processes + 1).astype(int)
            step = steps[1]
            steps2 = np.asarray([x + step for x in steps])
            limits = [s for s in zip(steps, steps2)][:-1]
            # start writing
            pool = ThreadPool(self.num_processes)
            _ = pool.map(self.write_clip, limits)

    @staticmethod
    def show_mem_stats():
        """ shows memory usage """
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
        """ create a .cv2 videowriter and start writing """
        vidname = 'output_{}.mp4'.format(limits[0])
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        videowriter = cv2.VideoWriter(os.path.join(self.tempdir, vidname), fourcc,
                                      120, (self.real_width, self.height), self.iscolor)

        for framen in tqdm(range(limits[0], limits[1])):
            videowriter.write(self.tdms[framen])
        videowriter.release()

if __name__=="__main__":
    converter = TDMs_to_Video()