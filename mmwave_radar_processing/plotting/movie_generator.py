import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import tqdm
import imageio

from cpsl_datasets.cpsl_ds import CpslDS

class MovieGenerator:

    def __init__(self,
                 cpsl_dataset:CpslDS,
                 temp_dir_path="~/Downloads/odometry_temp",
                 ) -> None:
        
        self.dataset:CpslDS = cpsl_dataset
        
        self.temp_dir_path = os.path.expanduser(temp_dir_path)
        self.temp_file_name = "frame"

        self.next_frame:int = 0

        self.figure:Figure = None
        self.axs:list[Axes] = []

        self.reset()

    ####################################################################
    #Helper functions - directories
    #################################################################### 

    def _create_temp_dir(self):

        path = self.temp_dir_path
        if os.path.isdir(path):

            print("found temp dir: {}".format(path))

            # clear the temp directory
            self._clear_temp_dir()

        else:
            print("creating temp directory: {}".format(path))
            os.makedirs(path)

        return

    def _clear_temp_dir(self):

        path = self.temp_dir_path

        if os.path.isdir(path):
            print("clearing temp directory {}".format(path))
            for file in os.listdir(path):

                file_path = os.path.join(path, file)

                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print("Failed to delete {}".format(file_path))

        else:
            print("temp directory {} not found".format(path))

    def _delete_temp_dir(self):

        path = self.temp_dir_path

        if os.path.isdir(path):

            print("deleting temp dir: {}".format(path))

            # clear the directory first
            self._clear_temp_dir()

            # delete the directory
            os.rmdir(path)

        else:
            print("temp directory {} not found".format(path))
    
    ####################################################################
    #Helper functions - movie generation
    #################################################################### 

    def reset(self):

        self._create_temp_dir()
        self.next_frame = 0

    def initialize_figure(self,nrows=2,ncols=2,figsize=(10,10),wspace=0.3,hspace=0.3):

        self.figure,self.axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize
        )

        self.figure.subplots_adjust(wspace=wspace,hspace=hspace)

    def clear_axes(self):

        for ax in self.axs.flat:
            ax.cla()
    
    def save_frame(self,clear_axs = True):
        
        #save the current frame
        file_name = "{}_{}.png".format(self.temp_file_name,self.next_frame+1000)
        path = os.path.join(self.temp_dir_path,file_name)
        self.figure.savefig(path,format="png",dpi=200)

        self.next_frame+=1

        #clear the axes if desired
        if clear_axs:
            self.clear_axes() 

    
    def save_movie(self,video_file_name:str="result.mp4",fps:int=20):

        writer = imageio.get_writer(video_file_name,fps=fps)
        for i in tqdm.tqdm(range(self.next_frame)):

            file_name = "{}_{}.png".format(self.temp_file_name,i+1000)
            path = os.path.join(self.temp_dir_path,file_name)

            writer.append_data(imageio.imread(path))
        
        writer.close()
    
    ####################################################################
    #Child class functions
    #################################################################### 
    def generate_movie_frames(
            self,
            **kwargs):
        """Generates the movie frames. May need to be modified by child class
        """

        for i in tqdm.tqdm(range(self.dataset.num_frames)):

            #update the current frame
            self.generate_movie_frame(idx=i,**kwargs)

            #save the frame
            self.save_frame(clear_axs=True)
    
    def generate_movie_frame(self,idx,**kwargs):
        """Functionality implemented by child class

        Args:
            idx (_type_): _description_
        """
        pass   