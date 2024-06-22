import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import h5py
import numpy as np
import re
from PIL import Image, ImageTk
import os
import subprocess

import get_conventional as con
import get_radiomics as radio


def find_volume_number(file_path):
    # Regular expression pattern: search for 'volume_' followed by one or more digits
    pattern = r"volume_(\d+)"
    
    # Use re.search to find the first match
    match = re.search(pattern, file_path)
    
    if match:
        # Return the first matched number (match.group(1) returns the content within the first parentheses)
        return match.group(1)
    else:
        # If no match is found, return None or an appropriate message
        return None



class MriApp:
    def __init__(self, master):
        self.master = master
        self.master.title("MRI Slice Viewer")

        self.image = None
        self.mask=None
        self.mode=0 # 0: image/1: masked
        self.file_path = 'archive/BraTS2020_training_data'


        # This is based on feature selection
        self.radiomic_feature_list=['original_shape_Sphericity',
        'original_shape_SurfaceVolumeRatio',
        'original_shape_Flatness',
        'original_shape_Maximum3DDiameter',
        'original_shape_Elongation',
        'original_shape_LeastAxisLength',
        'original_shape_Maximum2DDiameterSlice',
        'original_shape_MajorAxisLength',
        'original_shape_MeshVolume',
        'original_shape_SurfaceArea',
        'original_firstorder_Mean',
        'original_firstorder_RootMeanSquared',
        'original_firstorder_90Percentile',
        'original_firstorder_Median',
        'original_firstorder_InterquartileRange',
        'original_firstorder_RobustMeanAbsoluteDeviation',
        'original_firstorder_Maximum',
        'original_firstorder_MeanAbsoluteDeviation',
        'original_firstorder_Range',
        'original_firstorder_10Percentile',
        'original_glszm_GrayLevelNonUniformity',
        'original_firstorder_Variance',
        'original_glszm_ZonePercentage',
        'original_glszm_SizeZoneNonUniformity',
        'original_glszm_ZoneEntropy',
        'original_gldm_DependenceNonUniformityNormalized',
        'original_gldm_LargeDependenceEmphasis',
        'original_gldm_DependenceEntropy',
        'original_glszm_SizeZoneNonUniformityNormalized',
        'original_glrlm_RunPercentage']

        # show the image
        self.image_label = tk.Label(master)
        self.image_label.grid(row=0, column=0, columnspan=2)

        # 'Load Slice Directory' Button
        self.load_button = tk.Button(master, text='Load Slice Directory', command=self.load_directory)
        self.load_button.grid(row=8, column=0, sticky='w')
       
        # to select channel
        self.channel_var = tk.StringVar(master)
        self.channel_var.set("T1")  # set default
        self.channel_options = ['T1', 'T1Gd', 'T2', 'T2-FLAIR']
        self.channel_menu = tk.OptionMenu(master, self.channel_var, *self.channel_options,command=self.load_directory)
        self.channel_menu.grid(row=9, column=1)

        self.channel_label = tk.Label(master, text="Channel:")
        self.channel_label.grid(row=9, column=0, sticky='w')

        

        # to show annotation or not 
        self.annotation_var = tk.StringVar(master)
        self.annotation_var.set("Off")  # set default
        self.annotation_options = ['On', 'Off']
       
       
        # 【TODO】
        self.annotation_menu = tk.OptionMenu(master, self.annotation_var, *self.annotation_options)
        self.annotation_menu.grid(row=10, column=1)

        self.annotation_label = tk.Label(master, text="Annotation:")
        self.annotation_label.grid(row=10, column=0, sticky='w')

        # A slider to choose slice id 
        self.slice_id_slider = tk.Scale(master, from_=0, to=154, orient='horizontal',command=self.change_slice_id)
        self.slice_id_slider.grid(row=11, column=1)
        
        # set default value to 10
        self.slice_id_slider.set(10)
        self.slice_label = tk.Label(master, text="Slice ID:")
        self.slice_label.grid(row=11, column=0, sticky='w')

        # to get conventional features
        self.path_subfolders='./test_dir' # defailt dir for testing
        self.extract_conventional_button = tk.Button(master, text='Extract Conventional Features', command=self.extract_conventional_features)
        self.extract_conventional_button.grid(row=12, column=0, sticky='w')
        # self.load_button = tk.Button(master, text='Load Slice Directory', command=self.load_directory)
        # self.load_button.grid(row=0, column=0, sticky='w')
       

        # button to extract radiomic features
        self.extract_radiomic_button = tk.Button(master, text='Extract Radiomic Features', command=self.extract_radiomic_features)
        self.extract_radiomic_button.grid(row=12, column=1, sticky='e')
    
    def extract_conventional_features(self):
        self.path_subfolders=filedialog.askdirectory()
        print("Extracting conventional features...")
        con.get_all(self.path_subfolders)
        if os.path.exists(self.path_subfolders):  # Check if the directory exists
            subprocess.run(["open" if os.name == 'posix' else "explorer", self.path_subfolders], check=True)
        
            
    
    def extract_radiomic_features(self):
        self.path_subfolders=filedialog.askdirectory()
        print("Extracting radiomic features...")
        radio.get_all_radiomics(self.path_subfolders,col_list=self.radiomic_feature_list)
        if os.path.exists(self.path_subfolders):  # Check if the directory exists
            subprocess.run(["open" if os.name == 'posix' else "explorer", self.path_subfolders], check=True)

    def load_image(self, mode=0):  # 0: image / 1: masked
        # Show selected channel
        channel_names = ['T1', 'T1Gd', 'T2', 'T2-FLAIR']
        channel_id = channel_names.index(self.channel_var.get())
        channel_image = self.image[channel_id, :, :]

        # Normalize the image array to 0-255
        normalized_array = (channel_image - channel_image.min()) / (channel_image.max() - channel_image.min())

        if mode == 1:  # Show photo with mask
            sum_mask = self.mask[0, :, :] + self.mask[1, :, :] + self.mask[2, :, :]
            yellow_mask = np.stack([sum_mask, sum_mask, np.zeros_like(sum_mask)], axis=-1)
            rgb_image = np.stack([normalized_array] * 3, axis=-1)
            overlay_image = (rgb_image) + (yellow_mask * 0.5)
            normalized_array = (overlay_image - overlay_image.min()) / (overlay_image.max() - overlay_image.min())

        scaled_array = (255 * normalized_array).astype(np.uint8)

        # Rotate the image counterclockwise by 90 degrees
        pil_image = Image.fromarray(scaled_array).rotate(90,
                                                         expand=True)  # `expand=True` adjusts the size to fit the new orientation

        self.photo_image = ImageTk.PhotoImage(pil_image)
        self.image_label.config(image=self.photo_image)
        self.image_label.image = self.photo_image

        # Explicitly update the widget
        self.image_label.update_idletasks()  # Use this if the rotation still doesn't appear

    def load_directory(self):
        self.file_path = filedialog.askdirectory()
        self.change_slice_id()


    def change_slice_id(self, value=10):    
        if self.file_path:
            print("Directory chosen:", self.file_path)
            volume_number = find_volume_number(self.file_path)
            print(volume_number)
            
            data = {} # to load h5 file
            # file_name = self.file_path + "/volume_" + volume_number + "_slice_" + str(self.slice_id_slider.get()) + ".h5"
            file_name = os.path.join(self.file_path, f"volume_{volume_number}_slice_{self.slice_id_slider.get()}.h5")

            # print(file_name)
            with h5py.File(file_name, 'r') as file:
                # self.dataset = file['image'].transpose(2,0,1)
                for key in file.keys():  # keys: image & mask
                    data[key] = file[key][()]          
                self.image = data['image'].transpose(2, 0, 1)
                self.mask = data['mask'].transpose(2, 0, 1)
                # annotation mode
                self.mode = 1 if self.annotation_var.get() == 'On' else 0
                self.load_image(mode=self.mode)

    def add_annotation(self):
        if self.annotation_var.get()=='On':
            self.merge_mask()

            

    def merge_mask(self):
    
        merged_array=self.mask[1:0:0]+self.mask[0:0:0]+self.mask[2:0:0]
        scaled_image = (self.image - self.image.min()) / (self.image.max() - self.image.min())
        scaled_array = (scaled_array - scaled_array.min()) / (scaled_array.max() - scaled_array.min())

        # a yellow mask for annotation
        yellow_image = np.stack([scaled_array, scaled_array, np.zeros_like(scaled_array)], axis=-1)
        yellow_image[:, :, :2] *= 255  # red + green = yellow

        # background in RGB
        background_rgb = np.stack([scaled_image]*3, axis=-1) * 255

        # make the mask alpha=0.5
        alpha = 0.5
        blended_image = (1 - alpha) * background_rgb + alpha * yellow_image
        blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)  # make sure it is between 0-255

        # transfer to PIL
        pil_image = Image.fromarray(blended_image, 'RGB')
        self.photo_image = ImageTk.PhotoImage(pil_image)
        self.image_label.config(image=self.photo_image)
        self.image_label.image = self.photo_image  


    def load_directory(self):
        self.file_path = filedialog.askdirectory()
        self.change_slice_id()
        

# run this app :)
root = tk.Tk()
app = MriApp(root)
root.mainloop()



