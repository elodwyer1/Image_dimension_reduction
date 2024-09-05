#-*- coding: utf-8 -*-


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import mlflow
import mlflow.tensorflow
import the_forest_palette

class DataGen(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size, image_w, image_h):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_w = image_w
        self.image_h =image_h

        
    def __load__(self, id_name):
        ###########Path###############
        image_path = os.path.join(self.path, id_name, "images", id_name) + ".npy"

        #######Reading Image###########
        #load 3D array of flux density and normalised degree of circular polarization.
        image = np.load(image_path, allow_pickle=True)
        
        flux_image = image[:, :, 0:1]

        #Resize array to pre-defined width and height using bilinear interpolation.
        resize = tf.keras.Sequential([layers.Resizing(self.image_h, self.image_w)])
        image = resize(flux_image).numpy()

        # Flatten the grayscale image array
        flattened_array = image.flatten()


        ###########Label############
        #Load class label for image.
        label_path = os.path.join(self.path, id_name, "label", id_name) + ".npy"
        label = np.load(label_path, allow_pickle=True).item()

        return flattened_array, label
    
    def __getitem__(self, index):
      #this generates input and output data for each batch.
        if(index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        
        im=[]
        labels=[]
        
        for id_name in files_batch:
            im_, label_ = self.__load__(id_name)
            im.append(im_)
            labels.append(label_)
        
        im=np.array(im)

        return im, labels
    



class Colors:
    def __init__(self):
        self.color_dict = the_forest_palette.forest_colors
        self.color_dict['ashmint'] = 'olive'
        self.color_dict['brightmint'] = 'darkorange'
        self.color_dict['softapple'] = 'peru'
        self.color_dict['yellow'] = 'yellow'
        self.color_dict['cornflowerblue'] = 'cornflowerblue'

    def get_colors(self):
        return self.color_dict