import numpy as np
from PIL import Image
import csv
from pathlib import Path
import h5py
import os

def read_single_disk(image_path):

    image = np.array(Image.open(image_path),dtype='float64')

    return image

def read_many_disk(disk_glob):
    """ Reads image from disk.
        Parameters:
        ---------------
        disk_glob   iglob iterator containing image files to be imported
        img_size    Size of the image files to be imported

        Returns:
        ----------
        images      images array, (N, img_size, img_size, 3) to be stored
        labels      associated meta data, int label (N, 3)
    """
    images = []
    labels = {}
    
    for i, image_id in enumerate(disk_glob):
        
        images.append(np.array(Image.open(image_id)))
        img_cat = os.path.dirname(image_id).split("\\")
        labels[i]=[img_cat[1],img_cat[2].split("___")[0],img_cat[2].split("___")[1],
            "".join([img_cat[1],img_cat[2].split("___")[0],img_cat[2].split("___")[1]])]
        print(i)
    return images, labels

def store_many_hdf5(images, labels, file_name):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    with h5py.File(file_name, "w") as my_file:
    
        my_file.create_dataset("images", np.shape(images), h5py.h5t.STD_U8BE, data=images)
        my_file.create_dataset("meta", (len(labels)),h5py.h5t.STD_U32BE, data=labels)

    
def read_hdf5(hdf_file):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, h, w, rgb) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [],[]
    
    with h5py.File(hdf_file, "r+") as my_file:
    
        images = np.array(my_file["/images"]).astype("uint8")
        labels = np.array(my_file["/meta"]).astype("uint32")
    
    return images, labels

def display_image(img_np):
    
    # Show an image given in the form of an NP array on the screen
    # NB - PIL fromarray is expecting the integer values to be INT8 type for RGB, so recast just in case
    
    my_img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    my_img.show()
    
def store_single_disk(image, disk_dir, image_id):
    """ Stores a single image as a .png file on disk.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        disk_dir    directory to save the image
        image_id    integer name for image
    """
    # We get image as an array, and save it to a png file
    Image.fromarray(image).save(disk_dir / f"{image_id}.png")
