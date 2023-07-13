import numpy as np
import pickle
from PIL import Image
import csv
from pathlib import Path
import lmdb
import h5py
from timeit import timeit

# This was written following the tutorial at https://realpython.com/storing-images-in-python/
# The intent is to learn different ways to store image data on disk
# and understand the plusses and minuses of each wrt disk space used,
# speed to write to disk, and speed to read into memory

# I needed to do this for my project to identify diseases by examining
# photos of leaves

data_dir = Path("data/images_for_tutorial/cifar-10-python/")

# The source data for the tutorial is the famous CIFAR-10
# https://www.cs.toronto.edu/~kriz/cifar.html
# which is familiar from the Stanford online lecture series on
# convulational neural networks. The data is stored in batch files
# which need to be "unpickled"

# pickle converts objects into bitewise data which is a much more efficient
# way of storing it
def unpickle(file):
    with open(file,"rb") as fo:
        my_dict = pickle.load(fo, encoding= "bytes")
    return my_dict

images, labels = [],[]
for batch in data_dir.glob("data_batch_*"):
    batch_data = unpickle(batch)
    # The images in the CIFAR-10 data set are 32x32 color images
    # But they're stored in the batch files as single 3072 (= 32*32*3)
    # entry arrays, so we need to deflatten them to get them into 
    # a standard 32x32 RGB array structure
    
    # b in front of a string returns a sequence of octet bytes (ie - integers between 0 and 255)
    for i, flat_im in enumerate(batch_data[b"data"]):
        # This deflattens the 3072x1 array into a 32x32x3 array
        im_channels = []
        for j in range(3):
            im_channels.append(
                flat_im[j * 1024 : (j+1) * 1024].reshape((32,32))
            )
        # And this appends it to the array of arrays which will 
        # eventually hold all the 50k images and the array which
        # will hold the respective labels
        images.append(np.dstack((im_channels)))
        labels.append(batch_data[b"labels"][i])
        
print("Loaded CIFR-10 training set:")
print(f" - np.shape(images)   {np.shape(images)}")
print(f" - np.shape(labels)   {np.shape(labels)}")

# Create folders to hold the outputs of the different methods 
# of storing image data on disk
disk_dir = Path("data/disk/")
lmdb_dir = Path("data/lmdb/")
hdf5_dir = Path("data/hdf5")

disk_dir.mkdir(parents=True, exist_ok=True)
lmdb_dir.mkdir(parents=True, exist_ok=True)
hdf5_dir.mkdir(parents=True,exist_ok=True)

# We start with the most basic method - individual image files
# which can be viewed by the user

def store_single_disk(image, image_id, label):
    """ Stores a single image as a .png file on disk.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    # We get image as a 32x32x3 array, and save it to a png file
    Image.fromarray(image).save(disk_dir / f"{image_id}.png")
    # and we create a .csv file to store the metadata
    with open(disk_dir / f"{image_id}.csv", "wt") as csvfile:
        writer = csv.writer(csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        writer.writerow([label])
        
# Next LMDB - or "Lightning Memory-Mapped Database"
# This is a key-value store - not a relational database
# This method returns direct pointers to memory addresses 
# of both keys and values, without needing to copy anything in 
# memory like most database do, which is what makes it fast

# Both keys and values are expected to be strings so you typically
# serialise the values when converting to strings
# Pickle is the most common way to serialize

class CIFAR_image:
    def __init__(self, image, label):
        # this dataset has all images the same size, but many datasets won't
        # therefore it's good practice to define the dimensions of the 
        # image before reconstruction
        self.channels = image.shape[2]
        self.size = image.shape[:2]
        self.image = image.tobytes()
        self.label = label
        
    def get_image(self):
        """ Returns the image as a numpy array. """
        image = np.frombuffer(self.image, dtype=np.uint8)
        # the * before the parameter self unrolls a list into a set of arguments
        return image.reshape(*self.size, self.channels)
        
# new LMDB databases need to know how much memory they're expected to use
# and this is stored in the variable map_size
# Data is written and read from LMDB using transactions

def store_single_lmdb(image, image_id, label):
    """ Stores a single image to a LMDB.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    map_size = image.nbytes * 10
    
    env = lmdb.open(str(lmdb_dir / f"single_lmdb"), map_size = map_size)
    
    with env.begin(write = True) as txn:
        value = CIFAR_image(image,label)
        key = f"{image_id:08}"
        txn.put(key.encode("ascii"),pickle.dumps(value))
    env.close()
    
# Finally HDF5 - or Hierarchical Data Format.
# HDF files consist of Datasets and Groups
# Datasets are multidimensional arrays, and groups can be of 
# datasets or other groups
# You can store arrays of any size and shape
# but within a dataset the dimensions and type must be uniform
# IE - each dataset must contain an homogenous N-dimensional array
# Luckily NumPy ndarrays are homogenous n-dimensional objects

def store_single_hdf5(image, image_id, label):
    """ Stores a single image to an HDF5 file.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    # Create a new HDF5 file
    my_file = h5py.File(hdf5_dir / f"{image_id}.h5", "w")
    
    # and create a dataset in the file
    
    dataset = my_file.create_dataset(
        "image", np.shape(image), h5py.h5t.STD_U8BE, data=image
    )   
    # h5py.h5t.STD_U8BE specifies the type of data that will be stored in the dataset, which in this case is unsigned 8-bit integers
    # see full list of datatypes http://api.h5py.org/h5t.html
    # the datatype chosen has a big impact on performance!
    meta_set = my_file.create_dataset(
        "meta", np.shape(label),h5py.h5t.STD_U8BE, data=label
    )
    my_file.close
    
# TESTING WRITING A SINGLE IMAGE

# Storing the function names for the 3 methods in a dictionary gives
# an easy way to iterate through them

# _store_single_funcs = dict(disk=store_single_disk, lmdb=store_single_lmdb, hdf5=store_single_hdf5)

# store_single_timings = dict()

# for method in ("disk", "lmdb", "hdf5"):
#     t = timeit(
#         "_store_single_funcs[method](image, 0, label)",
#         setup="image=images[0]; label=labels[0]",
#         number=1,
#         globals=globals()
#     )
#     store_single_timings[method] = t
#     print(f"Method: {method}, Time usage: {t}")
    
# TESTING WRITING MULTIPLE IMAGES

def store_many_disk(images, labels):
    """ Stores an array of images to disk
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)
    
    for i, image in enumerate(images):
        Image.fromarray(image).save(disk_dir / f"{i}.png")
    
    with open(disk_dir / f"{num_images}.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        for label in labels:
            writer.writerow([label])
            
def store_many_lmdb(images,labels):
    """ Stores an array of images to LMDB.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)
    
    map_size = num_images * images[0].nbytes * 10
    
    # Create a new lmdb for all images
    env = lmdb.open(str(lmdb_dir / f"{num_images}_lmdb"),map_size=map_size)
    
    # we can write all the images in a single transaction
    
    with env.begin(write=True) as txn:
        for i in range(num_images):
            value = CIFAR_image(images[i],labels[i])
            key = f"{i:08}"
            txn.put(key.encode("ascii"),pickle.dumps(value))
    env.close()
    
def store_many_hdf5(images, labels):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)
    
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "w")
    
    dataset = file.create_dataset("images", np.shape(images), h5py.h5t.STD_U8BE, data=images)
    meta_set = file.create_dataset("meta", np.shape(labels),h5py.h5t.STD_U8BE, data=labels)
    
    file.close()
    
# To run the experiment for many images we need to do a bit of preparation

# First lets define the cut offs for the iterations

cutoffs = [10, 100000]

# We'd like to do 100k images, which means we need to double our image set

images = np.concatenate((images,images), axis=0)
labels = np.concatenate((labels,labels), axis=0)

#Let's check this worked

print(np.shape(images))    
print(np.shape(labels))

# And now let's run the experiment

# _store_many_funcs = dict(
#     disk=store_many_disk, lmdb=store_many_lmdb, hdf5=store_many_hdf5
# )

# store_many_timings = {"disk": [], "lmdb": [], "hdf5": []}

# for cutoff in cutoffs:
#     for method in ("disk", "lmdb", "hdf5"):
#         t = timeit(
#             "_store_many_funcs[method](images_,labels_)",
#             setup="images_=images[:cutoff]; labels_=labels[:cutoff]",
#             number=1,
#             globals=globals(),
#         )
#         store_many_timings[method].append(t)
        
#         print(f"Method: {method}, Time usage: {t}")
        
# BUT WHAT ABOUT READING IMAGES???

def read_single_disk(image_id):
    """ Stores a single image to disk.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    image = np.array(Image.open(disk_dir / f"{image_id}.png"))
    
    with open(disk_dir / f"{image_id}.csv", "r") as csvfile:
        reader = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )    
        label = int(next(reader)[0])
        
    return image, label

def read_single_lmdb(image_id):
    """ Stores a single image to LMDB.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    
    # readonly=True means no writes will be allowed to the lmdb until we're finished
    # effectively it's like a read lock on a database
    
    env = lmdb.open(str(lmdb_dir / f"single_lmdb"), readonly=True)
    
    with env.begin() as txn:
        # encode the key the same way as it was stored
        data = txn.get(f"{image_id:08}".encode("ascii"))
        # Don't forget it was loaded as a CIFAR_Image object
        cifar_image = pickle.loads(data)
        # And now retrieve the relevant bits
        # this is why we created the get_image method in the CIFAR_object class
        image = cifar_image.get_image()
        label = cifar_image.label
    env.close()
    
    return image, label

def read_single_hdf5(image_id):
    """ Stores a single image to HDF5.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "r+")
    
    image = np.array(file["/image"]).astype("uint8")
    label = int(np.array(file["/meta"]).astype("uint8"))
    
    return image, label

# _read_single_funcs = dict(
#     disk=read_single_disk, lmdb=read_single_lmdb, hdf5=read_single_hdf5
# )

# read_single_timings = dict()

# for method in ("disk", "lmdb", "hdf5"):
#     t = timeit(
#         "_read_single_funcs[method](0)",
#         setup="image=images[0]; label=labels[0]",
#         number=1,
#         globals=globals()
#     )
#     read_single_timings[method] = t
#     print(f"Method: {method}, Time usage: {t}")

def read_many_disk(num_images):
    """ Reads image from disk.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []
    
    for image_id in range(num_images):
        images.append(np.array(Image.open(disk_dir / f"{image_id}.png")))
        
    with open(disk_dir / f"{num_images}.csv", "r") as csvfile:
        reader = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for row in reader:
            labels.append(int(row[0]))
    
    return images, labels

def read_many_lmdb(num_images):
    """ Reads image from LMDB.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []
    
    env = lmdb.open(str(lmdb_dir / f"{num_images}_lmdb"), readonly=True)
    
    with env.begin() as txn:
        # Read all images in a single transaction
        # Can split into multiple transactions if necessary
        for image_id in range(num_images):
            data = txn.get(f"{image_id:08}".encode("ascii"))
            cifar_image = pickle.loads(data)
            images.append(cifar_image.get_image())
            labels.append(cifar_image.label)
    env.close()
    return images, labels

def read_many_hdf5(num_images):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [],[]
    
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "r+")
    
    images = np.array(file["/images"]).astype("uint8")
    labels = np.array(file["/meta"]).astype("uint8")
    
    return images, labels

_read_many_funcs = dict(
    disk=read_many_disk, lmdb=read_many_lmdb, hdf5=read_many_hdf5
)

read_many_timings = {"disk": [], "lmdb": [], "hdf5": []}

for cutoff in cutoffs:
    for method in ("disk", "lmdb", "hdf5"):
        t = timeit(
            "_read_many_funcs[method](num_images)",
            setup="num_images=cutoff",
            number=1,
            globals=globals(),
        )
        read_many_timings[method].append(t)

        # Print out the method, cutoff, and elapsed time
        print(f"Method: {method}, No. images: {cutoff}, Time usage: {t}")     