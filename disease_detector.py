# these modules are for importing and accessing the image data

import numpy as np
from PIL import Image
from pathlib import Path
import glob
import json
import itertools

# these modules are for building and interrogating the model

import pandas as pd
import keras
from keras import layers
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt
import seaborn as sns 

# these are modules internal to the project

import Image_handler
import Modelling

def import_source_images(my_path, my_h5):
    # Open all the image files in a path and its subfolders and place the image arrays and associate labels in numpy arrays
    path = my_path
    images, labels = Image_handler.read_many_disk(glob.iglob(path,recursive=True))
    labels_np = np.array([i for i in labels.keys()])

    # Create dictionaries to translate the text metadata (test/train, tree names, disease names) and populate them
    
    data_set_d = {}
    tree_d = {}
    disease_d = {}

    for item in labels.items():
        if item[1][0] not in data_set_d:
            i = len(data_set_d)+1
            data_set_d[item[1][0]] = i
        if item[1][1] not in tree_d:
            i = len(tree_d) + 1
            tree_d[item[1][1]] = i
        if item[1][2] not in disease_d:
            i = len(disease_d) + 1
            disease_d[item[1][2]] = i
        item[1][0] = data_set_d[item[1][0]]
        
        item[1][1] = tree_d[item[1][1]]
        item[1][2] = disease_d[item[1][2]]

    # Save the dictionaries to json files for later reference
    
    with open("label_dict.json", "w") as json_file:
        json.dump(labels, json_file)
    with open("data_set.json", "w") as json_file:
        json.dump(data_set_d,json_file)
    with open("tree_d.json", "w") as json_file:
        json.dump(tree_d,json_file)
    with open("disease_d.json", "w") as json_file:
        json.dump(disease_d,json_file)
    
    # Store the image and label data held in the numpy arrays in a hft5 file for later reference
    
    Image_handler.store_many_hdf5(images,labels_np,f"{my_h5}.h5")

# Let's create a 1000 entry subset of the data set to make it easier to work with

# NB - because we created the subset from the processed labels dictionary the metadata dictionaries don't have text any more
# but for this test that's not important
# I can always enter it manually if necessary

def sub_set_builder(images,labels,my_h5,fraction=90,model_suff="_sub"):

    # Create a subset of the data to make it easier to work on the code
    # In fact there's no way my PC can process 90k image data set so this will have to do :)
    # Fraction is the size of the subset relative to the original data-set - the default 90
    # gives a 1,000 = 90,000 / 90 set of images
    
    # create numpy arrays of the passed full set of images and labels    
    images_sub = images[::fraction]
    labels_sub = labels[::fraction]

    # Store the subset of images and labels as hfd5 files
            
    Image_handler.store_many_hdf5(images_sub,labels_sub,f"{my_h5}{model_suff}.h5")

    # And create dictionary files with the reduced set of tree and disease names
    
    # first the labels dictionary
    with open(f"label_dict.json","r") as json_file:
        label_dict = json.load(json_file)
    i = 0
    label_dict_sub = {}
    for item in itertools.islice(label_dict.items(),0,len(label_dict),fraction):
        label_dict_sub[i] = item[1]
        i += 1

    with open(f"label_dict{model_suff}.json","w") as json_file:
        json.dump(label_dict_sub,json_file)

    # and now create the test/train, tree and disease dictionaries from the subset of labels
    data_set_d = {}
    tree_d = {}
    disease_d = {}

    for item in label_dict_sub.items():
        if item[1][0] not in data_set_d:
            i = len(data_set_d)+1
            data_set_d[item[1][0]] = i
        if item[1][1] not in tree_d:
            i = len(tree_d) + 1
            tree_d[item[1][1]] = i
        if item[1][2] not in disease_d:
            i = len(disease_d) + 1
            disease_d[item[1][2]] = i
        item[1][0] = data_set_d[item[1][0]]
        item[1][1] = tree_d[item[1][1]]
        item[1][2] = disease_d[item[1][2]]

    with open(f"data_set{model_suff}.json", "w") as json_file:
        json.dump(data_set_d,json_file)
    with open(f"tree_d{model_suff}.json", "w") as json_file:
        json.dump(tree_d,json_file)
    with open(f"disease_d{model_suff}.json", "w") as json_file:
        json.dump(disease_d,json_file)
        
def build_tree_model(images, Y_Tree, model_suff=""):
    
    print("Building tree model")
    
    # We need a model where the X values are the image arrays, and the y values are 
    # the tree types. We need to import the tree dictionaries and the label dictionary
    
    with open(f"label_dict{model_suff}.json", "r") as json_file:
        label_dict = json.load(json_file)
    
    # Split the data into train and test sets
    # Train has an id of 2, test has an id of 1 in our dataset
    # I may change this to look this up in the dictionary later if I have time
    
    shuffled_ids = np.arange(len(images))
    np.random.shuffle(shuffled_ids)
    
    X_train = [images[i] for i in shuffled_ids if label_dict[str(i)][0]==2]
    X_test = [images[i] for i in shuffled_ids if label_dict[str(i)][0]==1]
    
    y_train = [Y_Tree[i] for i in shuffled_ids if label_dict[str(i)][0]==2]
    y_test = [Y_Tree[i] for i in shuffled_ids if label_dict[str(i)][0]==1]

    # Now build the model
    
    my_model = digit_modeller(X_train, X_test, y_train, y_test, 100,10)

    return my_model

def build_disease_model(images, Y_disease, tree_id, model_suff=""):
    
    print(f"Building disease model for tree number {tree_id}")
    
    # This builds a model to identify the diseases specific to a type of tree
    
    # We need a model where the X values are the image arrays for a particular tree, 
    # and the y values are the diseases. We need to import the label dictionary
    
    with open(f"label_dict{model_suff}.json", "r") as json_file:
        label_dict = json.load(json_file)

    # Train has an id of 2, test has an id of 1 in our dataset
    # I may change this to look this up in the dictionary later if I have time
    # The tree type is passed to the function
    
    shuffled_ids = np.arange(len(images))
    np.random.shuffle(shuffled_ids)
            
    X_train = [images[i] for i in shuffled_ids if (label_dict[str(i)][0]==2 and label_dict[str(i)][1]==tree_id)]
    X_test = [images[i] for i in shuffled_ids if (label_dict[str(i)][0]==1 and label_dict[str(i)][1]==tree_id)]

    y_train = [Y_disease[i] for i in shuffled_ids if (label_dict[str(i)][0]==2 and label_dict[str(i)][1]==tree_id)]
    y_test = [Y_disease[i] for i in shuffled_ids if (label_dict[str(i)][0]==1 and label_dict[str(i)][1]==tree_id)]

    # Now create the disease model
    
    my_model = digit_modeller(X_train, X_test, y_train, y_test, 100,10)
    return my_model
        
def digit_modeller(X_train, X_test, y_train, y_test, batch_size=32, epochs=10):
   
    model = Sequential()
    
    # the input layer will be as defined by the source data = a bunch of 256x256x3 arrays
    # The filter kernel is 3x3, and we will apply 32 filters. The stride is 1 by default
    # No padding - just take source image

    # This model uses two convolution layers, a pooling layer and a dropout layer
    # Pooling takes the max value from each 2x2 grid in the previous layer
    # Dropout randomly removes a proportion of neurons from layer on each iteration
    # It helps prevent overfitting by ignoring some neurons which initially have a 
    # very strong signal from the source data. It's a form of regularisation

    # This model uses the Relu activation function, which is considered a good place to start

    model.add(Convolution2D(32,(3,3),activation='relu',input_shape=(256,256,3)))
    print("1st layer",model.output.shape)
    model.add(Convolution2D(32,(3,3),activation='relu'))
    print("2nd layer",model.output.shape)
    model.add(MaxPooling2D(pool_size=(2,2)))
    print("Pooling layer",model.output.shape)
    model.add(Dropout(0.25))
    print("Dropout layer",model.output.shape)
    model.add(Flatten())
    print("Flattened layer",model.output.shape)
    model.add(Dense(128,activation='relu'))
    print("Dense x1 layer",model.output.shape)
    model.add(Dropout(0.5))
    model.add(Dense(np.shape(y_train)[1],activation='softmax'))
    print("Dense x2 layer",model.output.shape)

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    print("Model compiled")
   
    history = model.fit(np.array(X_train),np.array(y_train), batch_size=batch_size, epochs=epochs, verbose=1)
    
    print(history.history['loss'])
    
    score = model.evaluate(np.array(X_test), np.array(y_test), verbose=0)
    
    
    print(f"Loss: {score[0]}, Accuracy: {score[1]}")
    
    return model
    
 # This is the code to bild the tree identification model and the various disease identification models   
def build_models(images_h5, model_h5_suff=""):
    
    # First load the image Numpy arrays from the hdf5 where they have been saved
    
    images, _ = Image_handler.read_hdf5(f"{images_h5}{model_h5_suff}.h5")
    print("Images loaded from hd5")

    # Zero center and normalise to the [0,1] range
    # Save the zero mean transform to pass to the models later - we'll need this to apply to the validation data
    
    images = np.array(images,dtype='float32')
    zero_mean_transform = np.mean(images,axis=0,dtype='float32').reshape(1,256,256,3)
    images -= zero_mean_transform
    images /= 255

    zero_mean_transform.tofile(f"zero_mean_transform{model_h5_suff}.csv",sep=",")
        
    print("Image data pre-processed")
        
    # We're going to use categorical crossentropy as the loss function, so the y data needs to be transformed
    # into a n*m matrix, where n is the number of images and m is the number of trees or diseases

    with open(f"label_dict{model_h5_suff}.json", "r") as json_file:
        label_dict = json.load(json_file)
    y_tree = [label_dict[str(i)][1] for i in range(len(images))]
    y_disease = [label_dict[str(i)][2] for i in range(len(images))]
    # Maybe change these to integer later?
    y_tree = np.array(np_utils.to_categorical(y_tree).astype('uint8'))
    y_disease = np.array(np_utils.to_categorical(y_disease).astype('uint8'))

    print("Result matrices built")
    # build_tree_model(images, y_tree, model_h5_suff).save(f"tree_model{model_h5_suff}.h5")

    # # Building many models in a loop like this generates a tensorflow warning because
    #     # WARNING:tensorflow:5 out of the last 11 calls to <function 
    #     # Model.make_train_function.<locals>.train_function at 0x0000012D18AA5BC0> triggered tf.function retracing. 
    #     # Tracing is expensive
    # # Some dude on Stackoverflow tells me I can ignore this as I'm consciously building multiple models
    # #   https://stackoverflow.com/questions/75843622/r-tensorflow-warning-when-building-model-inside-a-loop-please-define-your-tf#:~:text=For%20%281%29%2C%20please%20define%20your%20%40tf.function%20outside%20of,refer%20to%20https%3A%2F%2Fwww.tensorflow.org%2Fguide%2Ffunction%23controlling_retracing%20and%20https%3A%2F%2Fwww.tensorflow.org%2Fapi_docs%2Fpython%2Ftf%2Ffunction%20for%20more%20details.%22
    
    for i in range(1,np.shape(y_tree)[1]):
        build_disease_model(images,y_disease,i,model_h5_suff).save(f"disease_model_{i}{model_h5_suff}.h5")

# # And now test whether we can identify a tree and it's disease
def test_all(test_path,model_suff=""):
    
    # open the tree model    
    
    tree_model = load_model(f'tree_model{model_suff}.h5')
    
    # And the tree and disease dictionaries so we can translate the IDs from the labels into english
    
    with open(f"disease_d.json") as json_file:
        disease_dict = json.load(json_file)
    with open(f"tree_d.json") as json_file:
        tree_dict = json.load(json_file)

    # Import the zero-transform applied to the training data
    zero_mean_transform = np.genfromtxt(f"zero_mean_transform{model_suff}.csv", delimiter=',',dtype='float32').reshape(1,256,256,3)
    
    for i, image in enumerate(glob.iglob(test_path,recursive=True)):
        # SHould probably put a test in here to check whether the file is a 256x256 jpg
        test_img = np.array(Image.open(image),dtype='float32').reshape(1,256,256,3)
        test_img -= zero_mean_transform
        test_img /= 255
        predicted_tree = tree_model.predict(test_img,verbose=0)
        tree_id = np.argmax(predicted_tree)
        tree = [tree for tree, id in tree_dict.items() if id == tree_id]
        disease_model = load_model(f"disease_model_{tree_id}{model_suff}.h5")
        disease_id = disease_model.predict(test_img,verbose=0)
        disease_id = np.argmax(disease_id)
        disease = [disease for disease, id in disease_dict.items() if id == disease_id]
        print(f"The image {image} is of a {tree} which has {disease}")


def test_single_image(image_path,model_suff):
    tree_model = load_model(f'tree_model{model_suff}.h5')
    test_img = Image_handler.read_single_disk(image_path).reshape(1,256,256,3)
    predicted_tree = tree_model.predict(test_img)
    tree_id = np.argmax(predicted_tree)
    print(f"{image_path}  returns tree id {tree_id} from tree_model{model_suff}")
    disease_model = load_model(f"disease_model_{tree_id}{model_suff}.h5")
    predicted_disease = disease_model.predict(test_img)
    disease_id = np.argmax(predicted_disease)
    print(f"{image_path}  returns disease id {disease_id} from disease_model_{tree_id}{model_suff}")
    
  
        
######## THIS IS THE TESTING CODE ######################
train_and_test_path = "./Data/dataset/**/*.jpg"
image_h5 = "leaf_image_data"
validate_path = "Data/Images_for_test/*.jpg"
# print("Importing source images")
# import_source_images(train_and_test_path,image_h5)
# print("Reading source images")
# images, labels = Image_handler.read_hdf5(f"{image_h5}.h5")
# print("Building a subset of ~10k images")
# sub_set_builder(images,labels,image_h5,45)
# print("Modelling the subset of images")
# build_models(image_h5,"_sub")
print("Validating the models")
test_all(validate_path,"_sub")
# image_path = "Data/images_for_test/AppleCedarRust1.JPG"
# test_single_image(image_path)
