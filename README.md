# disease_detector
Code to detect diseases in plants from photos of their leaves

This project was inspired by https://machinelearningprojects.net/leaf-disease-detection-flask-app/
however I wrote the code from scratch without looking at Abishek's solution

The data set of leaf images came from https://www.kaggle.com/datasets/dev523/leaf-disease-detection-dataset

I figured out how to import the images and translate them into numpy arrays using the tutorial at https://realpython.com/storing-images-in-python/
The code I wrote and my comments are in the folder image_tutorial

To learn how to build a CNN I followed the tutorial at https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-2 while watching the Stanford lecture series CS231n at https://youtube.com/playlist?list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk
The code I wrote and my notes on the mathematical basis for the code based on the Stanford lectures are in the folder digit_tutorial

All images in a folder are imported converted into numpy arrays and stored in HDF5 files
The metadata is captured into dictionaries based on the folder structure (ie - the folder tree is structured based on the type of plant and disease) and is stored in json files

A model is then created to identify the tree type from the photos, and then for each tree type a model is built to identify the disease type. The model parameters are saved in HDF5 files

Test images can be loaded and first the plant type is identified, then the disease type (if any) is identified by applying the appropriate model
