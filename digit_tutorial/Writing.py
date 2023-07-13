# This is the code I wrote following along to the tutorial at
# https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-2
# The commentary is mostly based on what I learnt in the online Stanford CS231n lectures
# on Convolutional Neural Networks for Image Identification
# https://youtube.com/playlist?list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt

# setting the seed ensures the calculation to produce the pseudo randon number 
# gives the same result every time
# If you do not set this seed the module uses the last random number returned 
# as the seed for the next one produced, so it will always be unpredictable
# Having the same number every time helps with debugging while the app is being developed 

np.random.seed(123)

# MNIST is a preloaded data set. Use inbuilt load data method to load train and test data

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(X_train.shape)
# print(X_train[0].shape)
# plt.imshow(X_train[0])
# plt.show()

# Convolutional networks need to define channels (ie - the depth of the source)
# but the MNIST sample images only have one channel (ie - they're 28 x 28)
# W need to explicitly declare they are one channel only by adding another dimension

X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)

#print(X_train.shape)

# Finally change the data type to FLoat32 and scale values to [0,1]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# The y data comes as a single 1x10 array, but we need to have 10 1x1 arrays (class values)
# NB - it's 10 because we're classifying digits, but it seems 10 categories is pretty standard for this stuff :)

y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)

# print(y_train.shape)

# now define the model architecture. This needs thought in a real project

# this model will use a sequential format

model = Sequential()
# the input layer will be as defined by the source data = a bunch of 28x28x1 arrays
# The filter kernel is 3x3, and we will apply 32 filters. The stride is 1 by default
# No padding - just take source image

# This model uses two convolution layers a pooling layer and a dropout layer
# Pooling takes the max value from each 2x2 grid in the previous layer
# Dropout randomly removes a proportion of neurons from layer on each iteration
# It helps prevent overfitting by ignoring some neurons which initially have a 
# very strong signal from the source data. It's a form of regularisation

# This model uses the Relu activation function, which is considered a good place to start. Main options, with pros and cons, are
#  Sigmoid - squashes numbers to range [0,1] and give nuce interpretation of saturating "firing rate" of neuron
#       - However have 3 problems. 1. Satruateed neurons 'kill' the gradients because dL/dx ~0 for large, small x
#       2. Outputs are not zero centered, which means that if inputs are always one sign, gradients will also always
#       be one sign (not necessarily the same sign :)) which means the back prop will always move in the same direction
#       resulting in a zig-zag path of optimisation which is inefficient 3. Sigmoid uses exp(), which is computationally
#       expensive DON'T USE THIS
#  Tanh(x) - squashes numbers to range [-1,1], so fixes non-zero centered issue of sigmoid, but still kills gradients
#       when saturated. CAN TRY THIS
#  ReLU (Rectified Linear Unit) - f(x) = max(0,x) - several advantages. 1. Does not saturate for x>0, 2. Computationally
#       efficient 3. Converges 6x for sigmoid, tanh 4. Closer approximation to experimentation results for neurons
#       However, still not zero-centered and saturated for x<0 => get dead ReLU for data in area which will never update
#       This may happen when you have bad initialisation. More common is when your update rate is too high. Can knock 
#       model out of kilter. BEST PLACE TO START
#       In practice people like to initialise ReLU with small positive biases (eg - 0.01)
#  Leaky ReLU - f(x) = max(0.01x,x) - 1. Doesn't saturate 2. Computationally efficent 3. Converges 6x vs sigmoid, tanh 
#       4. will not "die" like ReLU PLAY WITH THIS
#  Paramatric Rectifier - f(x) = max([alpha]x,x) - Leaky ReLU is this with [alpha] = 0.01
#  ELU (Exponential Linear Units) - for x>0 f(x) = x, else [alpha](exp(x)-1). Has ReLU advantages, but closer to zero mean
#       outputs and adds robustness to noise with negative saturation regime
#       HOWEVER more computationally heavy due to exp() PLAY WITH THIS
#  Maxout Neuron - max(w^T1x + b1, w^T2x +b2) - effectivel takes the max of the ReLU and Leaky ReLU
#       so it is a Linear Regime which does not saturate and does not die. However it doubles the number of parameters 
#       per neuron PLAY WITH THIS

model.add(Convolution2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
print("1st layer",model.output.shape)
model.add(Convolution2D(32,(3,3),activation='relu'))
print("2nd layer",model.output.shape)
model.add(MaxPooling2D(pool_size=(2,2)))
print("Pooling layer",model.output.shape)
model.add(Dropout(0.25))
print("Dropout layer",model.output.shape)
# print(model.output.shape)

# Finally add a fully connected layer and then output
# flatten transforms the input array into a 1d array - in this case a 12x12x32 into a 1x4,608 array
# The model is then condensed down into a 1x10 later, which corresponds the the 10 class values
model.add(Flatten())
print("Flattened layer",model.output.shape)
model.add(Dense(128,activation='relu'))
print("Dense x1 layer",model.output.shape)
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax')) #Softmax means multinomial logistic regression
                    # take model output by category, exponentize, normalise so sum across cats comes to 1, 
                    # then take negative log of the target category. CLoser to 0 the better
                    # Using this gives a model which tries to get as close to the right answer as possible
print("Dense x2 layer",model.output.shape)

# The model is compiled with a crossentropy loss function and uses ADAM for the optimiser
# The loss function is one which allows us to classify which models given better result fits
# - it quantifies our unhappiness with the scores across the system
# Adam is short for Adaptive Moment Estimation. It combines gradient descent with momentum
# and Route Mean Square Propogation 
# It works well on big problems with lots of data or parameters

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# The model is fit to the training data using a batch size of 32
# It does 10 full run throughs of optimisation (epochs)

model.fit(X_train,y_train, batch_size=32, epochs=10, verbose=1)
model.evaluate(X_test, y_test, verbose=0)

model.save("C:/TestPython/vscodepractice/sudoku/digitmodel.h5")