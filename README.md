## tensorflow

for machine learning

### definitions

easy_install - a python package management tool 

python virtual environment - isolates package installation from system 

jupyter - aka jupyter notebook - edit and run notebook documents
when running tensorflow in docker container we will start a jupyter server

Keras - python module 
machine learning framework - neural network components
greek: horn

Colaboratory aka colab - writes and executes python in browser 
access to GPUs - this may be key 

MNIST - modified national insitute of standards and technology dataset 
images of handwritten numbers

layers (in keras) - basic building blocks of neural networks
tensor-in tensor-out computation function ( call() 
has state held in tensorflow variables aka weights
instances are callable like functions

tensor - algebraic object - describes relationship between sets of algebraic objects related to vector space
map between vectors and scalars, or other tensors
multidimensional array 

activation function - transforms summed weighted input from node to activation of node or output for input

relu - rectified linear unit - peicewise linear function that outputs input directly if positive, or 0 
the default for many nn - easier to train 

dotproduct aka scalar product - two equal length vectors, sum the products

dense neural network - fully connected by neurons in network layer each neuron receives input from neurons in previous layer 

Yann Lecun - deep learning and machine learning expert

pb file - protobuf file - used in tensorflow - contains graph definition and weights of model - aka protocol buffers - like binary encoded json file
a way to serialized structured data 
language neutral
compiler provided with library to build objects in pb format

deep convolutional neural network - useful in image recognition
aka CNN aka ConvNet
an algorithm on which deep learning and computer vision have been built on
takes as input image, assigns weights and biases (aka importance) to objects/aspects in image, differentiate the identified objects, 
lower preprocessing requirement
inspired to connectivity of neurons in human brain esp. that in visual cortex
question: why not flatten the 3x3 matrix to 9x1 and feed to perceptron - for basic images, average precision score, low accuracy for complex images
convnet captures spatial and temporal dependencies by applying filters
convolved feature is a smaller matrix processed from larger image matrix
goal of convolution operation: extract high-level features (edges) from image
mimics the human brain recognizing wholesome high-level understanding of images
further reduce or expand the size of convolved feature by applying valid padding or same padding
pooling layer - reduces size of CF for performance
  max pooling - max value from portion of image - acts as noise supressant
  avg pooling - average of all values from portion of image


stride length - how much to move on the image when calculating next cell of convolved feature

perceptron - simple model of biological neuron in artificial neural network - 
also - name of early algorithm for binary classifiers



### install

options
1.  download package
2.  run docker container tensorflow/tensorflow:latest

0.  required
  a.  python 3.5-3.8
  b.  ubuntu 16.04
  c.  windows 7 or later/mac 10.12.6 or later 
  d.  pip 19 or higher
1.  install package with pip 

### sequential api 

for beginners
create models by plugging together building blocks

given a dataset, which has two tuples,
(x_train, y_train)
and
(x_test, y_test),
convert samples (x) to floating point by dividing x_train and x_test by 255.0

build model by stacking layers
call keras.layers.Flatten # flattens input - takes in data_format (string) as argument
.Dense x2 - "regular densely connected NN layer" implements operation - 
and
.Dropout

### misc

increasing pip3 timeout 
pip3 install -U --timeout 10000 tensorflow # increases timeout time to 10 seconds

using a pip3 mirror
pip3 install -U --timeout 10000 tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple # U flag is upgrade, -i is index-url 

### neural network

#### first steps

neural network - from [basics of neural networks](https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/)
starter:
is a price for a house given square footage and price good?
get 3 data points (example house area and price)
calculate average price per area 
this is first neural network 
input area 
function * avg price
output price 
average price is weight
result is output 
finding weights is training stage 
to improve weight calculating function,
take into account error (deviation)
error = actual - prediction (output)
function modified to 
input * weight + bias
still a linear function 
how to automate training algorithm, tweaking weight and bias until error is minimized?
gradient descent - algorithm that steps to correct weight and bias 

#### next step 

neural network may do regression - calculate and output continuous value (float)
extending the house example
add another variable, # of bathrooms (x2)
new model with two weights and a bias
W1x1 + W2x2 + b
y is always real world data 
could be an enum like Good/Bad
boolean like true/false
this is called classification type problems
discrete set of values for y 
model gives % chance that an input is of a value in the set of values 
softmax - calculates probability of each "class" (enum value)
  may take array of n items outputs same # of items
  the outputs add up to 1
  is not necessarily input/total of inputs
  output is actually exaggerated probability
adding classes (y) increases number of biases and weights
adding features (x) increases number of weights only

### supercharging android apps with tensorflow article

[taken from here](https://jalammar.github.io/Supercharging-android-apps-using-tensorflow/)

1.  tensorflow opensourced in 11/2015
2.  goog dominant force in ML
3.  ML played big part in search 
4.  TF originally designed for internal use 
5.  can run on mobile 
6.  TF repo on github has android directory
    a.  android demo app goes thru camera and tries to identify objects 
    b.  displays an array of strings describing object and the probability score next to it
    c.  to run, create apk, install camera2 package, adb install
    d.  bitmap files transformed into input tensors
    e.  tensors consist of 3-dimentional arrays that supplies rgb values of every pixel in image (x-index, y-index, 0 (r), 1 (g), 2 (b)) while value of this cell is RGB value 
    f.  model file downloaded to assets directory
    g.  54Mb pb file
    h.  txt file containing 1000 classifications (label strings)
7.  core built with C++
8.  TF software can be written in c++ or python
9.  c++ code is accessed thru Android NDK with JNI calls, methods declared with native keywords

### neural network zoo

[Neural network zoo](https://www.asimovinstitute.org/neural-network-zoo/)

<img src="https://www.asimovinstitute.org/wp-content/uploads/2019/04/NeuralNetworkZo19High.png"/>

attempts to keep track of neural net architectures 

