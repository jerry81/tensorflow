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

supervised/unsupervised learning
  supervised - inputs and outputs given
  unsupervised - only inputs given

MSE - mean squared error - the squared difference between expected and predicted
compare with linear difference

backfed input cell - feedback deliberately introduced

noisy input cell - chaotic, thermal, quantum noise introduced

hidden cell - attenuates (applies weight to) input from activation function outputs in prior layer 

probablistic HC - applies radial basis function to diff between test case and cell's mean 

spiking HC - employs spiking model of activation vs perceptron model - more accurate simulation of bio neurons

match input output cell - uses "auto-encoders" to converge on identify function from network input to output

recurrent cell - applies historical signal state to improve convergence where time domain is important dimension of learning
example of time domain sensitive - video, audio
applied in RNN

memory cell - applies historic signal state to improve convergence with a time domain
applied in LTSM

different MC - applies historical signal state
applied in GRU

convolution cell - passes result of convolution kernel applied to previous layer's output

pool cell - agregates results of cell sin convolution layer 

kernel - filter that extracts features from image
also most basic level or core of OS

stochiastic network - random

contrastive divergence - running markov chain on sample, starting with last example processed

gradient - slope prepresenting rel between weights and error

numpy - package for scientific computing with python
numpy() - converts tensor obj to numpy.ndarray object
numpy.ndarray - array object

logit - output of model, before being converted to probabilities with a softmax

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

FF, FFNN, P - feed forward neural networks, perceptrons 
  straightforward - feed info from front to back
  layer has either input, hidden, or output cells.
  layer does not have connection
  two layers fully connected 
  simplest: two input cells and one output cell (logic gate)
  trained with back-propagation - (given paired dataset of input and expected output) - error is back-propagated
  problem: loses information in depth

RBF - radial basis function - FFNN with radial basis functions
  assigns real value to each input from domain - output is always an absolute value.  

RNN - recurrent NN - FNN with time twist and states
  a neuron may feed itself
  info of past stored in weights
  problem: vanishing/exploding gradient - information rapidly lost ovver time
  good for autocompletion

LTSM - combats vansihing/exploding gradient issue
  introduces gates and memory cells
  three gates: input/output/forget - gates may stop or allow flow of informaation - forget gate may forget some historical data
  inspired by circuitry 
  not as much biology
  good for processing books or composing music

GRU - gated recurrent units 
  one less gaate than LSTM.  has update gate - termines how much info to keep from last state and how much to let in from previous layer
  has reset gate - like forget gate
  faster than LTSM

BiRNN/LTSM/GRU - connected to both past and future

AE - autoencoder - similar to FFNN
  encode info (compress) automatically 
  hidden layers smaller than input and output layers
  input - encode - attenuate - decode - output

VAE - variational AE - taught approximated probability distribution of input
  uses Probabilistic HC instead of normal HC

DAE - denoising AE - feed input data with noise (like grainy image)
  encourages network to learn with broader features

SAE - sparse AE - opposite of AE - 
  encodes info to more space
  attempts to create match input output cells
  filters out errors above a certain threshold for further training
  resembles spiking

MC - Markov Chain
  predecessor to BM and HN 
  memory-less
  from any node, what are odds of going to any neighboring node?
  looks like a symmetrical web, every probablistic HN connected to every other.
  
HN - hopfield network - every neuron connected to every other 
  every node functions as everything - input (before training), hidden during training, output afterwards
  will reduce and converge into a stable conventional nn (but not always)
  entropy or temperature gradually reduced
  updating nodes done synchronously or one by one
  stable when all cells update and none change 

BM - boltzmann machine - similar to HN
  but not all cells start as input, some marked hidden
  begin with random weights, learns thru back propagation and in modern times, contrastive divergence ( using markov chain to determine gradients )
  stabalizes when global temperature reaches equillibrium

RBM - restricted BM
  more usable than BM - not every neuron is connected to every other.  only connects different group of neurons to every other group 

DBN - deep belief network - aka greedy training - uses locally optimal solutions to approximate answers
  input then large web of hidden nodes then output
  stacked architecture - trainable stack by stack vs. just the previous network
  
D/CNN - deep convolutional NN - used for image processing 
  input then network of converging convolutions, followed by pool cells, then large network of hidden nodes then outputs
  FFNN glued to end to further process data.

DN - deconvolutional network - opposite of DCNN aka IGN - inverse graphics network - 
  produces graphical images based on text input.

DCIGN - deep convolutional inverse graphics network - 
  like VAE but CNN and DNN for encoder aand decoders
  trained with back propagation

  ### TF beginner tutorial
  1.  in order to download mnist.npz, had to install python certificaates
    a.  go to Applications/Python
    b.  double click install certificates command

### tf "learn ml"

TF make building neural networks for ML easy 
Uses API called Keras - creaged by Francois Chollet
Deep learning with python first step

### datasets

MNIST - first dataset researchers try.  if it doesn't work on MNIST, it won't work on others.
issues: 
  MNIST too easy 
  overused
  cannot represent modern CV tasks

Fashion MNIST - replacement for MNIST - 9 classes x 3 rows per class 
### ML Basics with Keras tutoriaal 

train_images and train_labels - training set - data model uses to learn 
test_images, test_labels - test set 
train_images, test_images - 28x28 numpy arrays - pixel values 0 to 255 
x_labels - array of intergers ranging from 0 to 9. corresponds to 9 classes of clothing
0 - t-shirt
1 - trouser 
etc...
9 - ankle boot

build model 

1.  configure layers (model = ...)
  a.  basic building block of nn
  b.  extract representation of data fed in
  c.  chained together, example - tf.keras.layers.Dense 
2.  compile model (model.compile)
  a.  needs a loss function - how accurate model is during training <-- minimize this 
  b.  optimizer <-- how model updates itself based on data and loss function 
  c.  metrics - monitor training and testing % correctly classified images
3.  train model (model.fit)
  a.  feed in training data 
  b.  model created
  c.  test created model against test data (test_images/test_labels) are all correct
  d.  test_image fed into model = output and check output vs test_label (model.evaluate)
4.  make prediction - (tf.keras.layers.Softmax())
  a.  this is an application of the model
  b.  softmax creates the probability scores from logits
    i.  short for soft argmax - soft refers to smooth approximation
    ii.  argmax finds arguments that maximizes value
  c.  model.predict actually gets the predictions - outputs an array of len matching the number of classes, the values represent the "confidence" that the input is of the class referred to by the index