import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
print(tf.__version__)
# np.set_printoptions(threshold=np.inf)
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0 # scales rgb values to 0 to 1
# make a lookup table 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('shape', train_images.shape) # 60k  data points, 28 x 28 
print('trainLabels len', len(train_labels))

# test set has 10k images 
# train set 60k 

#plt acts as our visualizer

""" plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary) # takes in a 2d array
    plt.xlabel(class_names[train_labels[i]])
plt.show() """

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), # transforms to 1 dimensional array 28*28 = 784
    tf.keras.layers.Dense(128, activation='relu'), # densely connected, fully connected - 128 is number of nodes (neurons)
    tf.keras.layers.Dense(10) # logits array - length of 10 - matches the number of output classes
])

model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2) # output is array len 2 
print('\nTest accuracy:', test_acc)
