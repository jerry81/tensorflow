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

# predictions

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()]) # take model worked on before and feed to new set of layers
predictions = probability_model.predict(test_images) # take a list of image input models
print('predictions for first are ', predictions[0])

predictedClass = np.argmax(predictions[0]) # gets the maximum of the array

print('predicted class for first item is ', predictedClass)

# visualize fubnctions

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()