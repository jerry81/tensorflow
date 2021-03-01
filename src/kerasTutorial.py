import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
print(tf.__version__)
# np.set_printoptions(threshold=np.inf)
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# make a lookup table 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('shape', train_images.shape) # 60k  data points, 28 x 28 
print('trainLabels len', len(train_labels))

# test set has 10k images 
# train set 60k 

#plt acts as our visualizer

plt.figure()
plt.imshow(train_images[0]) 
plt.colorbar() 
plt.grid(False)
plt.show()