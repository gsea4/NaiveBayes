import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as smp
from PIL import Image

training_images_file = open('train-images.idx3-ubyte','rb')
training_images = training_images_file.read()
training_images_file.close()

training_images = bytearray(training_images)
training_images = training_images[16:]

training_images = np.array(training_images).reshape(60000, 784)

img = Image.fromarray(training_images[5].reshape((28,28)))
img.show()