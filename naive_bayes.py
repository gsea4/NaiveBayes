import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as smp
import math
from time import time
from PIL import Image

training_images_file = open('train-images.idx3-ubyte','rb')
training_images = training_images_file.read()
training_images_file.close()

training_labels_file = open('train-labels.idx1-ubyte', 'rb')
training_labels = training_labels_file.read()
training_labels_file.close()

training_images = bytearray(training_images)
training_images = training_images[16:]

training_labels = bytearray(training_labels)
training_labels = training_labels[8:]

training_images = np.array(training_images).reshape(60000, 784)
training_labels = np.array(training_labels)

# img = Image.fromarray(training_images[5].reshape((28,28)))
# img.show()

training_images[training_images < 50] = 0
training_images[training_images >= 50] = 1

testing_images_file = open('t10k-images.idx3-ubyte', 'rb')
testing_images = testing_images_file.read()
testing_images_file.close()

testing_labels_file = open('t10k-labels.idx1-ubyte', 'rb')
testing_labels = testing_labels_file.read()
testing_labels_file.close()

testing_images = bytearray(testing_images)
testing_images = testing_images[16:]

testing_labels = bytearray(testing_labels)
testing_labels = testing_labels[8:]

testing_images = np.array(testing_images).reshape(10000, 784)
testing_labels = np.array(testing_labels)

testing_images[testing_images < 50] = 0
testing_images[testing_images >= 50] = 1

t0 = time()
unique_elem, counts = np.unique(training_labels, return_counts = True)
priors = np.append(unique_elem.reshape(10,1), counts.reshape(10,1), 1)

likelihoods = np.zeros((10, 784), dtype='int16')

for row in range(training_images.shape[0]):
    mask = training_images[row] > 0
    likelihoods[training_labels[row], mask] += 1

print("training time: " + str(round(time() - t0, 3)) + "s")

def classify(image, priors, likelihoods):
    max_pros_class = (-1E6, -1)
    for c in range(10):
        log_prior = math.log(priors[c][1])
        pros = log_prior
        n = float(np.sum(likelihoods[c]))
        for pixel in range(28 * 28):
            if image[pixel] == 1:
                if likelihoods[c][pixel] == 0:
                    likelihoods[c][pixel] = 1
                log_likelihood = math.log(likelihoods[c][pixel]/n)
                pros = pros + log_likelihood
        
        if pros > max_pros_class[0]:
            max_pros_class = (pros, c)

    return max_pros_class

image = 0
t1 = time()
classify(testing_images[image], priors, likelihoods)[1]
print("prediction time for 1 image: " + str(round(time() - t1, 3)) + "s")
print("predicted label: " + str(classify(testing_images[image], priors, likelihoods)[1]) + " | correct label: " + str(testing_labels[image]))
correct = 0

t2 = time()
for image in range(10000):
    if classify(testing_images[image], priors, likelihoods)[1] == testing_labels[image]:
        correct += 1

print("prediction time for 10000 images: " + str(round(time() - t2, 3)) + "s")
print("accuracy: " + str(correct/10000.))
