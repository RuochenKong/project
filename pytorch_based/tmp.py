import numpy as	np

label = np.load('../data/labels.npy')

print(len(label[label == 0]))