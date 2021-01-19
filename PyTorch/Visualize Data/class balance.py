images = [] # List of images
labels = [] # List of labels

no_classes = 3 # Number of classes

import numpy as np
un = np.unique(np.array(labels))

class_bal = []
for u in un:
  class_bal.append((np.array(labels)==u).sum())

class_bal.sort()

import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.bar(range(0,no_classes),class_bal, label="Number of images")
plt.ylabel('no of images')
plt.show()
