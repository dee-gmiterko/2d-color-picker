import numpy as np
import pylab as plt
import pandas as pd

data = np.load("data.npy")
model = np.load("model.npy")
model = model.transpose()
colors = data[:,0:3]

plt.scatter(model[0], model[1], color=colors)
plt.show()
