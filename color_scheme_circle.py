import os
import numpy as np
from scipy import misc
import pylab as plt
from math import sin, cos, sqrt

K = 5

data = np.load("data.npy")
model = np.load("model.npy")

model = np.transpose(model)

model_center = np.array([np.min(model[0]), np.mean(model[1])])

unused = list(range(model.shape[1]))
circleModel = []
circleColor = []

i=0
y=0
modelLastLayer = []
while i < len(model[0]):
    print(y, i)
    c = 6*(y+1)
    modelThisLayer = []
    for x in range(c):
        xr = (x / c) * 3.14 * 2

        if len(modelLastLayer) > 0:
            modelLastLayer.sort(key=lambda l: abs(l[0] - xr))
            closestInLastLayer = np.sum(np.array(modelLastLayer[min(len(modelLastLayer),K):]), 0) / K
            closestInLastLayer = closestInLastLayer[1:]
        else:
            closestInLastLayer = model_center

        closest = None
        closestDistance = None
        for j in unused:
            d = sqrt((model[0][j] - closestInLastLayer[0])**2 + (model[1][j] - closestInLastLayer[1])**2)
            if closestDistance is None or d < closestDistance:
                closest = j
                closestDistance = d

        unused.remove(closest)
        circleModel.append([cos(xr)*y, sin(xr)*y])
        modelThisLayer.append([xr, model[0][closest], model[1][closest]])
        circleColor.append(data[closest])
        i += 1

        if i == len(model[0]):
            break
    modelLastLayer = modelThisLayer
    y += 1

circleModel = np.transpose(circleModel)

plt.scatter(circleModel[0], circleModel[1], color=circleColor)
plt.show()
