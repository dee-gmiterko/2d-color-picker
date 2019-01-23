import numpy as np
import pylab as plt
import pandas as pd
from scipy import misc

def generatePalette(data, model):

    print("GENERATING PALETTE..")

    model = model.transpose()
    colors = data[:,0:3]

    SIZE = 128

    def dstack_product(arrays):
        return np.dstack(
            np.meshgrid(*arrays, indexing='ij')
            ).reshape(-1, len(arrays))

    df = pd.DataFrame(dstack_product([np.linspace(np.min(model[0]), np.max(model[0]), SIZE), np.linspace(np.min(model[1]), np.max(model[1]), SIZE)]), columns=['x', 'y'])

    print("Computing distances..")
    distances = df.apply(lambda point: pd.Series( np.sqrt( (model[0]-point['x'])*(model[0]-point['x']) + (model[1]-point['y'])*(model[1]-point['y']) ) ), axis=1)

    print("Calculating weights..")
    weights = distances.apply(lambda distances: np.exp(-distances)/np.sum(np.exp(-distances)), axis=1)

    print("Generating colors..")
    colorScheme = weights.apply(lambda weights: pd.Series([np.sum(weights * colors[:,0]), np.sum(weights * colors[:,1]), np.sum(weights * colors[:,2])], index=['r', 'g', 'b']), axis=1)

    misc.imsave("output/colorScheme.png", np.reshape(colorScheme.values, (SIZE, SIZE, 3)))

def modelView(data, model):
    model = model.transpose()
    colors = data[:,0:3]

    plt.scatter(model[0], model[1], color=colors)
    plt.show()
