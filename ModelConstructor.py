import numpy as np
from sklearn.manifold import TSNE
from pathlib import Path
import pandas as pd

TSNE_ITERS = 500
TSNE_VERBOSE = False

def constructModel(data, modelIterations, magicSize):

    print("CONSTRUCTING MODEL..")

    data_emb = None

    for i in range(modelIterations):

        print("Iteration "+str(i)+"/"+str(modelIterations)+"..")

        # apply tsne
        if data_emb is None:
            data_emb = TSNE(n_components=2, perplexity=120, verbose=TSNE_VERBOSE, n_iter=TSNE_ITERS).fit_transform(data)
        else:
            tsneLearningRate = 200.0 * (float(modelIterations-i) / modelIterations)
            data_emb = TSNE(n_components=2, perplexity=120, verbose=TSNE_VERBOSE, n_iter=TSNE_ITERS, learning_rate=tsneLearningRate, init=data_emb).fit_transform(data)

        # make ellipse
        df = pd.DataFrame(data_emb, columns=['x', 'y'])
        df['x'] = df['x'] - np.mean(df['x'])
        df['y'] = df['y'] - np.mean(df['y'])

        df['distance'] = np.sqrt( (df['x'])*(df['x']) + (df['y'])*(df['y']) )
        df['new_distance'] = (np.arctan(df['distance'] / np.mean(df['distance'])) / (np.pi / 2)) * magicSize
        df['x'] = df['x'] * (df['new_distance'] / df['distance'])
        df['y'] = df['y'] * (df['new_distance'] / df['distance'])

        data_emb = df[['x', 'y']].values

        np.save("output/model.npy", data_emb)

    return data_emb
