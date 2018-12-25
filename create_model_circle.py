import os
import numpy as np
from scipy import misc, stats
from sklearn.manifold import TSNE
from pathlib import Path
import pandas as pd
import pylab as plt

data = np.load("data.npy")

TSNE_ITERS = 500
CIRCULAR_ITERS = 10
CIRCULAR_MAX_TSNE_ITERS = 500
MAGIC_SIZE = 42

modelPath = Path("model.npy")

if not modelPath.is_file():
    data_emb = TSNE(n_components=2, perplexity=120, verbose=1, n_iter=TSNE_ITERS).fit_transform(data)

    np.save("model.npy", data_emb)

for i in range(CIRCULAR_ITERS):
    data_emb = np.load("model.npy")

    # apply tsne
    tsneLearningRate = 200.0 * (float(CIRCULAR_ITERS-i) / CIRCULAR_ITERS)
    data_emb = TSNE(n_components=2, perplexity=120, verbose=1, n_iter=TSNE_ITERS, learning_rate=tsneLearningRate, init=data_emb).fit_transform(data)

    # make ellipse
    df = pd.DataFrame(data_emb, columns=['x', 'y'])
    df['x'] = df['x'] - np.mean(df['x'])
    df['y'] = df['y'] - np.mean(df['y'])
    # df['x'] = (df['x'] / np.mean(df['x']*df['x']))
    # df['y'] = (df['y'] / np.mean(df['y']*df['y']))

    df['distance'] = np.sqrt( (df['x'])*(df['x']) + (df['y'])*(df['y']) )
    df['new_distance'] = (np.arctan(df['distance'] / np.mean(df['distance'])) / (np.pi / 2)) * MAGIC_SIZE
    df['x'] = df['x'] * (df['new_distance'] / df['distance'])
    df['y'] = df['y'] * (df['new_distance'] / df['distance'])

    data_emb = df.as_matrix(columns=['x', 'y'])

    np.save("model.npy", data_emb)


    model = np.transpose(data_emb)
    colors = data[:,0:3]
    plt.scatter(model[0], model[1], color=colors)
    plt.savefig("model.png")
    plt.clf()
