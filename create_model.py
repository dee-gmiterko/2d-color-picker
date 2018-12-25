import os
import numpy as np
from scipy import misc
from sklearn.manifold import TSNE

data = np.load("data.npy")

data_emb = TSNE(n_components=2, perplexity=120, verbose=1).fit_transform(data)

np.save("model.npy", data_emb)
