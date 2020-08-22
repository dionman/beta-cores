import numpy as np
import pandas as pd
import io, os, requests
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
import itertools
import pickle as pk

with open('_phishing.txt') as f:
    lines = (line for line in f if not line.startswith('having'))
    dataset = np.loadtxt(lines, delimiter=',', skiprows=1)

X, y= dataset[:,:-1], dataset[:,-1]
print(dataset.shape)
enc = OneHotEncoder()
enc.fit(X)

pca = PCA(n_components=10)
pca.fit(X)
X = pca.transform(X)
X = np.c_[ X, np.ones(X.shape[0])]

print(X.shape, y)
np.savez('phish', X=X, y=y, Xt=None, yt=None)
