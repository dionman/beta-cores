from __future__ import division
import numpy as np
import pickle
import sys
import pymc3 as pm
from pymc3.math import logsumexp
import theano.tensor as tt
import collections
sys.path.append('../')
from privacy import *
sys.path.append('../../')
from dp_advi_pymc import advi_minibatch as ad
from multiprocessing import Pool

Cs=[1., 10., 5., 50.] 

##### values for ds1.100 experiment
mul_sigmas = np.asarray([None, 0.5, 0.7, 1., 2, 10., 100.])

'''
##### values for fma experiment
mul_sigmas = np.asarray([None, 0.7, 1., 2, 10., 30, 100.])
'''
'''
##### values for santa100K experiment
mul_sigmas = np.asarray([None, 0.4, 0.5, 0.7, 1., 1.5, 4, 10.,  100.])
'''

def linearize():
  args_dict = dict()
  c = -1
  for i in range(5):
    for C in Cs:
      for k in range(mul_sigmas.shape[0]):
        for dnm in ['ds1.100']:
          c += 1
          args_dict[c] = (i, C, k, dnm)
  return args_dict

mapping = linearize()
i, C, k, dnm = mapping[int(sys.argv[1])]
#i, C, k, dnm = mapping[0]

np.random.seed(i)

def sigmoid(x):
  lower = 1e-6
  upper = 1-1e-6
  return lower + (upper - lower) * np.exp(x) / (1 + np.exp(x))

def _advi(data, T, q, learning_rate=0.01, C=None, delta=None, sigma=None):
  y, x = data
  N, D = x.shape
  # Create PyMC3 model with subsampling
  B = int(q*N) # Batch size
  x_t = tt.matrix()
  y_t = tt.vector()
  x_t.tag.test_value = np.zeros((B, D)).astype(float)
  y_t.tag.test_value = np.zeros((B,)).astype(float)
  with pm.Model() as logistic_model:
    w = pm.MvNormal('w', mu = np.zeros(D), tau = np.eye(D), shape=(D,))
    y_obs = pm.Bernoulli('y_obs', p=sigmoid(tt.dot(x_t,w)), observed=y_t)
  def minibatch_gen(y, x):
    while True:
      ixs = np.random.choice(range(0,N),B)
      yield y[ixs], x[ixs]
  minibatches = minibatch_gen(y, x)
  # With privacy use advi_minibatch
  if(sigma!=None): means, sds, elbos = ad.advi_minibatch(vars=None, start=None, model=logistic_model, n=T, n_mcsamples=1,
					minibatch_RVs=[y_obs], minibatch_tensors=[y_t, x_t], minibatches=minibatches, total_size=N, learning_rate=learning_rate, verbose=0, dp_par = [sigma, C])
  # Non private version uses pm original advi
  else: means, sds, elbos = ad.advi_minibatch(vars=None, start=None, model=logistic_model, n=T, n_mcsamples=1,
					minibatch_RVs=[y_obs], minibatch_tensors=[y_t, x_t], minibatches=minibatches, total_size=N, learning_rate=learning_rate)
  # Laplace approx
  w = means['w']
  S = np.diag(sds['w'])**2
  # Privacy calculation
  if(sigma!=None):
    return w, S, analysis.epsilon(x.shape[0], q*x.shape[0], sigma/C, T, delta)
  else:
    return w, S


# Read data
np.random.seed(int(i))
data = np.load('../../../data/'+dnm+'.npz')
X = data['X']
y = data['y']
y[y==-1]=0
m = X[:, :-1].mean(axis=0)
V = np.cov(X[:, :-1], rowvar=False)+1e-12*np.eye(X.shape[1]-1)
X[:, :-1] = np.linalg.solve(np.linalg.cholesky(V), (X[:, :-1] - m).T).T
data = [y,X]


# Inference and DP hyperparams
T = 2000 #  Number of ADVI iterations
delta = float(1./X.shape[0])
q = float(200./X.shape[0]) # subsampling ratio

'''
for C_ in Cs:
  for k_ in range(mul_sigmas.shape[0]):
    if mul_sigmas[k_] is not None:
      sigma_ = mul_sigmas[k_]*C_
      print(analysis.epsilon(X.shape[0], q*X.shape[0], sigma_/C_, T, delta)) # see privacy levels covered by experiment
  print('\n')
exit()
'''

# Run inference and store results
print('running with : ', i, C, k)
if mul_sigmas[k]:
  sigma = mul_sigmas[k]*C
  mu,Sigma,eps=_advi(data=data, T=T, q=q, learning_rate=0.01, C=C, delta=delta, sigma=sigma)
  f = open('../../../riemann_logistic_poisson_regression/results/'+dnm+'DPVI'+'_C'+str(C)+'_'+str(k)+'_'+str(i)+'.pk', 'wb')
  res = (mu, Sigma, eps)
else:
  sigma = None
  f = open('../../../riemann_logistic_poisson_regression/results/'+dnm+'DPVI'+'_nonpriv_'+str(i)+'.pk', 'wb')
  mu,Sigma =_advi(data=data, T=T, q=q, learning_rate=0.01, C=C, delta=delta, sigma=sigma)
  res = (mu, Sigma)
pickle.dump(res, f)
f.close()
