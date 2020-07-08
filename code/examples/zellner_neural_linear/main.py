import numpy as np
import pickle as pk
import os, sys, time
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import bayesiancoresets as bc
from scipy.stats import multivariate_normal
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from model_neurlinr import *
from neural import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal as MVN

nm = sys.argv[1]
tr = sys.argv[2]
np.random.seed(int(tr))

synthetic = True
## Prepare and read dataset
test_size = 1000 # number of test datapoints
if synthetic:
  N = 3000  # number of data points
  X, Y = build_synthetic_dataset(N)
else:
  X, Y = load_data('naval', data_dir='../data', seed=int(tr))
X, Y = X.astype(np.float32), Y.astype(np.float32)
X, Y, X_test, Y_test = X[:-test_size,:], Y[:-test_size], X[-test_size:,:], Y[-test_size:]
Z = np.hstack((X, Y)).astype(np.float32)
Z_test = np.hstack((X_test, Y_test)).astype(np.float32)
print('Z and Z_test shape : ', Z.shape, Z_test.shape)

results_fldr = 'results'
if not os.path.exists(results_fldr):
  os.mkdir(results_fldr)

nl = NeuralLinear(Z, out_features=30)
M = 500 # max coreset sz
SVI_opt_itrs = 1000
BPSVI_opt_itrs = 1000
BCORES_opt_itrs = 1000
n_subsample_opt = 200
n_subsample_select = 1000
proj_dim = 100
i0 = 0.1 # starting learning rate
BPSVI_step_sched = lambda m: lambda i : i0/(1.+i)
SVI_step_sched = lambda i : i0/(1.+i)
BCORES_step_sched = lambda i : i0/(1.+i)

out_features = 30 # dimension of the ouput of the neural encoder used for lin reg
#get empirical mean/std
datastd = Y.std()
datamn = Y.mean()

mu0 = datamn*np.ones(out_features)
ey = np.eye(out_features)
Sig0 = (datastd**2+datamn**2)*ey
Sig0inv = np.linalg.inv(Sig0)

#create function to output log_likelihood given param samples
print('Creating log-likelihood function')
deep_encoder = lambda nl, z: (np.hstack((nl.encode(torch.from_numpy(z[:, :-1].astype(np.float32))).detach().numpy(),
                                            z[:,-1][:,np.newaxis].astype(np.float32))))
log_likelihood = (lambda z, th, nl: neurlinr_loglikelihood(deep_encoder(nl, z), th, datastd**2))
grad_log_likelihood = lambda z, th: None

'''
print('Creating gradient log-likelihood function')
grad_log_likelihood = lambda z, th, nl: neurlinreg_grad_x_loglikelihood(z, th, nl)
print('Creating gradient grad beta function')
grad_beta = lambda z, th, nl, beta : neurlinreg_beta_gradient(z, th, nl, beta)
'''

print('Creating black box projector for sampling from coreset posterior')
def sampler_w(n, wts, pts):
  if pts.shape[0] == 0:
    wts = np.zeros(1)
    pts = np.zeros((1, Z.shape[1]))
  muw, LSigw, LSigwInv = weighted_post(mu0, Sig0inv, datastd**2, deep_encoder(nl, pts), wts)
  return muw + np.random.randn(n, muw.shape[0]).dot(LSigw.T)

prj_w = bc.BlackBoxProjector(sampler_w, proj_dim, log_likelihood, grad_log_likelihood, nl=nl)
#prj_bw = bc.BetaBlackBoxProjector(sampler_w, proj_dim, beta_likelihood, log_likelihood, grad_beta)


#create coreset construction objects
print('Creating coreset construction objects')
sparsevi = bc.SparseVICoreset(Z, prj_w, opt_itrs = SVI_opt_itrs, n_subsample_opt = n_subsample_opt,
                              n_subsample_select = n_subsample_select, step_sched = SVI_step_sched)
'''
bpsvi = bc.BatchPSVICoreset(X, prj_w, opt_itrs = BPSVI_opt_itrs, n_subsample_opt = n_subsample_opt,
                            step_sched = BPSVI_step_sched)
bcoresvi = bc.BetaCoreset(X, prj_bw, opt_itrs = BCORES_opt_itrs, n_subsample_opt = n_subsample_opt,
                           n_subsample_select = n_subsample_select, step_sched = BCORES_step_sched,
                           beta = .1, learn_beta=False)
'''
unif = bc.UniformSamplingCoreset(X)

algs = {#'BCORES': bcoresvi,
        #'BPSVI': bpsvi,
        'SVI': sparsevi,
        'RAND': unif,
        'PRIOR': None}
alg = algs[nm]

print('Building coreset')
#build coresets
w = [np.array([0.])]
p = [np.zeros((1, Z.shape[1]))]

def build_per_m(m): # construction in parallel for different coreset sizes used in BPSVI
  print('building for m=', m)
  alg.build(1, m)
  print('built for m=',m)
  return alg.get()

if nm in ['BPSVI']:
  from multiprocessing import Pool
  pool = Pool(processes=10)
  res = pool.map(build_per_m, range(1, M+1))
  i=1
  for (wts, pts, _) in res:
    w.append(wts)
    p.append(pts)
    i+=1
else:
  for m in range(1, M+1):
    if nm!='PRIOR':
      print('trial: ' + str(tr) +' alg: ' + nm + ' ' + str(m) +'/'+str(M))
      alg.build(1, m)
      #store weights
      if nm=='BCORES':
        wts, pts, idcs, beta = alg.get()
        print(alg.get())
      else:
        wts, pts, idcs = alg.get()
      w.append(wts)
      p.append(pts)
      nl = NeuralLinear(p[-1].astype(np.float32), out_features=30)
    else:
      w.append(np.array([0.]))
      p.append(np.zeros((1,Y.shape[0])))
    if m%10==0:
      # train deep feature extractor with current coreset datapoints
      nl.optimize(torch.from_numpy(w[-1].astype(np.float32)), torch.from_numpy(p[-1].astype(np.float32)))
    alg.update_encoder(nl)
    test_nll, test_performance = nl.test(torch.from_numpy(Z_test))
    print(test_nll, test_performance)


f = open('results/results_'+nm+'_'+str(tr)+'.pk', 'wb')
f.close()
