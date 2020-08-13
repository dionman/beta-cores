import numpy as np
import pickle as pk
import os, sys, torch
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import bayesiancoresets as bc
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from model_neurlinr import *
from neural import *

# randomize datapoints order
def unison_shuffled_copies(a, b):
  assert a.shape[0] == b.shape[0]
  p = np.random.permutation(a.shape[0])
  return a[p], b[p]

# Parse input arguments
dnm = sys.argv[1]
algnm = sys.argv[2]
tr = sys.argv[3]
f_rate = float(sys.argv[4])
beta = float(sys.argv[5])
np.random.seed(int(tr))
#Specify results folder
results_fldr = 'results'
if not os.path.exists(results_fldr):
  os.mkdir(results_fldr)

## Prepare and read dataset
if dnm=='synthetic':
  N = 3000  # number of data points
  X, Y = build_synthetic_dataset(N)
else:
  X, Y = load_data(dnm, data_dir='../data')
  N = Y.shape[0]  # number of data points
init_size = 10 #max(20, int(0.01*N))
test_size = int(0.1*N)

# Split datasets
X, Y = unison_shuffled_copies(X.astype(np.float32), Y.astype(np.float32))
X_init, Y_init, X, Y, X_test, Y_test=(
          X[:init_size,:], Y[:init_size], X[init_size:-test_size,:],
          Y[init_size:-test_size], X[-test_size:,:], Y[-test_size:])
X, Y, X_init, Y_init, X_test, Y_test, input_mean, input_std, output_mean, output_std = preprocessing(X, Y, X_init, Y_init, X_test, Y_test)
X, Y = perturb(X, Y, f_rate=f_rate)# corrupt datapoints
Z_init = np.hstack((X_init, Y_init)).astype(np.float32)
Z = np.hstack((X, Y)).astype(np.float32)
Z_test = np.hstack((X_test, Y_test)).astype(np.float32)
print(X.shape, Y.shape, X_init.shape, Y_init.shape, X_test.shape, Y_test.shape)

# Specify encoder and coreset hyperparameters
out_features = 30 # dimension of the ouput of the neural encoder used for lin reg
nl = NeuralLinearTB(Z_init, out_features=out_features, input_mean=input_mean, input_std=input_std, output_mean=output_mean, output_std=output_std, seed=tr)
train_nn_freq = 1 # frequency of nn training wrt coreset iterations
VI_opt_itrs = 1000
n_subsample_opt = 1000
n_subsample_select = 1000
proj_dim = 200
i0 = .1 # starting learning rate
SVI_step_sched = lambda i : i0/(1.+i)
#BPSVI_step_sched = lambda m: lambda i : i0/(1.+i)
BCORES_step_sched = lambda i : i0/(1.+i)
M = 50 # max num of coreset iterations

#Specify priors
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
log_likelihood = lambda z, th, nl: neurlinr_loglikelihood(deep_encoder(nl, z), th, datastd**2)
grad_log_likelihood = lambda z, th, nl:  NotImplementedError
beta_likelihood = lambda z, th, beta, nl: neurlinr_beta_likelihood(deep_encoder(nl, z), th, beta, datastd**2)
grad_beta = lambda z, th, beta, nl : NotImplementedError #neurlinr_beta_gradient(z, th, beta, Siginv, logdetSig)
#neurlinr_grad_x_loglikelihood(deep_encoder(nl, z), th, datastd**2)

print('Creating black box projector for sampling from coreset posterior')
def sampler_w(n, wts, pts):
  if pts.shape[0] == 0:
    wts = np.zeros(1)
    pts = np.zeros((1, Z.shape[1]))
  muw, LSigw, LSigwInv = weighted_post(mu0, Sig0inv, datastd**2, deep_encoder(nl, pts), wts)
  return muw + np.random.randn(n, muw.shape[0]).dot(LSigw.T)

prj_w = bc.BlackBoxProjector(sampler_w, proj_dim, log_likelihood, grad_log_likelihood, nl=nl)
prj_bw = bc.BetaBlackBoxProjector(sampler_w, proj_dim, beta_likelihood, log_likelihood, grad_beta, nl=nl)

#create coreset construction objects
print('Creating coreset construction objects')

in_batches = True
if in_batches:
  batch_size = max(20, int(N/200.))
  groups = list(np.split(np.arange(X.shape[0]), range(batch_size, X.shape[0], batch_size)))
  sparsevi = bc.SparseVICoreset(Z, prj_w, opt_itrs=VI_opt_itrs, n_subsample_opt=n_subsample_opt, n_subsample_select=None,
                              step_sched=SVI_step_sched, wts=np.ones(init_size), idcs=np.arange(init_size), pts=Z_init, groups=groups, initialized=True)
  bcoresvi = bc.BetaCoreset(Z, prj_bw, opt_itrs = VI_opt_itrs, n_subsample_opt = n_subsample_opt, n_subsample_select = None,
                              step_sched = BCORES_step_sched, beta = beta, learn_beta=False, wts=np.ones(init_size), idcs=np.arange(init_size), pts=Z_init, groups=groups)
  unif = bc.UniformSamplingCoreset(Z, wts=np.ones(init_size), idcs=np.arange(init_size), pts=Z_init, groups=groups)
else:
  sparsevi = bc.SparseVICoreset(Z, prj_w, opt_itrs=VI_opt_itrs, n_subsample_opt=n_subsample_opt, n_subsample_select=n_subsample_select,
                              step_sched=SVI_step_sched, wts=np.ones(init_size), idcs=np.arange(init_size), pts=Z_init, groups=None)

algs = {'BCORES': bcoresvi,
        #'BPSVI': bpsvi,
        'SVI': sparsevi,
        'RAND': unif,
        'PRIOR': None}
alg = algs[algnm]

# Diagnostics
nlls = np.zeros(M+1)
rmses = np.zeros(M+1)

# Build coreset
print('Building coreset')
#build coresets
w = [np.array([0.])]
p = [np.zeros((1, Z.shape[1]))]

def build_per_m(m): # construction in parallel for different coreset sizes used in BPSVI
  print('building for m=', m)
  alg.build(init_size+1, init_size+m)
  print('built for m=',m)
  return alg.get()

m=0
test_nll, test_performance = nl.test(torch.from_numpy(Z_test))
nlls[m], rmses[m] = test_nll, test_performance
if alg in ['BPSVI']:
  from multiprocessing import Pool
  pool = Pool(processes=10)
  res = pool.map(build_per_m, range(1, M+1))
  i=1
  for (wts, pts, _) in res:
    w.append(wts)
    p.append(pts)
    i+=1
    nl.update_batch(p[-1].astype(np.float32))
    if m%train_nn_freq==0:  # train deep feature extractor with current coreset datapoints
      nl.optimize(torch.from_numpy(w[-1].astype(np.float32)), torch.from_numpy(p[-1].astype(np.float32)), weight_decay=1., initial_lr=1e-3)
    test_nll, test_performance = nl.test(torch.from_numpy(Z_test))
    nlls[m], rmses[m] = test_nll, test_performance
else:
  for m in range(1, M+1):
    if algnm!='PRIOR':
      print('trial: ' + str(tr) +' alg: ' + algnm + ' ' + str(m) +'/'+str(M))
      alg.build(1, N)
      #store weights
      if algnm=='BCORES': wts, pts, idcs, beta = alg.get()
      else:
        wts, pts, idcs = alg.get()
      #print('points shape : ', pts.shape, wts)
      w.append(wts)
      p.append(pts)
      nl.update_batch(p[-1].astype(np.float32))
      if m%train_nn_freq==0:   # train deep feature extractor with current coreset datapoints
        nl.optimize(torch.from_numpy(w[-1].astype(np.float32)), torch.from_numpy(p[-1].astype(np.float32)), weight_decay=.1, initial_lr=1e-2)
      test_nll, test_performance = nl.test(torch.from_numpy(Z_test))
      nlls[m], rmses[m] = test_nll, test_performance
    else:
      w.append(np.array([0.]))
      p.append(np.zeros((1,Y.shape[0])))
      test_nll, test_performance = nl.test(torch.from_numpy(Z_test),
                      torch.from_numpy(np.asarray([datamn]*test_size).astype(np.float32)),
                      torch.from_numpy(np.asarray([datastd]*test_size).astype(np.float32)))
      nlls[m], rmses[m] = test_nll, test_performance

# Save results
f = open('results/results_'+dnm+'_'+algnm+'_frate_'+str(f_rate)+'beta'+str(beta)+'_'+str(tr)+'.pk', 'wb')
res = (w, p, rmses, nlls)
pk.dump(res, f)
f.close()
