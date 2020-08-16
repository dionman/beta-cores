import numpy as np
import pickle as pk
import os, sys, torch
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import bayesiancoresets as bc
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from model_neurlinr import *
from neural import *

def linearize():
  args_dict = dict()
  c = -1
  for beta in [0.2]:
    for tr in range(30): # trial number
      for nm in ["BCORES", "RAND", "SVI"]: # coreset method
        for i0 in [.1]:
          for f_rate in [0, 30]:
            for dnm in ["year"]: #, "prices2018"]:
              c += 1
              args_dict[c] = (tr, nm, dnm, f_rate, beta, i0)
  return args_dict

mapping = linearize()
#tr, algnm, dnm, f_rate, beta, i0 = mapping[int(sys.argv[1])]
tr, algnm, dnm, f_rate, beta, i0 = mapping[0]

# randomize datapoints order
def unison_shuffled_copies(a, b):
  assert a.shape[0] == b.shape[0]
  p = np.random.permutation(a.shape[0])
  return a[p], b[p]

# Parse input arguments
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
  #X, Y = load_data(dnm, data_dir='../data')
  X, Y = load_data(dnm, data_dir='/home/dm754/rds/hpc-work/zellner_neural/data')
  N = Y.shape[0]  # number of data points
if dnm=='boston':
  init_size = 10
  batch_size = 20
  out_features = 20 # dimension of the ouput of the neural encoder used for lin reg
  weight_decay = 1.
  initial_lr = 1e-2
  n_subsample_select = None
elif dnm=='year':
  init_size = 200
  batch_size = 100
  out_features = 100
  weight_decay = 3.
  initial_lr = 1e-2
  n_subsample_select = 100
elif dnm=='prices2018':
  init_size = 200
  batch_size = 100
  out_features = 5
  weight_decay = 1.
  initial_lr = 1e-2
  n_subsample_select = 100
test_size = int(0.1*N)
tss = min(500, test_size) # test set sample size

# Split datasets
X, Y = unison_shuffled_copies(X.astype(np.float32), Y.astype(np.float32))
X_init, Y_init, X, Y, X_test, Y_test=(
         X[:init_size,:], Y[:init_size], X[init_size:-test_size,:],
         Y[init_size:-test_size], X[-test_size:,:], Y[-test_size:])
X, Y, X_init, Y_init, X_test, Y_test, input_mean, input_std, output_mean, output_std = preprocessing(X, Y, X_init, Y_init, X_test, Y_test)

#Specify priors
#get empirical mean/std
datastd = Y.std()
datamn = Y.mean()

groups = list(np.split(np.arange(X.shape[0]), range(batch_size, X.shape[0], batch_size)))
X, Y = perturb(X, Y, f_rate=0.01*f_rate, groups=groups) # corrupt datapoints
Z_init = np.hstack((X_init, Y_init)).astype(np.float32)
Z = np.hstack((X, Y)).astype(np.float32)
Z_test = np.hstack((X_test, Y_test)).astype(np.float32)

# Specify encoder and coreset hyperparameters
nl = NeuralLinear(Z_init, out_features=out_features, input_mean=input_mean, input_std=input_std, output_mean=output_mean, output_std=output_std, seed=tr)
train_nn_freq = 1 # frequency of nn training wrt coreset iterations
VI_opt_itrs = 1000
n_subsample_opt = 1000
proj_dim = 100
SVI_step_sched = lambda i : i0/(1.+i)
#BPSVI_step_sched = lambda m: lambda i : i0/(1.+i)
BCORES_step_sched = lambda i : i0/(1.+i)
M = 50 # max num of coreset iterations

mu0 = datamn*np.ones(out_features)
ey = np.eye(out_features)
Sig0 = (datastd**2+datamn**2)*ey
Sig0inv = np.linalg.inv(Sig0)

#create function to output log_likelihood given param samples
print('Creating log-likelihood function')
deep_encoder = lambda nl, pts: (np.hstack((nl.encode(torch.from_numpy(pts[:, :-1].astype(np.float32))).detach().numpy(),
                                            pts[:,-1][:,np.newaxis].astype(np.float32))))
log_likelihood = lambda pts, th, nl: neurlinr_loglikelihood(deep_encoder(nl, pts), th, datastd**2)
grad_log_likelihood = lambda pts, th, nl:  NotImplementedError
beta_likelihood = lambda pts, th, beta, nl: neurlinr_beta_likelihood(deep_encoder(nl, pts), th, beta, datastd**2)
grad_beta = lambda pts, th, beta, nl : NotImplementedError

print('Creating black box projector for sampling from coreset posterior')
'''
def sampler_w(n, wts, pts):
  if pts.shape[0] == 0:
    wts = np.zeros(1)
    pts = np.zeros((1, Z.shape[1]))
  muw, LSigw, LSigwInv = weighted_post(mu0, Sig0inv, datastd**2, deep_encoder(nl, pts), wts)
  return muw + np.random.randn(n, muw.shape[0]).dot(LSigw.T)
'''
def sampler_w(n, wts, pts):
  if pts.shape[0] == 0:
      wts = np.zeros(1)
      pts = np.zeros((1, Z.shape[1]))
  sigsq = datastd**2
  z=deep_encoder(nl, pts)
  X = z[:, :-1]
  Y = z[:, -1]
  Sigp = np.linalg.inv(Sig0inv + (wts[:, np.newaxis]*X).T.dot(X)/sigsq)
  mup = np.dot(Sigp, np.dot(Sig0inv,np.ones(out_features)) + (wts[:, np.newaxis]*Y[:,np.newaxis]*X).sum(axis=0)/datastd**2)
  return np.random.multivariate_normal(mup, Sigp, n)


prj_w = bc.BlackBoxProjector(sampler_w, proj_dim, log_likelihood, grad_log_likelihood, nl=nl)
prj_bw = bc.BetaBlackBoxProjector(sampler_w, proj_dim, beta_likelihood, log_likelihood, grad_beta, nl=nl)

#create coreset construction objects
print('Creating coreset construction objects')

in_batches = True
if in_batches:
  sparsevi = bc.SparseVICoreset(Z, prj_w, opt_itrs=VI_opt_itrs, n_subsample_opt=n_subsample_opt, n_subsample_select=n_subsample_select,
                              step_sched=SVI_step_sched, wts=np.ones(init_size), idcs=1e7+np.arange(init_size), pts=Z_init, groups=groups, initialized=True, enforce_new=False)
  bcoresvi = bc.BetaCoreset(Z, prj_bw, opt_itrs=VI_opt_itrs, n_subsample_opt=n_subsample_opt, n_subsample_select=n_subsample_select,
                              step_sched=BCORES_step_sched, beta=beta, learn_beta=False, wts=np.ones(init_size), idcs=1e7+np.arange(init_size), pts=Z_init, groups=groups, initialized=True)
  unif = bc.UniformSamplingCoreset(Z, wts=np.ones(init_size), idcs=1e7+np.arange(init_size), pts=Z_init, groups=groups)
else:
  raise NotImplementedError("Supported only batch data acquisition")

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
def build_per_m(m): # construction in parallel for different coreset sizes used in B$
  alg.build(init_size+1, init_size+m)
  return alg.get()
m=0
test_nll, test_performance = nl.test(torch.from_numpy(Z_test[np.random.choice(Z_test.shape[0], tss, replace=False), :]))
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
    if m%train_nn_freq==0:  # train deep feature extractor with current coreset data$
      nl.optimize(torch.from_numpy(w[-1].astype(np.float32)), torch.from_numpy(p[-1].astype(np.float32)), weight_decay=weight_decay, initial_lr=initial_lr)
    test_nll, test_performance = nl.test(torch.from_numpy(Z_test[np.random.choice(Z_test.shape[0], tss, replace=False), :]))
    nlls[m], rmses[m] = test_nll, test_performance
else:
  for m in range(1, M+1):
    print('\n m=', m)
    if algnm!='PRIOR':
      alg.build(1, N)
      #store weights
      if algnm=='BCORES': wts, pts, idcs, beta = alg.get()
      else: wts, pts, idcs = alg.get()
      w.append(wts)
      p.append(pts)
      nl.update_batch(p[-1].astype(np.float32))
      print('points shape : ', pts.shape, idcs[init_size:])
      if m%train_nn_freq==0:   # train deep feature extractor with current coreset d$
        nl.optimize(torch.from_numpy(w[-1].astype(np.float32)), torch.from_numpy(p[-1].astype(np.float32)), weight_decay=weight_decay, initial_lr=initial_lr)
      test_nll, test_performance = nl.test(torch.from_numpy(Z_test[np.random.choice(Z_test.shape[0], tss, replace=False), :]))
      nlls[m], rmses[m] = test_nll, test_performance
      muw, LSigw, LSigwInv = weighted_post(mu0, Sig0inv, datastd**2, deep_encoder(nl, pts), wts)
    else:
      w.append(np.array([0.]))
      p.append(np.zeros((1,Y.shape[0])))
      test_nll, test_performance = nl.test(torch.from_numpy(Z_test[np.random.choice(Z_test.shape[0], tss, replace=False), :]),
                      torch.from_numpy(np.asarray([datamn]*test_size).astype(np.float32)),
                      torch.from_numpy(np.asarray([datastd]*test_size).astype(np.float32)))
      nlls[m], rmses[m] = test_nll, test_performance

# Save results
f = open('results/results_'+dnm+'_'+algnm+'_frate_'+str(f_rate)+'_beta_'+str(beta)+'_i0_'+str(i0)+'_'+str(tr)+'.pk', 'wb')
res = (w, p, rmses, nlls)
pk.dump(res, f)
f.close()
