import numpy as np
import pickle as pk
import os, sys, time
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import bayesiancoresets as bc
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
import model_neurlinear
from model_neurlinear import *
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.sparse import csr_matrix
from pystan import stan, StanModel
from collections import OrderedDict

nm = sys.argv[1]
tr = sys.argv[2]
np.random.seed(int(tr))

results_fldr = 'results'
if not os.path.exists(results_fldr):
  os.mkdir(results_fldr)

M = 100 # max coreset sz
SVI_opt_itrs = 100
BPSVI_opt_itrs = 100
BCORES_opt_itrs = 100
n_subsample_opt = 100
n_subsample_select = 400
proj_dim = 50
i0 = 0.1 # starting learning rate
BPSVI_step_sched = lambda m: lambda i : i0/(1.+i)
SVI_step_sched = lambda i : i0/(1.+i)
BCORES_step_sched = lambda i : i0/(1.+i)


X, num_entries, useridx, itemidx, U, I, K = read_data()

#create function to output log_likelihood given param samples
print('Creating log-likelihood function')
log_likelihood = lambda x, samples: pmf_log_likelihood(x, samples[0], samples[1])
print('Creating gradient log-likelihood function')
grad_log_likelihood = lambda x, samples: pmf_grad_x_loglikelihood(x, samples[0], samples[1])
print('Creating black box projector for sampling from coreset posterior')


def sampler_w(sz, wts, pts, l=10**3):
  start_time = time.time()
  phis = np.random.exponential(l, (proj_dim, U, K))
  psis = np.random.exponential(l, (proj_dim, I, K))
  if pts.shape[0]>0:
    pts = csr_matrix(pts)
    us, items = pts.nonzero()
    us+=1
    items+=1
    ratings = np.array(pts.data, dtype=int)
    number_entries = len(items)
    sampler_data = dict(U=len(set(us)),I=I, K=int(K),
              number_entries=int(np.sum(ratings)),
              user_index=us, item_index=items,
              rating=ratings,l=10**3, wts=wts)
    #fit = sm.sampling(data=sampler_data, iter=1000, verbose=False)
    fit = sm.vb( data=sampler_data, iter=1000, verbose=False)
    #results = fit.extract(permuted=True)
    results = pystan_vb_extract(fit)
    phis, psis = results['theta'][-proj_dim:], results['beta'][-proj_dim:]
  print('time required : ', time.time() - start_time)
  return (phis,psis)



prj_w = bc.BlackBoxProjector(sampler_w, proj_dim, log_likelihood, grad_log_likelihood)
#prj_bw = bc.BetaBlackBoxProjector(sampler_w, proj_dim, beta_likelihood, log_likelihood, grad_beta)

#create coreset construction objects
print('Creating coreset construction objects')
sparsevi = bc.SparseVICoreset(X, prj_w, opt_itrs = SVI_opt_itrs, n_subsample_opt = n_subsample_opt,
                              n_subsample_select = n_subsample_select, step_sched = SVI_step_sched)
bpsvi = bc.BatchPSVICoreset(X, prj_w, opt_itrs = BPSVI_opt_itrs, n_subsample_opt = n_subsample_opt,
                            step_sched = BPSVI_step_sched)
'''
bcoresvi = bc.BetaCoreset(Xcorrupted, prj_bw, opt_itrs = BCORES_opt_itrs, n_subsample_opt = n_subsample_opt,
                           n_subsample_select = n_subsample_select, step_sched = BCORES_step_sched,
                           beta = .1, learn_beta=False)
giga_optimal = bc.HilbertCoreset(X, prj_optimal)
giga_realistic = bc.HilbertCoreset(X, prj_realistic)
'''
unif = bc.UniformSamplingCoreset(X)

algs = {#'BCORES': bcoresvi,
        'BPSVI': bpsvi,
        'SVI': sparsevi,
        #'GIGAO': giga_optimal,
        #'GIGAR': giga_realistic,
        'RAND': unif,
        'PRIOR': None}
alg = algs[nm]

print('Building coreset')
#build coresets

w = [np.array((1, X.shape[0]))]
p = [np.zeros((1, X.shape[1]))]

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
    else:
      w.append(np.array([0.]))
      p.append(np.zeros((1,Y.shape[0])))
    print('built for m=',m)

# saving results
f = open('results/results_'+nm+'_'+str(tr)+'.pk', 'wb')
res = (X, w, p)
pk.dump(res, f)
f.close()
