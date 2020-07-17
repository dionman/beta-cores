import numpy as np
import pickle as pk
import os, sys
from multiprocessing import Pool
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import bayesiancoresets as bc
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
import gaussian
from scipy.optimize import minimize, nnls
import scipy.linalg as sl
from model_lr import *
import pystan

nm = sys.argv[1]
dnm = sys.argv[2]
ID = sys.argv[3]
graddiag = (sys.argv[4]=="True") # diagonal Gaussian assumption for coreset sampler
riemann_coresets = ['BPSVI', 'SVI', 'BCORES']
if nm in riemann_coresets: i0 = float(sys.argv[5])
f_rate = float(sys.argv[6])
np.random.seed(int(ID))

weighted_logistic_code = """
data {
  int<lower=0> N; // number of observations
  int<lower=0> d; // dimensionality of x
  matrix[N,d] x; // inputs
  int<lower=0,upper=1> y[N]; // outputs in {0, 1}
  vector[N] w; // weights
}
parameters {
  real theta0; // intercept
  vector[d] theta; // logreg params
}
model {
  theta0 ~ normal(0, 1);
  theta ~ normal(0, 1);
  for(n in 1:N){
    target += w[n]*bernoulli_logit_lpmf(y[n]| theta0 + x[n]*theta);
  }
}
"""

if not os.path.exists('pystan_model_logistic.pk'):
  sml = pystan.StanModel(model_code=weighted_logistic_code)
  f = open('pystan_model_logistic.pk','wb')
  pk.dump(sml, f)
  f.close()
else:
  f = open('pystan_model_logistic.pk','rb')
  sml = pk.load(f)
  f.close()

#computes the Laplace approximation N(mu, Sig) to the posterior with weights wts
def get_laplace(wts, Z, mu0, diag=False):
  trials = 10
  Zw = Z[wts>0, :]
  ww = wts[wts>0]
  while True:
    try:
      res = minimize(lambda mu : -log_joint(Zw, mu, ww)[0], mu0,
                        jac=lambda mu : -grad_th_log_joint(Zw, mu, ww)[0,:])
    except:
      mu0 = mu0.copy()
      mu0 += np.sqrt((mu0**2).sum())*0.1*np.random.randn(mu0.shape[0])
      trials -= 1
      if trials <= 0:
        print('Tried laplace opt 10 times, failed')
        break
      continue
    break
  mu = res.x
  if diag:
    sqrts_hess = np.sqrt(-diag_hess_th_log_joint(Zw, mu, ww)[0,:])
    LSigInv = np.diag(sqrts_hess)
    LSig = np.diag(1./sqrts_hess)
  else:
    LSigInv = np.linalg.cholesky(-hess_th_log_joint(Zw, mu, ww)[0,:,:])
    LSig = sl.solve_triangular(LSigInv, np.eye(LSigInv.shape[0]), lower=True, overwrite_b=True, check_finite=False)
  return mu, LSig, LSigInv

###############################
## TUNING PARAMETERS ##
M = 100
SVI_step_sched = lambda itr : i0/(1.+itr)
BPSVI_step_sched = lambda m: lambda itr : i0/(1.+itr) # make step schedule potentially dependent on coreset size
BCORES_step_sched = lambda itr : i0/(1.+itr)

n_subsample_opt = 200
n_subsample_select = 1000
projection_dim = 100 #random projection dimension
SVI_opt_itrs = 500
BPSVI_opt_itrs = 500
BCORES_opt_itrs = 500
sz = 1000
###############################

print('Loading dataset '+dnm)
X, Y, Xt, Yt = load_data('../data/'+dnm+'.npz') # read train and test data
X, Y, Z, x_mean, x_std = std_cov(X, Y) # standardize covariates
X, Y = perturb(X, Y, f_rate=f_rate)# corrupt datapoints
D = X.shape[1]
#create the prior
mu0 = np.zeros(D)
Sig0 = np.eye(D)

print('Building projectors')
def sampler_w(sz, w, pts, diag=graddiag):
  if pts.shape[0] == 0:
    w = np.zeros(1)
    pts = np.zeros((1, Z.shape[1]))
  muw, LSigw, LSigwInv = get_laplace(w, pts, mu0, diag)
  return muw + np.random.randn(sz, muw.shape[0]).dot(LSigw.T)

grad_beta = lambda x, th, beta : gaussian_beta_gradient(x, th, beta, Siginv, logdetSig)
prj_w = bc.BlackBoxProjector(sampler_w, projection_dim, log_likelihood, grad_z_log_likelihood)
prj_bw = bc.BetaBlackBoxProjector(sampler_w, projection_dim, beta_likelihood, beta_likelihood, grad_beta)

print('Creating coresets object')
#create coreset construction objects

unif = bc.UniformSamplingCoreset(Z)
sparsevi = bc.SparseVICoreset(Z, prj_w, opt_itrs = SVI_opt_itrs, n_subsample_opt = n_subsample_opt,
                              n_subsample_select = n_subsample_select, step_sched = SVI_step_sched)
bpsvi = bc.BatchPSVICoreset(Z, prj_w, opt_itrs = BPSVI_opt_itrs, n_subsample_opt = n_subsample_opt,
                            step_sched = BPSVI_step_sched)
bcoresvi = bc.BetaCoreset(Z, prj_bw, opt_itrs = BCORES_opt_itrs, n_subsample_opt = n_subsample_opt,
                           n_subsample_select = n_subsample_select, step_sched = BCORES_step_sched,
                           beta = .1, learn_beta=False)
algs = {'BCORES': bcoresvi,
        'SVI': sparsevi,
        'BPSVI': bpsvi,
        'RAND': unif,
        'PRIOR': None}
alg = algs[nm]

print('Building coresets via ' + nm)
w = [np.array([0.])]
p = [np.zeros((1, Z.shape[1]))]
ls = [np.array([0.])]

def build_per_m(m): # construction in parallel for different coreset sizes used in BPSVI
  alg.build(1, m)
  return alg.get()

if nm in ['BPSVI']:
  pool = Pool(processes=100)
  res = pool.map(build_per_m, range(1, M+1))
  i=1
  for (wts, pts, idcs) in res:
    w.append(wts)
    pts =
    print(idcs)
    pts = Y[idcs, np.newaxis]*pts
    p.append(pts)
    ls.append(Y[idcs])
    i+=1
else:
  for m in range(1, M+1):
    if nm != 'PRIOR':
      alg.build(1, m)
      #record weights
      if nm=='BCORES':
        wts, pts, idcs, beta = alg.get()
      else:
        wts, pts, idcs = alg.get()
      w.append(wts)
      pts = Y[idcs, np.newaxis]*pts
      p.append(pts)
      ls.append(Y[idcs])
    else:
      w.append(np.array([0.]))
      p.append(np.zeros((1,D)))

Xt = np.hstack((np.ones(Xt.shape[0])[:,np.newaxis], Xt))
N_per = 1000

accs = np.zeros(M+1)
pll = np.zeros(M+1)

print('Evaluation')
if nm=='PRIOR':
  sampler_data = {'x': np.zeros((1,D)), 'y': [0], 'd': D, 'N': 1, 'w': [0]}
  thd = sampler_data['d']+1
  fit = sml.sampling(data=sampler_data, iter=N_per*2, chains=1, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=False)
  thetas = fit.extract(permuted=False)[:, 0, :thd]
  for m in range(M+1):
    accs[m]= compute_accuracy(Xt, Yt, thetas)
    pll[m]=np.sum(log_likelihood(Yt[:, np.newaxis]*Xt,thetas))
else:
  for m in range(M+1):
    cx, cy = p[m], ls[m].astype(int)
    cy[cy==-1] = 0
    sampler_data = {'x': cx, 'y': cy, 'd': cx.shape[1], 'N': cx.shape[0], 'w': w[m]}
    thd = sampler_data['d']+1
    fit = sml.sampling(data=sampler_data, iter=N_per*2, chains=1, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=False)
    thetas = fit.extract(permuted=False)[:, 0, :thd]
    accs[m]= compute_accuracy(Xt, Yt, thetas)
    pll[m]=np.sum(log_likelihood(Yt[:, np.newaxis]*Xt,thetas))
print('accuracies : ', accs)
print('pll : ', pll)

#save results
f = open('results/'+dnm+'_'+nm+'_results_'+ID+'.pk', 'wb')
res = (w, p, accs, pll)
pk.dump(res, f)
f.close()
