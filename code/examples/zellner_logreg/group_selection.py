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

# specify random number only for test size randomization (common across trials)
np.random.seed(42)
rnd = np.random.rand()
flatten = lambda l: [item for sublist in l for item in sublist]

def linearize():
  args_dict = dict()
  c=-1
  for beta in [0.9]:
    for ID in range(5):
      for f_rate in [0, 0.1]:
        for nm in ['BCORES']: #['RAND']: #['DShapley']: #['RAND', 'DShapley', 'BCORES']:
          c+=1
          args_dict[c] = (ID, nm, f_rate, beta)
  return args_dict

mapping = linearize()
ID, nm, f_rate, beta = mapping[int(sys.argv[1])]
#ID, nm, f_rate, beta = mapping[0]

dnm = "diabetes" #"adult"
graddiag = False # diagonal Gaussian assumption for coreset sampler
structured = False
riemann_coresets = ['SVI', 'BCORES']
if nm in riemann_coresets: i0 = 1.0
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
M = 10
BCORES_step_sched = lambda itr : i0/(1.+itr)
n_subsample_opt = 1000
n_subsample_select = 1000
projection_dim = 200 #random projection dimension
BCORES_opt_itrs = 500
sz = 1000
N_per = 1000
###############################

print('Loading dataset '+dnm)
X, Y, Xt, Yt = load_data('../data/'+dnm+'.npz') # read train and test data
X, Y, Z, x_mean, x_std = std_cov(X, Y) # standardize covariates
N, D = X.shape

f = open('../data/vq_groups_sensemake_diabetes.pk', 'rb')
res = pk.load(f) 
f.close()
(groups, demos)=res
groups = [[k for k in g if k<Z.shape[0]] for g in groups]
grouptot = sum([len(g) for g in groups])

if f_rate>0:
  for (g,d) in zip(groups,demos):
    X[g,:], Y[g], Z[g,:], _ = perturb(X[g,:], Y[g], f_rate=2*d[0]*f_rate, structured=structured, noise_x=(0,10))

# make sure test set is adequate for evaluation via the predictive accuracy metric
if len(Yt[Yt==1])>0.55*len(Yt) or len(Yt[Yt==1])<0.45*len(Yt): # truncate for balanced test dataset
  totrunc=-1 # totrunc holds the majority label to be truncated (for fairness of accuracy metric)
  if len(Yt[Yt==1])> len(Yt[Yt==-1]):
    totrunc=1
  idcs = ([i for i, e in enumerate(Yt) if e == totrunc][:len(Yt[Yt==-totrunc])+int(0.01*len(Yt[Yt==-totrunc])*rnd)]
         +[i for i, e in enumerate(Yt) if e == -totrunc])
  Xt, Yt = Xt[idcs,:], Yt[idcs]
Xt, Yt, _, _, _ = std_cov(Xt, Yt, mean_=x_mean, std_=x_std) # standardize covariates for test data

####################################################################
# functions used in DShapley and RAND
def update_per_t(t, maxGroups=13):
  phis = np.zeros(len(groups),) # initialize Shapley values for all groups to zero
  vs = np.zeros(len(groups)+1,) # values for group combinations for all groups to zero
  idcs = np.random.permutation(len(groups))
  for j in range(maxGroups):
    datapoints = flatten([groups[idx] for idx in idcs[:j]])
    vs[j+1] = eval(datapoints, X, Y, Xt, Yt)
    phis[idcs[j]] += vs[j+1]-vs[j] # add new marginal for group idcs[j]
  return phis

def dshapley(groups, X, Y, Xt, Yt, T=1000):
  pool = Pool(processes=100)
  res = pool.map(update_per_t, range(T))
  phis = np.mean(res, axis=0)
  return phis

def eval(idcs, X, Y, Xt, Yt, N_per=1000):
  cx, cy = X[idcs][:, :-1], Y[idcs].astype(int)
  cy[cy==-1] = 0
  sampler_data = {'x': cx, 'y': cy, 'd': cx.shape[1], 'N': cx.shape[0], 'w': np.ones(cx.shape[0])}
  thd = sampler_data['d']+1
  fit = sml.sampling(data=sampler_data, iter=N_per*2, chains=1, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=False)
  thetas = np.roll(fit.extract(permuted=False)[:, 0, :thd], -1)
  acc = compute_accuracy(Xt, Yt, thetas)
  return acc
####################################################################

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
prj_bw = bc.BetaBlackBoxProjector(sampler_w, projection_dim, beta_likelihood, beta_likelihood, grad_beta)

print('Creating coresets object')
bcoresvi = bc.BetaCoreset(Z, prj_bw, opt_itrs = BCORES_opt_itrs, n_subsample_opt = n_subsample_opt,
                          n_subsample_select = None, step_sched = BCORES_step_sched,
                          beta = beta, learn_beta=False, groups=groups)
dem=[[]]
indices = [np.array([0.])]
accs = np.zeros(M+1)

if nm=='BCORES':
  algs = {'BCORES': bcoresvi,
          'PRIOR': None}
  alg = algs[nm]

  print('Building coresets via ' + nm)
  w = [np.array([0.])]
  p = [np.zeros((1, Z.shape[1]))]
  ls = [np.array([0.])]

  for m in range(1, M+1):
    if nm != 'PRIOR':
      alg.build(1, N)
      #record weights
      if nm=='BCORES':
        wts, pts, idcs, beta = alg.get()
      else:
        wts, pts, idcs = alg.get()
      w.append(wts)
      pts = Y[idcs, np.newaxis]*pts
      p.append(pts)
      dem+=[[demos[selgroup] for selgroup in alg.selected_groups]]
      ls.append(Y[idcs])
      indices.append(np.array(flatten([groups[idx] for idx in alg.selected_groups]))) 
    else:
      w.append(np.array([0.]))
      p.append(np.zeros((1,D)))

  print('Evaluation')
  if nm=='PRIOR':
    sampler_data = {'x': np.zeros((1,D-1)), 'y': [0], 'd': D, 'N': 1, 'w': [0]}
    thd = sampler_data['d']+1
    fit = sml.sampling(data=sampler_data, iter=N_per*2, chains=1, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=False)
    thetas = np.roll(fit.extract(permuted=False)[:, 0, :thd], -1)
    for m in range(M+1):
      accs[m]= compute_accuracy(Xt, Yt, thetas)
  else:
    # sample from prior for coreset size 0
    sampler_data = {'x': np.zeros((1,D-1)), 'y': [0], 'd': D, 'N': 1, 'w': [0]}
    thd = sampler_data['d']+1
    fit = sml.sampling(data=sampler_data, iter=N_per*2, chains=1, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=False)
    thetas = np.roll(fit.extract(permuted=False)[:, 0, :thd], -1)
    accs[0]= compute_accuracy(Xt, Yt, thetas)
    for m in range(1,M+1):
      print('selected cx with shape : ', p[m][:, :-1].shape, ' and weights', w[m])
      # subsample for MCMC
      cx, cy = p[m][:, :-1], ls[m].astype(int)
      cy[cy==-1] = 0
      sampler_data = {'x': cx, 'y': cy, 'd': cx.shape[1], 'N': cx.shape[0], 'w': w[m]}
      thd = sampler_data['d']+1
      fit = sml.sampling(data=sampler_data, iter=N_per*2, chains=1, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=False)
      thetas = np.roll(fit.extract(permuted=False)[:, 0, :thd], -1)
      accs[m]= compute_accuracy(Xt, Yt, thetas)
  print('accuracies : ', accs)

elif nm=='DShapley':
  phis = dshapley(groups, X, Y, Xt, Yt)
  selected_groups = np.argsort(phis)[::-1] # sort groups according to Shapley value and select greedily
  accs = np.zeros(M+1)
  print('Evaluation')
  for m in range(M+1):
    datapoints = flatten([groups[idx] for idx in selected_groups[:m]])
    accs[m] = eval(datapoints, X, Y, Xt, Yt, N_per=1000)
    dem+=[[demos[selgroup] for selgroup in selected_groups[:m]]]
    indices.append(np.array(datapoints))
  print('accuracies : ', accs)

elif nm=='RAND':
  selected_groups = np.random.permutation(len(groups)) # randomize order of groups
  for m in range(M+1):
    datapoints = flatten([groups[idx] for idx in selected_groups[:m]])
    accs[m] = eval(datapoints, X, Y, Xt, Yt, N_per=1000)
    dem+=[[demos[selgroup] for selgroup in selected_groups[:m]]]
    indices.append(np.array(datapoints))
  print('accuracies : ', accs)

#save results
f = open('/home/dm754/rds/hpc-work/zellner_logreg/group_results/'+dnm+'_'+nm+'_'+str(f_rate)+'_results_'+str(ID)+'.pk', 'wb')
res = (accs, indices, dem)
pk.dump(res, f)
f.close()
