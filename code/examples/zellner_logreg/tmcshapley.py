import numpy as np
import pickle as pk
import os, sys
from multiprocessing import Pool
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from scipy.optimize import minimize, nnls
import scipy.linalg as sl
from model_lr import *
import pystan

# specify random number only for test size randomization (common across trials)
np.random.seed(42)
rnd = np.random.rand()

nm = "TMCS"
dnm = "adult"
ID = 0
graddiag = False # diagonal Gaussian assumption for coreset sampler
structured = False
f_rate = 0.1
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

###############################
## TUNING PARAMETERS ##
M = 20

###############################

print('Loading dataset '+dnm)
X, Y, Xt, Yt = load_data('../data/'+dnm+'.npz') # read train and test data
X, Y, Z, x_mean, x_std = std_cov(X, Y) # standardize covariates
#if f_rate>0: X, Y, Z, outidx = perturb(X, Y, f_rate=f_rate)# corrupt datapoints
N, D = X.shape

f = open('../data/vq_groups_sensemake_adult.pk', 'rb')
res = pk.load(f) #(w, p, accs, pll)
f.close()
(groups, demos)=res

groups = [[k for k in g if k<Z.shape[0]] for g in groups]
grouptot = sum([len(g) for g in groups])

if f_rate>0:
  for (g,d) in zip(groups,demos):
    print(len(g), d, d[0])
    X[g,:], Y[g], Z[g,:], _ = perturb(X[g,:], Y[g], f_rate=0.*d[0]*f_rate, structured=structured, noise_x=(0,10))
    #input()#if f_rate>0: X, Y, Z, outidx = perturb(X, Y, f_rate=f_rate)

# make sure test set is adequate for evaluation via the predictive accuracy metric
if len(Yt[Yt==1])>0.55*len(Yt) or len(Yt[Yt==1])<0.45*len(Yt): # truncate for balanced test dataset
  totrunc=-1 # totrunc holds the majority label to be truncated (for fairness of accuracy metric)
  if len(Yt[Yt==1])> len(Yt[Yt==-1]):
    totrunc=1
  idcs = ([i for i, e in enumerate(Yt) if e == totrunc][:len(Yt[Yt==-totrunc])+int(0.01*len(Yt[Yt==-totrunc])*rnd)]
         +[i for i, e in enumerate(Yt) if e == -totrunc])
  Xt, Yt = Xt[idcs,:], Yt[idcs]
Xt, Yt, _, _, _ = std_cov(Xt, Yt, mean_=x_mean, std_=x_std) # standardize covariates for test data

#create the prior
mu0 = np.zeros(D)
Sig0 = np.eye(D)

p = [np.zeros((1, Z.shape[1]))]
ls = [np.array([0.])]


if nm in ['BPSVI']:
  pool = Pool(processes=100)
  res = pool.map(build_per_m, range(1, M+1))
  i=1
  for (wts, pts, idcs) in res:
    w.append(wts)
    pts = Y[idcs, np.newaxis]*pts
    p.append(pts)
    ls.append(Y[idcs])
    i+=1
else:
  for m in range(1, M+1):
    print('m = ', m)
    if nm != 'PRIOR':
      alg.build(1, N)
      #record weights
      if nm=='BCORES':
        wts, pts, idcs, beta = alg.get()
      else:
        wts, pts, idcs = alg.get()
      pts = Y[idcs, np.newaxis]*pts
      p.append(pts)
      ls.append(Y[idcs])
      print('selected groups info:', alg.selected_groups, [demos[selgroup] for selgroup in alg.selected_groups])
    else:
      p.append(np.zeros((1,D)))

N_per = 1000

accs = np.zeros(M+1)
pll = np.zeros(M+1)

print('Evaluation')
ssize=500
for m in range(1,M+1):
  print('selected cx with shape : ', p[m][:, :-1].shape, ' and weights', w[m])
  # subsample for MCMC
  #ridx = np.random.choice(range(p[m][:, :-1].shape[0]), size=min(ssize, p[m][:, :-1].shape[0]))
  #cx, cy = p[m][:, :-1][ridx,:], ls[m].astype(int)[ridx]
  #sampler_data = {'x': cx, 'y': cy, 'd': cx.shape[1], 'N': cx.shape[0], 'w': np.ones(w[m][ridx].shape[0])}
  cx, cy = p[m][:, :-1], ls[m].astype(int)
  cy[cy==-1] = 0
  sampler_data = {'x': cx, 'y': cy, 'd': cx.shape[1], 'N': cx.shape[0], 'w': np.ones(w[m].shape[0])}
  thd = sampler_data['d']+1
  fit = sml.sampling(data=sampler_data, iter=N_per*2, chains=1, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=False)
  thetas = np.roll(fit.extract(permuted=False)[:, 0, :thd], -1)
  accs[m]= compute_accuracy(Xt, Yt, thetas)
  pll[m] = np.sum(log_likelihood(Yt[:, np.newaxis]*Xt, thetas))/float(Xt.shape[0]*thetas.shape[0])
print('accuracies : ', accs)
print('pll : ', pll)

#save results
f = open('/home/dm754/rds/hpc-work/zellner_logreg/group_results/'+dnm+'_'+nm+'_'+str(f_rate)+'_results_'+str(ID)+'.pk', 'wb')
res = (p, accs, pll)
pk.dump(res, f)
f.close()
