import numpy as np
import pickle as pk
import os, sys
from multiprocessing import Pool
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from model_lr import *
import pystan

# specify random number only for test size randomization (common across trials)
np.random.seed(42)
rnd = np.random.rand()
nm = "DShapley"
dnm = "diabetes"
ID = 0
structured = False
f_rate = 0.1
np.random.seed(int(ID))

flatten = lambda l: [item for sublist in l for item in sublist]

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

print('Loading dataset '+dnm)
X, Y, Xt, Yt = load_data('../data/'+dnm+'.npz') # read train and test data
X, Y, Z, x_mean, x_std = std_cov(X, Y) # standardize covariates
N, D = X.shape

f = open('../data/vq_groups_sensemake_'+str(dnm)+'.pk', 'rb')
res = pk.load(f) #(w, p, accs, pll)
f.close()
(groups, demos)=res
groups = [[k for k in g if k<Z.shape[0]] for g in groups]
grouptot = sum([len(g) for g in groups])

if f_rate>0:
  for (g,d) in zip(groups,demos):
    X[g,:], Y[g], Z[g,:], _ = perturb(X[g,:], Y[g], f_rate=0.*d[0]*f_rate, structured=structured, noise_x=(0,10))

# make sure test set is adequate for evaluation via the predictive accuracy metric
if len(Yt[Yt==1])>0.55*len(Yt) or len(Yt[Yt==1])<0.45*len(Yt): # truncate for balanced test dataset
  totrunc=-1 # totrunc holds the majority label to be truncated (for fairness of accuracy metric)
  if len(Yt[Yt==1])> len(Yt[Yt==-1]):
    totrunc=1
  idcs = ([i for i, e in enumerate(Yt) if e == totrunc][:len(Yt[Yt==-totrunc])+int(0.01*len(Yt[Yt==-totrunc])*rnd)]
         +[i for i, e in enumerate(Yt) if e == -totrunc])
  Xt, Yt = Xt[idcs,:], Yt[idcs]
Xt, Yt, _, _, _ = std_cov(Xt, Yt, mean_=x_mean, std_=x_std) # standardize covariates for test data

def update_per_t(t, maxGroups=10):
  phis = np.zeros(len(groups),) # initialize Shapley values for all groups to zero
  vs = np.zeros(len(groups)+1,) # values for group combinations for all groups to zero
  idcs = np.random.permutation(len(groups))
  for j in range(maxGroups):
    datapoints = flatten([groups[idx] for idx in idcs[:j]])
    vs[j+1] = eval(datapoints, X, Y, Xt, Yt)
    phis[idcs[j]] += (vs[j+1]-vs[j]) # add new marginal for group idcs[j]
  return phis

def dshapley(groups, X, Y, Xt, Yt, T=20):
  pool = Pool(processes=10)
  res = pool.map(update_per_t, range(T))
  phis = np.mean(res, axis=0)
  return phis

def eval(idcs, X, Y, Xt, Yt, N_per=1000):
  cx, cy = X[idcs][:, :-1], Y[idcs].astype(int)
  cy[cy==-1] = 0
  print('\n\n num of datapoints : ', cx.shape[0], '\n\n\n')
  sampler_data = {'x': cx, 'y': cy, 'd': cx.shape[1], 'N': cx.shape[0], 'w': np.ones(cx.shape[0])}
  thd = sampler_data['d']+1
  fit = sml.sampling(data=sampler_data, iter=N_per*2, chains=1, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=False)
  thetas = np.roll(fit.extract(permuted=False)[:, 0, :thd], -1)
  acc = compute_accuracy(Xt, Yt, thetas)
  return acc

phis = dshapley(groups, X, Y, Xt, Yt)
print(phis)

sort_index = np.argsort(phis)[::-1]

selected_groups = sort_index[:10]
print('selected groups info:', selected_groups, [demos[selgroup] for selgroup in selected_groups])
'''
#save results
f = open('/home/dm754/rds/hpc-work/zellner_logreg/group_results/'+dnm+'_'+nm+'_'+str(f_rate)+'_results_'+str(ID)+'.pk', 'wb')
res = (p, accs, pll)
pk.dump(res, f)
f.close()
'''
