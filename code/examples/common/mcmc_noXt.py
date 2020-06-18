import numpy as np
import pystan
from stan_code import logistic_code, poisson_code
import os
import pickle as pk
import time

def load_data(dnm):
  data = np.load(dnm)
  X = data['X']
  Y = data['y']
  #standardize the covariates; last col is intercept, so no stdization there
  m = X[:, :-1].mean(axis=0)
  V = np.cov(X[:, :-1], rowvar=False)+1e-12*np.eye(X.shape[1]-1)
  X[:, :-1] = np.linalg.solve(np.linalg.cholesky(V), (X[:, :-1] - m).T).T
  data.close()
  return X[:, :-1], Y

def sampler(dnm, lr, datafldr, resfldr, N_samples, subset_size=5000, n_cores=8):
  print('STAN: loading data')
  X, Y = load_data(os.path.join(datafldr,dnm+'.npz'))
  Y[Y == -1] = 0 #convert to Stan LR label style if necessary
  idcs = np.random.randint(0, X.shape[0], subset_size)
  sampler_data = {'x': X[idcs], 'y': Y[idcs].astype(int), 'd': X.shape[1], 'n': subset_size} #X.shape[0]}
  print('STAN: building/loading model')
  if lr:
    if not os.path.exists(os.path.join(resfldr,'pystan_model_logistic.pk')): 
      print('STAN: building LR model')
      extra_compile_args = ['-pthread', '-DSTAN_THREADS']
      sm = pystan.StanModel(model_code=logistic_code,
    		extra_compile_args=extra_compile_args)
      f = open(os.path.join(resfldr,'pystan_model_logistic.pk'),'wb')
      pk.dump(sm, f)
      f.close()
    else:
      f = open(os.path.join(resfldr,'pystan_model_logistic.pk'),'rb')
      sm = pk.load(f)
      f.close()
  print('STAN: sampling posterior: ' + dnm)
  t0 = time.process_time()
  thd = sampler_data['d']+1
  fit = sm.sampling(data=sampler_data, iter=N_samples*2, chains=1, n_jobs=n_cores,  control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=True)
  samples = fit.extract(permuted=False)[:, 0, :thd]
  np.save(os.path.join(resfldr, dnm+'_samples.npy'), samples) 
  tf = time.process_time()
  np.save(os.path.join(resfldr, dnm+'_mcmc_time.npy'), tf-t0)

