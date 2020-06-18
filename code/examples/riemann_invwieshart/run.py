import numpy as np
import scipy.linalg as sl
import pickle as pk
import os, sys
from scipy.stats import invwishart, multivariate_normal
np.set_printoptions(precision=2)
hpc=True
if hpc: sys.path.insert(1, os.path.join(sys.path[0], '/home/dm754/bayesian-coresets-private'))
import bayesiancoresets as bc
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
import iwg
import time
from multiprocessing import Pool
from scipy import random, linalg

# linearize for slurm job array submission
def linearize():
  args_dict = dict()
  c = -1
  for ID in range(1):
    for d in [60, 80, 150]:
      for alg in ['BPSVI']: #['RAND', 'PRIOR', 'GIGAO', 'GIGAR']:#['BPSVI']:
        for i0 in [1., 10., 100.]:
          c +=1
          args_dict[c] = (alg, ID, d, i0)
  return args_dict

def generate_psd(d=10):
  A = random.rand(d,d)
  return np.dot(A,A.T)

def is_pos_def(x):
  return np.all(np.linalg.eigvals(x) >= 0)

mapping = linearize()
(alg, ID, d, i0) = mapping[int(sys.argv[1])]
#for i, (alg, ID, d, i0) in mapping.items():
if True:
  M = 100
  N = 1000
  effr = 50
  SVI_opt_itrs = 200
  BPSVI_opt_itrs = 500
  n_subsample_opt = 200
  n_subsample_select = 1000
  projection_dim = 200
  pihat_noise = 0.75
  BPSVI_step_sched = lambda m: lambda i : i0/(1+i) #(1.1-0.005*m)/(1+i)
  SVI_step_sched = lambda i : 1./(1+i)

  np.random.seed(int(ID))

  nu0 = 2*d+1 # prior dof
  Psi0 = np.eye(d) # prior scale
  mu = np.zeros(d) # observations mean
  Sigma = 0.01*generate_psd(d)#10*np.eye(d) # observations covariance (to be inferred)
  Sigma[:effr,:effr] = 10*np.eye(effr)
 
  x = np.random.multivariate_normal(mu, Sigma, N) # observations

  print('Computing true posterior')
  nup = N + nu0 # posterior dof
  #Psip = Psi0 + N*np.cov((x-mu).T)# posterior scale
  Psip = Psi0 + np.dot((x-mu).T, x-mu)
  if not(is_pos_def(Psip)):
    print('posterior covariance not pos semi-definite')
    input()

  #create the log_likelihood function
  print('Creating (grad)log-likelihood functions')
  loglik = lambda x, Sig : iwg.gaussian_for_IW_loglikelihood(x, Sig, mu)
  grad_log_likelihood = lambda x, Sig: iwg.gaussian_gradx_loglikelihood(x, mu, Sig)

  def sampler_w(sz, w, x):
    if x.shape[0]==0:
      return invwishart.rvs(df=nu0, scale=Psi0, size=sz)
    else:
      incr = np.dot(w*((x-mu).T), x-mu)
      return invwishart.rvs(df=nu0+w.sum(), scale=Psi0+incr, size=sz)

  print('Building projectors')
  sampler_optimal = lambda sz,w,x: invwishart.rvs(df=nup, scale=Psip, size=sz)
  U = np.random.rand()
  sampler_realistic = lambda sz,w,x: invwishart.rvs(df=U*nup + (1-U)*nu0, scale=U*Psip+(1.-U)*Psi0, size=sz)

  prj_optimal = bc.BlackBoxProjector(sampler_optimal, projection_dim, loglik, grad_log_likelihood)
  prj_realistic = bc.BlackBoxProjector(sampler_realistic, projection_dim, loglik, grad_log_likelihood)
  prj_w = bc.BlackBoxProjector(sampler_w, projection_dim, loglik, grad_log_likelihood)


  print('Creating coresets object')
  #create coreset construction objects

  t0 = time.perf_counter()
  giga_optimal = bc.HilbertCoreset(x, prj_optimal)
  gigao_t_setup = time.perf_counter()-t0

  t0 = time.perf_counter()
  giga_realistic = bc.HilbertCoreset(x, prj_realistic)
  gigar_t_setup = time.perf_counter()-t0

  t0 = time.perf_counter()
  unif = bc.UniformSamplingCoreset(x)
  unif_t_setup = time.perf_counter()-t0

  t0 = time.perf_counter()
  sparsevi = bc.SparseVICoreset(x, prj_w, opt_itrs=SVI_opt_itrs, n_subsample_opt=n_subsample_opt, n_subsample_select=n_subsample_select, step_sched=SVI_step_sched)
  sparsevi_t_setup = time.perf_counter()-t0

  t0 = time.perf_counter()
  bpsvi = bc.BatchPSVICoreset(x, prj_w, opt_itrs=BPSVI_opt_itrs, n_subsample_opt=n_subsample_opt, step_sched=BPSVI_step_sched)
  bpsvi_t_setup = time.perf_counter()-t0


  algs = {'BPSVI' : bpsvi,
          'SVI': sparsevi,
          'GIGAO': giga_optimal,
          'GIGAR': giga_realistic,
          'RAND': unif,
          'PRIOR': None}
  coreset = algs[alg]
  t0s = {'SVI' : sparsevi_t_setup,
         'BPSVI' : bpsvi_t_setup,
         'GIGAO' : gigao_t_setup,
         'GIGAR' : gigar_t_setup,
         'RAND' : unif_t_setup,
         'PRIOR' : 0.}

  print('Building coreset')
  w = [np.array([0.])]
  p = [np.zeros((1, x.shape[1]))]
  cputs = np.zeros(M+1)
  cputs[0] = t0s[alg]

  def build_per_m(m): # construction in parallel for different coreset sizes used in BPSVI
    print('m=', m)
    t0 = time.perf_counter()
    coreset.build(1, m)
    print('built for m=', m)
    return coreset.get(), time.perf_counter()-t0

  # construct differently per algorithm for size>1
  if alg=='BPSVI':
      pool = Pool(processes=100)
      res = pool.map(build_per_m, range(1, M+1))
      i=1
      for (wts, pts, _), cput in res:
        w.append(wts)
        p.append(pts)
        cputs[i] = cput
        i+=1
  else:
      for m in range(1,M+1):
        print('m=', m)
        if alg != 'PRIOR':
          t0 = time.perf_counter()
          coreset.build(1, m)
          cputs[m] = time.perf_counter()-t0
          #record time and weights
          wts, pts, idcs = coreset.get()
          w.append(wts)
          p.append(pts)
        else:
          w.append(np.array([0.]))
          p.append(np.zeros((1, x.shape[1])))


  # computing kld and saving results
  nuw = np.zeros((M+1, 1))
  Psiw = np.zeros((M+1,d,d))
  rklw = np.zeros(M+1)
  fklw = np.zeros(M+1)
  if not(alg=='PRIOR'):
    for m in range(1,M+1):
      nuw[m] = nu0 + w[m].sum()
      if nuw[m]<0:
        print('negative weights')
        input()
      #incr = M*np.cov((p[m]-mu).T, aweights=w[m]) #not correct
      incr = np.dot(w[m]*((p[m]-mu).T), p[m]-mu)
      Psiw[m] = Psi0  + incr
      if not(is_pos_def(Psiw)):
        print('Psiw not positive semidefinite')
        input()
      rklw[m] = iwg.KL_IW(nuw[m], Psiw[m], nup, Psip)
      fklw[m] = iwg.KL_IW(nup, Psip, nuw[m], Psiw[m])
  else:
    for m in range(1,M+1):
      nuw[m] = nu0
      Psiw[m] = Psi0
      if not is_pos_def(Psiw[m]):
        print('psiw error at m=',m)
        input()
      if not is_pos_def(Psip):
        print('psip error at m=',m)
        input()
      print('rklw for m=',m)
      rklw[m] = iwg.KL_IW(nuw[m], Psiw[m], nup, Psip)
      fklw[m] = iwg.KL_IW(nup, Psip, nuw[m], Psiw[m])
  rklw[0] = iwg.KL_IW(nu0, Psi0, nup, Psip)
  fklw[0] = iwg.KL_IW(nup, Psip, nu0, Psi0)  
  print('\n\nalg = ', alg, '\nrkl: ', rklw)


  #save results
  f = open('/home/dm754/rds/hpc-work/iw_results/TIWG_'+alg+'_results_'+'d_'+str(d)+'_i0_'+str(i0)+'_'+str(ID)+'.pk', 'wb')
  res = (cputs, w, p, rklw, nup, Psip)
  pk.dump(res, f)
  f.close()




