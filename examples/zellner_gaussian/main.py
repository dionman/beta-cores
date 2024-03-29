import numpy as np
import pickle as pk
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../..')) # read library from local folder: can be removed if it's installed systemwide
import bayesiancoresets as bc
from scipy.stats import multivariate_normal
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
import gaussian
from gaussian import *

nm = sys.argv[1]
tr = sys.argv[2]
np.random.seed(int(tr))

results_fldr = 'results'
if not os.path.exists(results_fldr):
  os.mkdir(results_fldr)

M = 200 # max coreset sz
SVI_opt_itrs = 1000
BPSVI_opt_itrs = 1000
BCORES_opt_itrs = 1000
n_subsample_opt = 200
n_subsample_select = 1000
proj_dim = 200
pihat_noise = 0.75
i0 = 0.1 # starting learning rate
BPSVI_step_sched = lambda m: lambda i : i0/(1.+i)
SVI_step_sched = lambda i : i0/(1.+i)
BCORES_step_sched = lambda i : i0/(1.+i)

N = 5000  # number of data points
d = 100  # number of dimensions

mu0 = np.zeros(d)
Sig0 = np.eye(d)
Sig = 500*np.eye(d)
SigL = np.linalg.cholesky(Sig)
th = np.zeros(d)
Sig0inv = np.linalg.inv(Sig0)
Siginv = np.linalg.inv(Sig)
SigLInv = np.linalg.inv(SigL)
logdetSig = np.linalg.slogdet(Sig)[1]
X = np.random.multivariate_normal(th, Sig, N)

mup, LSigp, LSigpInv = gaussian.weighted_post(mu0, Sig0inv, Siginv, X, np.ones(X.shape[0])) # true posterior
Sigp = LSigp.dot(LSigp.T)
SigpInv = LSigpInv.dot(LSigpInv.T)

Xoutliers1 = np.random.multivariate_normal(th+200, 0.5*Sig, int(N/50.))
Xoutliers2 = np.random.multivariate_normal(th+150, 0.1*Sig, int(N/50.))
Xoutliers3 = np.random.multivariate_normal(th, 10*Sig, int(N/10.))
Xcorrupted = np.concatenate((X, Xoutliers1, Xoutliers2, Xoutliers3))

#create function to output log_likelihood given param   samples
print('Creating log-likelihood function')
log_likelihood = lambda x, th : gaussian_loglikelihood(x, th, Siginv, logdetSig)

print('Creating gradient log-likelihood function')
grad_log_likelihood = lambda x, th : gaussian_grad_x_loglikelihood(x, th, Siginv)

print('Creating beta likelihood function')
beta_likelihood = lambda x, th, beta : gaussian_beta_likelihood(x, th, beta, Siginv, logdetSig)

print('Creating gradient grad beta function')
grad_beta = lambda x, th, beta : gaussian_beta_gradient(x, th, beta, Siginv, logdetSig)

#create tangent space for well-tuned Hilbert coreset alg
print('Creating tuned projector for Hilbert coreset construction')
sampler_optimal = lambda n, w, pts : mup + np.random.randn(n, mup.shape[0]).dot(LSigp.T)
prj_optimal = bc.BlackBoxProjector(sampler_optimal, proj_dim, log_likelihood, grad_log_likelihood)

#create tangent space for poorly-tuned Hilbert coreset alg
print('Creating untuned projector for Hilbert coreset construction')
U = np.random.rand()
muhat = U*mup + (1.-U)*mu0
Sighat = U*Sigp + (1.-U)*Sig0
#now corrupt the smoothed pihat
muhat += pihat_noise*np.sqrt((muhat**2).sum())*np.random.randn(muhat.shape[0])
Sighat *= np.exp(-2*pihat_noise*np.fabs(np.random.randn()))
LSighat = np.linalg.cholesky(Sighat)
sampler_realistic = lambda n, w, pts : mup + np.random.randn(n, mup.shape[0]).dot(LSighat.T)
prj_realistic = bc.BlackBoxProjector(sampler_realistic, proj_dim, log_likelihood, grad_log_likelihood)

print('Creating black box projector for sampling from coreset posterior')
def sampler_w(sz, wts, pts, diag=False):
  if pts.shape[0] == 0:
    wts = np.zeros(1)
    pts = np.zeros((1, Xcorrupted.shape[1]))
  muw, LSigw, LSigwInv = weighted_post(mu0, Sig0inv, Siginv, pts, wts)
  return muw + np.random.randn(sz, muw.shape[0]).dot(LSigw.T)

prj_w = bc.BlackBoxProjector(sampler_w, proj_dim, log_likelihood, grad_log_likelihood)
prj_bw = bc.BetaBlackBoxProjector(sampler_w, proj_dim, beta_likelihood, log_likelihood, grad_beta)

#create coreset construction objects
print('Creating coreset construction objects')
sparsevi = bc.SparseVICoreset(Xcorrupted, prj_w, opt_itrs = SVI_opt_itrs, n_subsample_opt = n_subsample_opt,
                              n_subsample_select = n_subsample_select, step_sched = SVI_step_sched)
bpsvi = bc.BatchPSVICoreset(Xcorrupted, prj_w, opt_itrs = BPSVI_opt_itrs, n_subsample_opt = n_subsample_opt,
                            step_sched = BPSVI_step_sched)
bcoresvi = bc.BetaCoreset(Xcorrupted, prj_bw, opt_itrs = BCORES_opt_itrs, n_subsample_opt = n_subsample_opt,
                           n_subsample_select = n_subsample_select, step_sched = BCORES_step_sched,
                           beta = .1, learn_beta=False)
giga_optimal = bc.HilbertCoreset(Xcorrupted, prj_optimal)
giga_realistic = bc.HilbertCoreset(Xcorrupted, prj_realistic)
unif = bc.UniformSamplingCoreset(Xcorrupted)

algs = {'BCORES': bcoresvi,
        'BPSVI': bpsvi,
        'SVI': sparsevi,
        'GIGAO': giga_optimal,
        'GIGAR': giga_realistic,
        'RAND': unif,
        'PRIOR': None}
alg = algs[nm]

print('Building coreset')
#build coresets
w = [np.array([0.])]
p = [np.zeros((1, Xcorrupted.shape[1]))]

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

# computing kld and saving results
muw = np.zeros((M+1, mu0.shape[0]))
Sigw = np.zeros((M+1,mu0.shape[0], mu0.shape[0]))
rklw = np.zeros(M+1)
fklw = np.zeros(M+1)
if nm=='BCORES': betas = np.zeros(M+1)
for m in range(M+1):
  muw[m, :], LSigw, LSigwInv = gaussian.weighted_post(mu0, Sig0inv, Siginv, p[m], w[m])
  Sigw[m, :, :] = LSigw.dot(LSigw.T)
  rklw[m] = gaussian.gaussian_KL(muw[m,:], Sigw[m,:,:], mup, SigpInv)
  fklw[m] = gaussian.gaussian_KL(mup, Sigp, muw[m,:], LSigwInv.dot(LSigwInv.T))
  if nm=='BCORES': betas[m] = beta

f = open('results/results_'+nm+'_'+str(tr)+'.pk', 'wb')
if nm=='BCORES':
  res = (Xcorrupted, mu0, Sig0, Sig, mup, Sigp, w, p, muw, Sigw, rklw, fklw, betas)
  print('betas : ', betas)
else:
  res = (Xcorrupted, mu0, Sig0, Sig, mup, Sigp, w, p, muw, Sigw, rklw, fklw)
print('rklw :', rklw)
pk.dump(res, f)
f.close()
