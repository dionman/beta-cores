import numpy as np
import pickle as pk
import os, sys
from multiprocessing import Pool
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import bayesiancoresets as bc
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from mcmc import sampler
import gaussian
from scipy.optimize import minimize, nnls
import scipy.linalg as sl
from model_lr import *

riemann_coresets = ['BPSVI', 'SVI', 'BCORES']
nm = sys.argv[1]
dnm = sys.argv[2]
ID = sys.argv[3]
stan_samples = (sys.argv[4]=="True") # use stan for true posterior sampling
samplediag = (sys.argv[5]=="True") # diagonal Gaussian assumption for posterior sampling
graddiag = (sys.argv[6]=="True") # diagonal Gaussian assumption for coreset sampler
if nm in riemann_coresets: i0 = float(sys.argv[7])
np.random.seed(int(ID))

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
pihat_noise = .75 #noise level (relative) for corrupting pihat
SVI_opt_itrs = 500
BPSVI_opt_itrs = 500
BCORES_opt_itrs = 500
###############################

print('Loading dataset '+dnm)
Z, Zt, D = load_data('../data/'+dnm+'.npz')
if not os.path.exists('results/'):
  os.mkdir('results')

N_samples=10000
if not stan_samples: # use laplace approximation (save, load from results/)
  if not os.path.exists('results/'+dnm+'_samples.npy'):
    print('sampling using laplace')
    mup_laplace, LSigp_laplace, LSigpInv_laplace = get_laplace(np.ones(Z.shape[0]), Z, Z.mean(axis=0)[:D], diag=samplediag)
    samples_laplace = mup_laplace + np.random.randn(N_samples, mup_laplace.shape[0]).dot(LSigp_laplace.T)
    np.save(os.path.join('results/'+dnm+'_samples.npy'), samples_laplace)
  else:
    print('Loading posterior samples for '+dnm)
  samples = np.load('results/'+dnm+'_samples.npy', allow_pickle=True)
else: # use stan sampler (save, load from results/pystan_samples/)
  if not os.path.exists('results/'+dnm+'_samples.npy'):
    print('No MCMC samples found -- running STAN')
    sampler(dnm, True, '../data/', 'results/pystan_samples/', N_samples)
  else:
    print('Loading posterior samples for '+dnm)
  samples = np.load('results/'+dnm+'_samples.npy', allow_pickle=True)
samples = np.hstack((samples[:, 1:], samples[:, 0][:,np.newaxis]))

#fit a gaussian to the posterior samples
#used for pihat computation for Hilbert coresets with noise to simulate uncertainty in a good pihat
mup = samples.mean(axis=0)
Sigp = np.cov(samples, rowvar=False)
LSigp = np.linalg.cholesky(Sigp)
LSigpInv = sl.solve_triangular(LSigp, np.eye(LSigp.shape[0]), lower=True, overwrite_b=True, check_finite=False)
print('posterior fitting done')

#create the prior -- also used for the above purpose
mu0 = np.zeros(mup.shape[0])
Sig0 = np.eye(mup.shape[0])

#get pihat via interpolation between prior/posterior + noise
#uniformly smooth between prior and posterior
U = np.random.rand()
muhat = U*mup + (1.-U)*mu0
Sighat = U*Sigp + (1.-U)*Sig0
#now corrupt the smoothed pihat
muhat += pihat_noise*np.sqrt((muhat**2).sum())*np.random.randn(muhat.shape[0])
Sighat *= np.exp(-2.*pihat_noise*np.fabs(np.random.randn()))
LSighat = np.linalg.cholesky(Sighat)

print('Building projectors')
sampler_optimal = lambda sz, w, pts : mup + np.random.randn(sz, mup.shape[0]).dot(LSigp.T)
sampler_realistic = lambda sz, w, pts : muhat + np.random.randn(sz, muhat.shape[0]).dot(LSighat.T)
def sampler_w(sz, w, pts, diag=graddiag):
  if pts.shape[0] == 0:
    w = np.zeros(1)
    pts = np.zeros((1, Z.shape[1]))
  muw, LSigw, LSigwInv = get_laplace(w, pts, mu0, diag)
  return muw + np.random.randn(sz, muw.shape[0]).dot(LSigw.T)

grad_beta = lambda x, th, beta : gaussian_beta_gradient(x, th, beta, Siginv, logdetSig)


prj_optimal = bc.BlackBoxProjector(sampler_optimal, projection_dim, log_likelihood, grad_z_log_likelihood)
prj_realistic = bc.BlackBoxProjector(sampler_realistic, projection_dim, log_likelihood, grad_z_log_likelihood)
prj_w = bc.BlackBoxProjector(sampler_w, projection_dim, log_likelihood, grad_z_log_likelihood)
prj_bw = bc.BetaBlackBoxProjector(sampler_w, projection_dim, beta_likelihood, beta_likelihood, grad_beta)

print('Creating coresets object')
#create coreset construction objects

unif = bc.UniformSamplingCoreset(Z)
sparsevi = bc.SparseVICoreset(Z, prj_w, opt_itrs=SVI_opt_itrs, n_subsample_opt = n_subsample_opt,
                              n_subsample_select = n_subsample_select, step_sched = SVI_step_sched)
bpsvi = bc.BatchPSVICoreset(Z, prj_w, opt_itrs = BPSVI_opt_itrs, n_subsample_opt = n_subsample_opt,
                            step_sched = BPSVI_step_sched, mup=mup, SigpInv=LSigpInv.dot(LSigpInv.T))
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

def build_per_m(m): # construction in parallel for different coreset sizes used in BPSVI
  coreset.build(1, m)
  return coreset.get()

if nm in ['BPSVI']:
  pool = Pool(processes=100)
  res = pool.map(build_per_m, range(1, M+1))
  i=1
  for (wts, pts, _) in res:
    w.append(wts)
    p.append(pts)
    i+=1
else:
  for m in range(1, M+1):
    if nm != 'PRIOR':
      alg.build(1, m)
      #record   weights
      if nm=='BCORES':
        wts, pts, idcs, beta = alg.get()
        print(alg.get())
      else:
        wts, pts, idcs = alg.get()
      w.append(wts)
      p.append(pts)
    else:
      w.append(np.array([0.]))
      p.append(np.zeros((1, Z.shape[1])))

#get laplace approximations for each weight setting, and KL divergence to full posterior laplace approx mup Sigp
#used for a quick/dirty performance comparison without expensive posterior sample comparisons (e.g. energy distance)
mus_laplace = np.zeros((M+1, D))
Sigs_laplace = np.zeros((M+1, D, D))
pred_accuracy = np.zeros(M+1)
print('Evaluation')
'''
for m in range(M+1):
  # Sample from coreset posterior
  mul, LSigl, LSiglInv = get_laplace(w[m], p[m], Z.mean(axis=0)[:D], diag=True)
  mus_laplace[m,:] = mul
  Sigs_laplace[m,:,:] = LSigl.dot(LSigl.T)
  thetas = mul + np.random.randn(sz, mul.shape[0]).dot(LSigl.T)
  # Evaluate on test datapoints
  y_pred_samples = self.forward(x, num_samples=100)
  y_pred = self._compute_predictive_posterior(y_pred_samples)[None, :, :]

  log_pred_samples = y_pred
  L = utils.to_gpu(torch.FloatTensor([log_pred_samples.shape[0]]))
  preds = torch.logsumexp(log_pred_samples, dim=0) - torch.log(L)
  if not logits:
    preds = torch.softmax(preds, dim=-1)
  loss = self._compute_log_likelihood(y, y_pred)  # use predictive at test time
  avg_loss = loss / len(x)
  performance = self._evaluate_performance(y, y_pred_samples)
  losses.append(avg_loss.cpu().item())
  performances.append(performance.cpu().item())

#save results
f = open('results/'+dnm+'_'+alg+'_results_'+ID+'.pk', 'wb')
res = (w, p, mus_laplace, Sigs_laplace, rkls_laplace, fkls_laplace)
pk.dump(res, f)
f.close()
'''
