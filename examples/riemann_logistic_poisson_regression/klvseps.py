from __future__ import print_function
import numpy as np
import scipy.linalg as sl
import pickle as pk
import os, sys
from scipy.optimize import minimize, nnls
import time
from multiprocessing import Pool

# linearize for slurm job array submission
def linearize():
  args_dict = dict()
  c = -1
  for stan_samples in [True]: #, False]:
    for samplediag in [True]: #, False]:
      for graddiag in [True]:
        for kldiag in [True]: #, False]:
          for down in [1.]: #[1., 0.1, 0.01]:
            for lrsch in ['lin']:
              for i0 in [10.]: #[0.1, 1., 10., 100.]:
                for ID in range(5):
                  for alg in ['DPBPSVI']:
                    for dnm in ['fma']: # , 'ds1.100']: # 'mnist2class_test']:
                      for nmult in [0.4, 0.7, 1., 2., 5., 20., 70.]:
                        for M in [10, 60, 100]:
                          c += 1
                          args_dict[c] = (stan_samples, samplediag, graddiag, kldiag, down, lrsch, i0, ID, alg, dnm, nmult, M)
  return args_dict



mapping = linearize()
stan_samples, samplediag, graddiag, kldiag, down, lrsch, i0, ID, alg, dnm, nmult, M = mapping[int(sys.argv[1])]
#stan_samples, samplediag, graddiag, kldiag, down, lrsch, i0, ID, alg, dnm, nmult, M = mapping[0]
if True:
    #for i, (stan_samples, samplediag, graddiag, kldiag, down, lrsch, i0, ID, alg, dnm, nmult, M) in mapping.items():
    hpc = True
    if hpc: sys.path.insert(1, os.path.join(sys.path[0], '/home/dm754/bayesian-coresets-private'))
    import bayesiancoresets as bc
    #make it so we can import models/etc from parent folder
    sys.path.insert(1, os.path.join(sys.path[0], '../common'))
    from mcmc_noXt import sampler
    import gaussian

    #computes the Laplace approximation N(mu, Sig) to the posterior with weights wts
    def get_laplace(wts, Z, mu0, diag=False):
      trials = 10
      Zw = Z[wts>0, :]
      ww = wts[wts>0]
      while True:
        try:
          res = minimize(lambda mu : -log_joint(Zw, mu, ww)[0], mu0, jac=lambda mu : -grad_th_log_joint(Zw, mu, ww)[0,:])
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

    np.random.seed(int(ID))
    lrdnms = ['santa100K', 'synth_lr', 'phishing', 'ds1', 'synth_lr_large', 'phishing_large', 'ds1_large', 'ds1.100', 'mnist2class_test', 'fma']
    largescale = ['ds1.100', 'mnist2class_test', 'fma', 'bank']
    from model_lr import *
    print('running ' + str(dnm)+ ' ' + str(alg)+ ' ' + str(ID))
    ###############################
    ## TUNING PARAMETERS ##
    if lrsch=='lin':
        SVI_step_sched = lambda itr : i0/(1.+itr)
        BPSVI_step_sched = lambda m: lambda itr : i0/(1.+itr) # make step schedule potentially dependent on coreset size
    elif lrsch=='sqrt':
        SVI_step_sched = lambda itr: i0/np.sqrt(1.+itr)
        BPSVI_step_sched = lambda m: lambda itr: i0/np.sqrt(1.+itr)
    n_subsample_opt = 400
    n_subsample_select = 1000
    projection_dim = 200 #random projection dimension
    pihat_noise = .75 #noise level (relative) for corrupting pihat
    SVI_opt_itrs = 500
    BPSVI_opt_itrs = 500
    DPBPSVI_step_sched = BPSVI_step_sched
    DPBPSVI_opt_itrs = 500
    q = 0.01 # subsampling ration for DP-PSVI batches
    ###############################
    print('Loading dataset '+dnm)
    Z, Zt, D = load_data('../data/'+dnm+'.npz')
    if not os.path.exists('results/'):
      os.mkdir('results')
    dp_n_subsample_opt = int(q*Z.shape[0])
    print('sample size :', dp_n_subsample_opt)

    def gen_inits(m, thetas):
      np.random.seed(m)
      X = np.random.multivariate_normal(mu0, Sig0, m)
      ps = 1.0/(1.0+np.exp(-(X*thetas).sum(axis=1)))
      y = (np.random.rand(m) <= ps).astype(int)
      y[y==0] = -1
      return y[:, np.newaxis]*X

    N_samples=10000
    if not stan_samples: # use laplace approximation (save, load from results/)
      if not os.path.exists('results/'+dnm+'_samples_gauss_diag'+str(samplediag)+'.npy'):
        print('sampling using laplace')
        mup_laplace, LSigp_laplace, LSigpInv_laplace = get_laplace(np.ones(Z.shape[0]), Z, Z.mean(axis=0)[:D], diag=samplediag)
        samples_laplace = mup_laplace + np.random.randn(N_samples, mup_laplace.shape[0]).dot(LSigp_laplace.T)
        np.save(os.path.join('results/'+dnm+'_samples_gauss_diag'+str(samplediag)+'.npy'), samples_laplace)
      else:
        print('Loading posterior samples for '+dnm)
      samples = np.load('results/'+dnm+'_samples_gauss_diag'+str(samplediag)+'.npy', allow_pickle=True)
    else: # use stan sampler (save, load from results/pystan_samples/)
      if not os.path.exists('results/pystan_samples/'+dnm+'_samples.npy'):
        print('No MCMC samples found -- running STAN')
        sampler(dnm, dnm in lrdnms, '../data/', 'results/pystan_samples/', N_samples)
      else:
        print('Loading posterior samples for '+dnm)
      samples = np.load('results/pystan_samples/'+dnm+'_samples.npy', allow_pickle=True)

    #TODO FIX SAMPLER TO NOT HAVE TO DO THIS
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

    def sampler_w(sz, w, pts, diag=graddiag):
      if pts.shape[0] == 0:
        w = np.zeros(1)
        pts = np.zeros((1, Z.shape[1]))
      muw, LSigw, LSigwInv = get_laplace(w, pts, mu0, diag)
      return muw + np.random.randn(sz, muw.shape[0]).dot(LSigw.T)

    n_subsample_opt = 200
    projection_dim = 100 #random projection dimension for Hilbert csts
    prj_w = bc.BlackBoxProjector(sampler_w, projection_dim, log_likelihood, grad_z_log_likelihood)

    #create coreset construction objects
    t0 = time.perf_counter()
    dpbpsvi = bc.DiffPrivBatchPSVICoreset(Z, prj_w, opt_itrs=DPBPSVI_opt_itrs, n_subsample_opt=dp_n_subsample_opt, step_sched=DPBPSVI_step_sched,
                                       init_sampler=sampler_w, gen_inits=gen_inits, noise_multiplier=nmult, l2normclip=100, down=down)
    dpbpsvi_t_setup = time.perf_counter()-t0

    algs= {'DPBPSVI': dpbpsvi,
            'PRIOR': None}
    coreset = algs[alg]
    t0s = {'DPBPSVI': dpbpsvi_t_setup,
           'PRIOR' : 0.}

    print('Building coresets via ' + alg)
    def build_per_m(m): # consturction in parallel for different coreset sizes used in BPSVI
      t0 = time.perf_counter()
      #print('building coreset with', m, 'points')
      coreset.build(1, m)
      return coreset.get()
    res = build_per_m(M)
    w=res[0]
    p=res[1]
    cputs=None

    #get laplace approximations for each weight setting, and KL divergence to full posterior laplace approx mup Sigp
    #used for a quick/dirty performance comparison without expensive posterior sample comparisons (e.g. energy distance)
    print('Computing coreset Laplace approximation + approximate KL(posterior || coreset laplace)')
    mul, LSigl, LSiglInv = get_laplace(w, p, Z.mean(axis=0)[:D], diag=kldiag)
    mus_laplace = mul
    Sigs_laplace = LSigl.dot(LSigl.T)
    rkls_laplace = gaussian.gaussian_KL(mul, Sigs_laplace, mup, LSigpInv.dot(LSigpInv.T))
    fkls_laplace = gaussian.gaussian_KL(mup, LSigpInv.dot(LSigpInv.T), mul, Sigs_laplace)
    print('rkls_laplace :', M, rkls_laplace)
    #save results
    f = open('results/'+str(M)+'_'+str(nmult)+'_'+dnm+'_'+alg+lrsch+str(i0)+'_'+str(samplediag)+'_'+str(graddiag)+'_'+str(kldiag)+'_'+str(down)+'grad_stan_'+str(stan_samples)+str(ID)+'.pk', 'wb')
    res = (cputs, w, p, mus_laplace, Sigs_laplace, rkls_laplace, fkls_laplace, coreset.get_privacy_params())
    pk.dump(res, f)
    f.close()
