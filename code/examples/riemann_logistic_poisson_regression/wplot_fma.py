import numpy as np
import pickle as pk
import time
import os, sys
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '/Users/dion.ysis/private-bayesian-coresets/code/bayesian-coresets-private/examples/common/'))
from plotting import *
from bokeh.io import export_png, export_svgs
import bokeh.plotting as bkp
import bokeh.layouts as bkl
import scipy.linalg as sl
from bokeh.models import (ColumnDataSource, LinearAxis, Plot, Text)
from bokeh.transform import dodge
np.set_printoptions(precision=2)

#computes the Laplace approximation N(mu, Sig) to the posterior with weights wts
def get_laplace(wts, Z, mu0, diag = False):
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
    LSigInv = np.sqrt(-diag_hess_th_log_joint(Zw, mu, ww)[0,:])
    LSig = 1./LSigInv
  else:
    LSigInv = np.linalg.cholesky(-hess_th_log_joint(Zw, mu, ww)[0,:,:])
    LSig = sl.solve_triangular(LSigInv, np.eye(LSigInv.shape[0]), lower=True, overwrite_b=True, check_finite=False)
  return mu, LSig, LSigInv


def gaussian_KL(mu0, Sig0, mu1, Sig1inv):
  t1 = np.dot(Sig1inv, Sig0).trace()
  t2 = np.dot((mu1-mu0),np.dot(Sig1inv, mu1-mu0))
  t3 = -np.linalg.slogdet(Sig1inv)[1] - np.linalg.slogdet(Sig0)[1]
  return 0.5*(t1+t2+t3-mu0.shape[0])

dnames = ['fma']
algs = [('BPSVI','PSVI', pal[-1]), ('DPBPSVI','DP-PSVI', pal[-2]), ('SVI', 'SparseVI', pal[0]), ('RAND', 'Uniform', pal[3]), ('GIGAO','GIGA (Optimal)', pal[1]), ('GIGAR','GIGA (Realistic)', pal[2])]
res_fldr = 'wlargescale/'
figs = []
fldr_figs = 'figs'
if not os.path.exists(fldr_figs):
  os.mkdir(fldr_figs)

for dnm in dnames:
  #fit a gaussian to the posterior samples
  samples = np.load('posteriorsamples/'+dnm+'_samples.npy')
  mup = samples.mean(axis=0)
  Sigp = np.cov(samples, rowvar=False)
  LSigp = np.linalg.cholesky(Sigp)
  LSigpInv = sl.solve_triangular(LSigp, np.eye(LSigp.shape[0]), lower=True, overwrite_b = True, check_finite = False)
  print('posterior fitting done')

  mu0 = np.zeros(mup.shape[0])
  Sig0 = np.eye(mup.shape[0])

  print('Plotting ' + dnm)
  fig = bkp.figure(y_axis_type='log', y_axis_label='Reverse KL',  x_axis_label='# Iterations', plot_height=1500, plot_width=2000, toolbar_location=None)
  preprocess_plot(fig, '72pt', False, True)
  fig2 = bkp.figure(y_axis_type='log', y_axis_label='', x_axis_type='log', x_axis_label='CPU Time (s)', plot_height=1500, plot_width=2000, toolbar_location=None)
  preprocess_plot(fig2, '72pt', True, True)
  fig3 = bkp.figure(y_axis_type='log', y_axis_label='',  x_axis_label='Coreset Size', plot_height=1500, plot_width=2000, toolbar_location=None)
  preprocess_plot(fig3, '72pt', False, True)
  fig4 = bkp.figure(y_axis_type='log', y_axis_label='', x_axis_type='log', x_axis_label='ε', plot_height=1500, plot_width=2000, toolbar_location=None)
  preprocess_plot(fig4, '72pt', True, True)
  figs.append([fig, fig2, fig3, fig4])

  #get normalizations based on the prior
  std_kls = {}
  exp_prfx = ''
  trials = [fn for fn in os.listdir(res_fldr) if fn.startswith('fma_PRIORsqrt10.0_True_True_True_1.0grad_stan_True')]
  if len(trials) == 0:
    print('Need to run prior to establish baseline first')
    quit()
  kltot = 0.
  M = 0
  for tridx, fn in enumerate(trials):
    f = open(res_fldr+fn, 'rb')
    res = pk.load(f) #(cputs, w, p, mus_laplace, Sigs_laplace, rkls_laplace, fkls_laplace)
    f.close()
    assert np.all(res[5] == res[5][0]) #make sure prior doesn't change...
    kltot += res[5][0]
    M = res[0].shape[0]
  std_kls[dnm] = kltot / len(trials)
  kl0=std_kls[dnm]


  for alg in algs:
    if  alg[0] == 'BPSVI':
      trials = [fn for fn in os.listdir(res_fldr) if fn.startswith('fma_BPSVIlin10.0_True_True_True_1.0grad_stan_True')]
    elif alg[0] == 'DPBPSVI':
      trials = [fn for fn in os.listdir(res_fldr) if fn.startswith('fma_DPBPSVIlin10.0_True_True_True_1.0grad_stan_True')]
    elif alg[0] == 'SVI':
      trials = [fn for fn in os.listdir(res_fldr) if fn.startswith('fma_SVIlin1.0_True_True_True_1.0grad_stan_True')]
    else:
      trials = [fn for fn in os.listdir(res_fldr) if fn.startswith('fma_'+alg[0]+'sqrt10.0_True_True_True_1.0grad_stan_True')]
    if len(trials) == 0:
      fig.line([], [], color=alg[2], legend=alg[1], line_width=10); fig.patch([], [], color=alg[2], legend=alg[1], alpha=0.3)
      fig2.line([], [], color=alg[2], legend=alg[1], line_width=10); fig2.patch([], [], color=alg[2], legend=alg[1], alpha=0.3)
      fig3.line([], [], color=alg[2], legend=alg[1], line_width=10); fig3.patch([], [], color=alg[2], legend=alg[1], alpha=0.3)
      fig4.line([], [], color=alg[2], legend=alg[1], line_width=10); fig4.patch([], [], color=alg[2], legend=alg[1], alpha=0.3)
      continue
    kls = np.zeros((len(trials), M))
    cputs = np.zeros((len(trials), M))
    cszs = np.zeros((len(trials), M))
    for tridx, fn in enumerate(trials):
      f = open(res_fldr+fn, 'rb')
      res = pk.load(f) #(cputs, w, p, mus_laplace, Sigs_laplace, rkls_laplace, fkls_laplace)
      f.close()
      cputs[tridx, :] = res[0]
      wts = res[1]
      pts = res[2]
      mu = res[3]
      Sig = res[4]
      kl = res[5]
      cszs[tridx, :] = np.array([len(w) for w in wts])
      kls[tridx, :] = kl/kl0
      if 'PRIOR' in fn:
        kls[tridx, :] = np.median(kls[tridx,:])


    if alg[0] in ['BPSVI', 'DPBPSVI']:
      cputs[:, 1:] += cputs[:, 0][:,np.newaxis]
    else:
      cputs = np.cumsum(cputs, axis=1)

    cput50 = np.percentile(cputs, 50, axis=0)
    cput25 = np.percentile(cputs, 25, axis=0)
    cput75 = np.percentile(cputs, 75, axis=0)

    csz50 = np.percentile(cszs, 50, axis=0)
    csz25 = np.percentile(cszs, 25, axis=0)
    csz75 = np.percentile(cszs, 75, axis=0)

    kl50 = np.percentile(kls, 50, axis=0)
    kl25 = np.percentile(kls, 25, axis=0)
    kl75 = np.percentile(kls, 75, axis=0)

    if alg[0]=='BPSVI':
        kl50bpsvi = np.copy(kl50)
        kl25bpsvi = np.copy(kl25)
        kl75bpsvi = np.copy(kl75)

    fig.line(np.arange(kl50.shape[0]), kl50, color=alg[2], legend=alg[1], line_width=10)
    fig.line(np.arange(kl25.shape[0]), kl25, color=alg[2], legend=alg[1], line_width=10, line_dash='dashed')
    fig.line(np.arange(kl75.shape[0]), kl75, color=alg[2], legend=alg[1], line_width=10, line_dash='dashed')
    # for computation time, don't show the coreset size 0 step since it's really really fast for all algs
    fig2.line(cput50[1:], kl50[1:], color=alg[2], legend=alg[1], line_width=10)
    fig2.patch(np.hstack((cput50[1:], cput50[1:][::-1])), np.hstack((kl75[1:], kl25[1:][::-1])), fill_color=alg[2], legend=alg[1], alpha=0.3)



    if alg[0] != 'PRIOR':
      fig3.line(csz50, kl50, color=alg[2], legend=alg[1], line_width=10)
      fig3.patch(np.hstack((csz50, csz50[::-1])), np.hstack((kl75, kl25[::-1])), fill_color=alg[2], legend=alg[1], alpha=0.3)
      fig3.legend.location='top_left'

  pal = bokeh.palettes.colorblind['Colorblind'][8]

  from bokeh.palettes import magma, inferno, viridis
  pal = inferno(9)[1:]
  x=[30]
  nmults =  [0.4, 0.7, 1., 2., 5., 20., 70.][::-1]
  for k, M in enumerate([10, 60, 100]):
      eps = np.zeros((len(nmults),))
      pkl50 = np.zeros((len(nmults),))
      pkl25 = np.zeros((len(nmults),))
      pkl75 = np.zeros((len(nmults),))
      for i, nmult in enumerate(nmults):
        trials = [fn for fn in os.listdir(res_fldr) if fn.startswith(str(M)+'_'+str(nmult)+'_'+'fma_DPBPSVIlin10.0_True_True_True_1.0grad_stan_True')]
        pkl = np.zeros((len(trials),))
        kl0 = std_kls[dnm]
        for tridx, fn in enumerate(trials):
          f = open(res_fldr+fn, 'rb')
          res = pk.load(f) #(cputs, w, p, mus_laplace, Sigs_laplace, rkls_laplace, fkls_laplace,..,eps)
          f.close()
          wts = res[1]
          mu = res[3]
          Sig = res[4]
          pkl_ = res[5]
          pkl[tridx] = pkl_/kl0
          eps[i] = res[-1][0]
        pkl50[i] = np.percentile(pkl, 50, axis=0)
        pkl25[i] = np.percentile(pkl, 25, axis=0)
        pkl75[i] = np.percentile(pkl, 75, axis=0)
      fig4.line(eps, pkl50, color=pal[k], legend='DP-PSVI (M='+str(M)+')', line_width=10)
      fig4.patch(np.hstack((eps, eps[::-1])), np.hstack((pkl75, pkl25[::-1])), fill_color=pal[k], legend='DP-PSVI (M='+str(M)+')', alpha=0.3)
      fig4.circle(x, kl50bpsvi[M], size=40, color=pal[k])
      fig4.line([x[0], x[0]], [kl25bpsvi[M], kl75bpsvi[M]], color=pal[k], line_width=5)


  #non-priv
  trials = [fn for fn in os.listdir(res_fldr) if fn.startswith('fmaDPVI_nonpriv_')]
  npadvis = np.zeros((len(trials),))
  for tridx, fn in enumerate(trials):
    dpvi_fnm = open(res_fldr+fn, 'rb')
    mu, Sigma = pk.load(dpvi_fnm, encoding='latin') # (mu, Sigma, eps)
    nonpriv_advi = gaussian_KL(mu, Sigma, mup, LSigpInv.dot(LSigpInv.T))
    npadvis[tridx] = nonpriv_advi/float(kl0)
    dpvi_fnm.close()
  nonpriv_advi50 = np.percentile(npadvis, 50)
  nonpriv_advi25 = np.percentile(npadvis, 25)
  nonpriv_advi75 = np.percentile(npadvis, 75)



  # read baselines results and compute DKL
  lensigmas=7
  n_ave = 5
  klsbaseline = np.zeros((lensigmas, n_ave))
  epslst=[]
  for k in range(1,lensigmas):
    kl=[]
    for i in range(n_ave):
      dpvi_fnm = open(res_fldr+'fmaDPVI_C5.0_'+str(k)+'_'+str(i)+'.pk', 'rb') #0.05, 0.5, 1.0, 5.0, 10.0, 100.0
      mu, Sigma, eps = pk.load(dpvi_fnm, encoding='latin') # (mu, Sigma, eps)
      dpvi_fnm.close()
      kl.append(gaussian_KL(mu, Sigma, mup, LSigpInv.dot(LSigpInv.T)))
    epslst.append(eps)
    klsbaseline[k, :] = kl/kl0
  klsbaseline = klsbaseline.transpose()


  kl50baseline = np.percentile(klsbaseline, 50, axis=0)
  kl25baseline = np.percentile(klsbaseline, 25, axis=0)
  kl75baseline = np.percentile(klsbaseline, 75, axis=0)


  # KLD vs eps plot for DPVI
  kl50baseline = kl50baseline[1:]
  fig4.line(epslst, kl50baseline, color=pal[-3], legend='DP-VI', line_width=10)
  kl75baseline = kl75baseline[1:]
  kl25baseline = kl25baseline[1:]
  fig4.patch(np.hstack((epslst, epslst[::-1])), np.hstack((kl75baseline, kl25baseline[::-1])), fill_color=pal[-3], legend='DP-VI', alpha=0.1)
  fig4.line([x[0], x[0]], [nonpriv_advi25, nonpriv_advi75], color=pal[-3], line_width=5)
  fig4.circle(x, nonpriv_advi50, size=40, color=pal[-3])

  for f in [fig, fig3]:
    f.legend.location='top_left'
    f.legend.label_text_font_size= '60pt'
    f.legend.glyph_width=50
    f.legend.glyph_height=50
    f.legend.spacing=10
    f.legend.visible = False
  for f in [fig2]:
    f.legend.location='bottom_center'
    f.legend.label_text_font_size= '60pt'
    f.legend.glyph_width=50
    f.legend.glyph_height=50
    f.legend.spacing=10
    f.legend.visible = False
  for f in [fig4]:
    f.legend.location='bottom_left'
    f.legend.label_text_font_size= '60pt'
    f.legend.glyph_width=50
    f.legend.glyph_height=50
    f.legend.spacing=10
    f.legend.visible = False

  export_png(fig, filename=os.path.join(fldr_figs, 'w'+dnm+"_KLDvsit.png"), height=1500, width=2000)
  export_png(fig2, filename=os.path.join(fldr_figs, 'w'+dnm+"_KLDvscput.png"), height=1500, width=2000)
  export_png(fig3, filename=os.path.join(fldr_figs, 'w'+dnm+"_KLDvssz.png"), height=1500, width=2000)
  export_png(fig4, filename=os.path.join(fldr_figs, 'w'+dnm+"_privacy.png"), height=1500, width=2000)

bkp.show(bkl.gridplot(figs))

