import numpy as np
from ..util.errors import NumericalPrecisionError
from ..util.opt import nn_opt, partial_nn_opt
from .coreset import Coreset

class BetaCoreset(Coreset):
  def __init__(self, data, ll_projector, n_subsample_select=None, n_subsample_opt=None,
              opt_itrs=100, step_sched=lambda i : 1./(1.+i), mup=None, SigpInv=None,
              beta=.5, learn_beta=True, **kw):
    self.data = data
    self.ll_projector = ll_projector
    self.n_subsample_select = None if n_subsample_select is None else min(data.shape[0], n_subsample_select)
    self.n_subsample_opt = None if n_subsample_opt is None else min(data.shape[0], n_subsample_opt)
    self.step_sched = step_sched
    self.opt_itrs = opt_itrs
    self.mup = mup
    self.SigpInv = SigpInv
    self.beta = beta
    self.learn_beta=learn_beta
    super().__init__(**kw)

  def _build(self, itrs, sz):
    if self.size()+itrs > sz:
      raise ValueError(self.alg_name + '._build(): # itrs + current size cannot exceed total desired size sz. # itr = ' + str(itrs) + ' cur sz: ' + str(self.size()) + ' desired sz: ' + str(sz))
    for i in range(itrs):
      #search for the next best point
      self._select()
      #update the weights
      self._optimize()

  def _get_projection(self, n_subsample, w, p, beta):
    #update the projector
    self.ll_projector.update(w, p)
    #construct a tangent space
    # component of full dataset (kl-divergence)
    if n_subsample is None:
      sub_idcs = None
      vecs = self.ll_projector.project_fprime(self.data)
      sum_scaling = 1.
    else:
      sub_idcs = np.random.randint(self.data.shape[0], size=n_subsample)
      vecs = self.ll_projector.project_fprime(self.data[sub_idcs])
      sum_scaling = self.data.shape[0]/n_subsample
    # component of coreset (using beta-divergence)
    if self.pts.size > 0:
      corevecs = self.ll_projector.project_f(self.pts, beta)
    else:
      corevecs = np.zeros((0, vecs.shape[1]))
    return vecs, sum_scaling, sub_idcs, corevecs

  def _get_projection_ii(self, n_subsample, w, p, beta):
    #update the projector
    self.ll_projector.update(w, p)
    #construct a tangent space
    # component of full dataset (kl-divergence)
    if n_subsample is None:
      sub_idcs = None
      vecs = self.ll_projector.project_fprime(self.data)
      sum_scaling = 1.
    else:
      sub_idcs = np.random.randint(self.data.shape[0], size=n_subsample)
      vecs = self.ll_projector.project_fprime(self.data[sub_idcs])
      sum_scaling = self.data.shape[0]/n_subsample
    # component of coreset (using beta-divergence)
    if self.pts.size > 0:
      corevecs, betagrads = self.ll_projector.project_f(self.pts, beta, grad=True)
    else:
      corevecs, betagrads = np.zeros((0, vecs.shape[1])), np.asarray([0.5])
    return vecs, sum_scaling, sub_idcs, corevecs, betagrads

  def _select(self):
    vecs, sum_scaling, sub_idcs, corevecs = self._get_projection(self.n_subsample_select, self.wts, self.pts, self.beta)

    #compute the residual error
    resid = sum_scaling*vecs.sum(axis=0) - self.wts.dot(corevecs)

    #compute the correlations for the new subsample
    corrs = vecs.dot(resid) / np.sqrt((vecs**2).sum(axis=1)) / vecs.shape[1] #up to a constant; good enough for argmax
    #compute the correlations for the coreset pts (use fabs because we can decrease the weight of these)
    corecorrs = np.fabs(corevecs.dot(resid) / np.sqrt((corevecs**2).sum(axis=1))) / corevecs.shape[1] #up to a constant; good enough for argmax

    #get the best selection; if it's an old coreset pt do nothing, if it's a new point expand and initialize storage for the new pt
    if corecorrs.size == 0 or corrs.max() > corecorrs.max():
      f = sub_idcs[np.argmax(corrs)] if sub_idcs is not None else np.argmax(corrs)
      #expand and initialize storage for new coreset pt
      #need to double-check that f isn't in self.idcs, since the subsample may contain some of the coreset pts
      if f not in self.idcs:
        self.wts = np.resize(self.wts,(self.wts.shape[0]+1))
        self.idcs.resize(self.idcs.shape[0]+1)
        self.pts.resize((self.pts.shape[0]+1, self.data.shape[1]))
        self.wts[-1] = 0.
        self.idcs[-1] = f
        self.pts[-1] = self.data[f]
    return

  def _optimize(self):
    if self.learn_beta:
      def grd(x):
        w = x[:-1]
        vecs, sum_scaling, sub_idcs, corevecs, betagrads = self._get_projection_ii(self.n_subsample_opt, x, self.pts, self.beta)
        resid = sum_scaling*vecs.sum(axis=0) - w.dot(corevecs)
        wgrad = -corevecs.dot(resid) / corevecs.shape[1]
        ugrad = -w*betagrads.dot(resid)/corevecs.shape[1]
        grad =  np.hstack((wgrad, ugrad))
        return grad
      x0 = np.hstack((self.wts, np.asarray([self.beta])))
      xf = partial_nn_opt(x0, grd, 1, self.opt_itrs, step_sched = self.step_sched)
      self.wts = xf[:-1]
      self.beta = xf[-1]
    else:
      def grd(w):
        vecs, sum_scaling, sub_idcs, corevecs = self._get_projection(self.n_subsample_opt, w, self.pts, self.beta)
        resid = sum_scaling*vecs.sum(axis=0) - w.dot(corevecs)
        #output gradient of weights at idcs
        return -corevecs.dot(resid) / corevecs.shape[1]
      x0 = self.wts
      self.wts = nn_opt(x0, grd, opt_itrs=self.opt_itrs, step_sched = self.step_sched)

  def error(self):
    return 0. #TODO: implement KL estimate

  def get(self):
    return self.wts[self.wts > 0], self.pts[self.wts > 0, :], self.idcs[self.wts > 0], self.beta
