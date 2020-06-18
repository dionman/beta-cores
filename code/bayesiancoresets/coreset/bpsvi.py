import numpy as np
from ..util.errors import NumericalPrecisionError
from ..util.opt import partial_nn_opt, nzc_opt, partial_nn_opt_diagn
from .coreset import Coreset

class BatchPSVICoreset(Coreset):
  def __init__(self, data, ll_projector, opt_itrs, n_subsample_opt=None, step_sched=lambda m: lambda i : 1./(1.+i), mup=None, Zmean=None, SigpInv=None, diagnostics=False, **kw): 
    self.data = data
    self.ll_projector = ll_projector
    self.opt_itrs = opt_itrs
    self.n_subsample_opt = None if n_subsample_opt is None else min(data.shape[0], n_subsample_opt)
    self.step_sched = step_sched
    self.mup = mup
    self.SigpInv = SigpInv
    self.Zmean = Zmean
    self.diagnostics = diagnostics
    super().__init__(**kw)

  def _build(self, itrs, sz):
    # initialize the points via full dataset subsampling
    init_idcs = np.random.choice(self.data.shape[0], size=sz, replace=False)
    self.pts = self.data[init_idcs]
    self.wts = self.data.shape[0]/sz*np.ones(sz)
    self.idcs = -1*np.ones(sz)
    # run gradient optimization for opt_itrs steps
    self._optimize()

  def _get_projection(self, n_subsample, w, p):
    #update the projector
    self.ll_projector.update(w, p)
    #construct a tangent space
    if n_subsample is None:
      sub_idcs = None
      vecs = self.ll_projector.project(self.data)
      sum_scaling = 1.
    else:
      sub_idcs = np.random.randint(self.data.shape[0], size=n_subsample)
      vecs = self.ll_projector.project(self.data[sub_idcs])
      sum_scaling = self.data.shape[0]/n_subsample
    if p.size > 0:
      corevecs, pgrads = self.ll_projector.project(p, grad=True)
    else:
      corevecs, pgrads = np.zeros((0, vecs.shape[1])), np.zeros((0, vecs.shape[1], p.shape[1]))
    return vecs, sum_scaling, sub_idcs, corevecs, pgrads

  def _optimize(self):
    sz = self.wts.shape[0]
    d = self.pts.shape[1]
    def grd(x):
      w = x[:sz]
      p = x[sz:].reshape((sz, d))
      vecs, sum_scaling, sub_idcs, corevecs, pgrads = self._get_projection(self.n_subsample_opt, w, p)
      #compute gradient of weights and pts
      resid = sum_scaling*vecs.sum(axis=0) - w.dot(corevecs)
      wgrad = -corevecs.dot(resid) / corevecs.shape[1]
      ugrad = -(w[:, np.newaxis, np.newaxis]*pgrads*resid[np.newaxis, :, np.newaxis]).sum(axis=1)/corevecs.shape[1]
      #return reshaped grad
      grad =  np.hstack((wgrad, ugrad.reshape(sz*d))) 
      return grad
    
    x0 = np.hstack((self.wts, self.pts.reshape(sz*d)))
    # initialization for mnist assign sign of label to zero x0 elements
    #self.pts = np.apply_along_axis(lambda r: r+(np.sign(np.max(r))+np.sign(np.min(r)))*(1e-12), axis=1, arr=self.pts )
    #x0 = np.hstack((self.wts, self.pts.reshape(sz*d)))
    # full non-negative projection for mnist visualization!!!
    #xf = nzc_opt(x0, grd, np.arange(sz), self.opt_itrs, step_sched=self.step_sched(sz))  
    # project weights on non-negative values
    if not self.diagnostics:
      xf = partial_nn_opt(x0, grd, np.arange(sz), self.opt_itrs, step_sched = self.step_sched(sz))
    else:
      xf = partial_nn_opt_diagn(x0, grd, np.arange(sz), self.opt_itrs, step_sched = self.step_sched(sz), Zmean=self.Zmean, mup=self.mup, SigpInv=self.SigpInv, m=sz, d=d)
    #print('self.Zmean inside bpsvi : ', self.Zmean)
    self.wts = xf[:sz]
    self.pts = xf[sz:].reshape((sz, d))

  def error(self):
    return 0. #TODO: implement KL estimate

 
