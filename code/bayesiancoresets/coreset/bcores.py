import numpy as np
from ..util.errors import NumericalPrecisionError
from ..util.opt import nn_opt, partial_nn_opt
from .coreset import Coreset

flatten = lambda l: [item for sublist in l for item in sublist]

class BetaCoreset(Coreset):
  def __init__(self, data, ll_projector, n_subsample_select=None, n_subsample_opt=None,
              opt_itrs=100, step_sched=lambda i : 1./(1.+i), mup=None, SigpInv=None,
              beta=.5, learn_beta=True, groups=None, selected_groups=None, initialized=False, **kw):
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
    self.groups = groups
    self.selected_groups = []
    super().__init__(**kw)
    self.initialized = int(initialized)*len(self.wts)

  def _build(self, itrs, sz):
    if self.groups is None:
      if self.size()+itrs > sz:
        raise ValueError(self.alg_name + '._build(): # itrs + current size cannot exceed total desired size sz. # itr = ' + str(itrs) + ' cur sz: ' + str(self.size()) + ' desired sz: ' + str(sz))
    for i in range(itrs):
      #search for the next best point
      self._select()
      #update the weights
      self._optimize()

  def _get_projection(self, n_subsample, w, p, beta, select=False):
    #update the projector
    self.ll_projector.update(w, p)
    #construct a tangent space
    # component of full dataset (kl-divergence)
    if n_subsample is None:
      sub_idcs = None
      vecs = self.ll_projector.project_f(self.data, beta)
      sum_scaling = 1.
    elif self.groups is None:
      sub_idcs = np.random.randint(self.data.shape[0], size=n_subsample)
      vecs = self.ll_projector.project_f(self.data[sub_idcs], beta)
      sum_scaling = self.data.shape[0]/n_subsample
    elif self.groups:
      group_idcs = np.random.randint(len(self.groups), size=n_subsample)
      group_idcs_lst = [self.groups[i] for i in group_idcs]
      sub_idcs = flatten([self.groups[idx] for idx in group_idcs])
      vecs = np.array([np.sum(self.ll_projector.project_f(self.data[idcs,:], beta), axis=0) for idcs in group_idcs_lst])
      sum_scaling = self.data.shape[0]/n_subsample
    # component of coreset (using beta-divergence)
    if self.pts.size > 0:
      corevecs = self.ll_projector.project_f(self.pts, beta)
    else:
      corevecs = np.zeros((0, vecs.shape[1]))
    if self.groups is None and select:
      return vecs[~np.all(vecs == 0., axis=1)], sum_scaling, sub_idcs, corevecs
    elif select:
      return vecs, sum_scaling, sub_idcs, group_idcs, corevecs
    else:
      return vecs, sum_scaling, sub_idcs, corevecs

  def _select(self):
    if self.groups is None: # add new individual datapoint to the coreset
      vecs, sum_scaling, sub_idcs, corevecs = self._get_projection(self.n_subsample_select, self.wts, self.pts, self.beta)
      resid = sum_scaling*vecs.sum(axis=0) - self.wts.dot(corevecs)
      corrs = vecs.dot(resid) / np.sqrt((vecs**2).sum(axis=1)) / vecs.shape[1] #up to a constant; good enough for argmax
      corecorrs = np.fabs(corevecs.dot(resid) / np.sqrt((corevecs**2).sum(axis=1))) / corevecs.shape[1] #up to a constant; good enough for argmax
      if corecorrs.size == 0 or corrs.max() > corecorrs.max():
        f = sub_idcs[np.argmax(corrs)] if sub_idcs is not None else np.argmax(corrs)
        #expand and initialize storage for new coreset pt
        #need to double-check that f isn't in self.idcs, since the subsample may contain some of the coreset pts
        if f not in self.idcs:
          self.wts.resize(self.wts.shape[0]+1, refcheck=False)
          self.idcs.resize(self.idcs.shape[0]+1, refcheck=False)
          self.pts.resize((self.pts.shape[0]+1, self.data.shape[1]), refcheck=False)
          self.wts[-1] = 0.
          self.idcs[-1] = f
          self.pts[-1] = self.data[f]
    else: # add new group to the coreset
      groupvecs, sum_scaling, sub_idcs, group_idcs, corevecs = self._get_projection(self.n_subsample_select, self.wts, self.pts, self.beta, select=True)
      #compute the residual error
      #compute the residual error
      if self.n_subsample_select is None:
        resid = groupvecs.sum(axis=0) - self.wts.dot(corevecs)
      else:
        resid = len(sub_idcs)/self.data.shape[0]*groupvecs.sum(axis=0) - self.wts.dot(corevecs)
      #resid = sum_scaling*vecs.sum(axis=0) - self.wts.dot(corevecs)
      #compute the correlations for the new subsample
      corrs = groupvecs.dot(resid) / np.sqrt((groupvecs**2).sum(axis=1)) / groupvecs.shape[1] #up to a constant; good enough for argmax
      #compute the correlations for the coreset pts (use fabs because we can decrease the weight of these)
      corecorrs = np.fabs(corevecs.dot(resid) / np.sqrt((corevecs**2).sum(axis=1))) / corevecs.shape[1] #up to a constant; good enough for argmax
      if corecorrs.shape[0]>self.initialized :
        maxcorecors = corecorrs[self.initialized:].max()
      else:
        maxcorecors = -np.inf
      #get the best selection; if it's an old coreset pt do nothing, if it's a new point expand and initialize storage for the new pt
      if corecorrs.size == 0 or corrs.max() > maxcorecors:
        if self.n_subsample_select is None:
          f = np.argmax(corrs)
        else:
          #print('self.groups[group_idcs[np.argmax(corrs)]] : ', self.groups[group_idcs[np.argmax(corrs)]], group_idcs)
          f = self.groups[group_idcs[np.argmax(corrs)]] if sub_idcs is not None else np.argmax(corrs)
        if f not in self.selected_groups:
          self.selected_groups.append(f)
          newpoints = self.data[f,:]
          self.wts.resize(self.wts.shape[0]+newpoints.shape[0], refcheck=False)
          self.idcs.resize(self.idcs.shape[0]+newpoints.shape[0], refcheck=False)
          self.pts.resize((self.pts.shape[0]+newpoints.shape[0], self.data.shape[1]), refcheck=False)
          self.wts[-newpoints.shape[0]:] = [0.]*newpoints.shape[0]
          self.idcs[-newpoints.shape[0]:] = f
          self.pts[-newpoints.shape[0]:,:] = newpoints
      print('idcs and pts shapes : ', self.idcs.shape, self.pts.shape)
    return

  def _optimize(self):
    if self.learn_beta:
      def grd(x):
        w = x[:-1]
        beta = x[-1]
        vecs, sum_scaling, sub_idcs, corevecs, betagrads = self._get_projection_ii(self.n_subsample_opt, w, self.pts, beta)
        resid = sum_scaling*vecs.sum(axis=0) - w.dot(corevecs)
        wgrad = -corevecs.dot(resid) / corevecs.shape[1]
        betagrad = -10**(-5)*w.dot(betagrads.dot(resid))/corevecs.shape[1]
        grad =  np.hstack((wgrad, betagrad))
        return grad
      x0 = np.hstack((self.wts, np.asarray([self.beta])))
      xf = partial_nn_opt(x0, grd, np.arange(x0.shape[0]), self.opt_itrs, step_sched = self.step_sched)
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
