import numpy as np
from ..util.errors import NumericalPrecisionError
from ..util.opt import nn_opt
from .coreset import Coreset

class SparseVICoreset(Coreset):
  def __init__(self, data, ll_projector, n_subsample_select=None, n_subsample_opt=None,
              opt_itrs=100, step_sched=lambda i : 1./(1.+i), mup=None, SigpInv=None,
              groups=None, selected_groups=None, initialized=False, **kwargs):
    self.data = data
    self.ll_projector = ll_projector
    self.n_subsample_select = None if n_subsample_select is None else min(data.shape[0], n_subsample_select)
    self.n_subsample_opt = None if n_subsample_opt is None else min(data.shape[0], n_subsample_opt)
    self.step_sched = step_sched
    self.opt_itrs = opt_itrs
    self.mup = mup
    self.SigpInv = SigpInv
    self.groups = groups
    self.selected_groups = []
    super().__init__(**kwargs)
    self.initialized = int(initialized)*len(self.wts)

  def _build(self, itrs, sz):
    if self.size()+itrs > sz:
      raise ValueError(self.alg_name + '._build(): # itrs + current size cannot exceed total desired size sz. # itr = ' + str(itrs) + ' cur sz: ' + str(self.size()) + ' desired sz: ' + str(sz))
    for i in range(itrs):
      #search for the next best point
      self._select()
      #update the weights
      self._optimize()

  def _get_projection(self, n_subsample, w, p, select=False):
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
    if self.pts.size > 0:
      corevecs = self.ll_projector.project(p)
    else:
      corevecs = np.zeros((0, vecs.shape[1]))
    if self.groups is None and select:
      return vecs[~np.all(vecs == 0., axis=1)], sum_scaling, sub_idcs, corevecs
    else:
      return vecs, sum_scaling, sub_idcs, corevecs

  def _select(self):
    if self.groups is None: # add new individual datapoint to the coreset
      vecs, sum_scaling, sub_idcs, corevecs = self._get_projection(self.n_subsample_select, self.wts, self.pts)
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
          self.wts.resize(self.wts.shape[0]+1, refcheck=False)
          self.idcs.resize(self.idcs.shape[0]+1, refcheck=False)
          self.pts.resize((self.pts.shape[0]+1, self.data.shape[1]), refcheck=False)
          self.wts[-1] = 0.
          self.idcs[-1] = f
          self.pts[-1] = self.data[f]
    else: # add new group to the coreset
      vecs, sum_scaling, sub_idcs, corevecs = self._get_projection(self.n_subsample_select, self.wts, self.pts, select=True)
      groupvecs = np.asarray([vecs[idx,:].sum(axis=0) for idx in self.groups])
      #compute the residual error
      resid = sum_scaling*groupvecs.sum(axis=0) - self.wts.dot(corevecs)
      #compute the correlations for the new subsample
      corrs = groupvecs.dot(resid) / np.sqrt((groupvecs**2).sum(axis=1)) / groupvecs.shape[1] #up to a constant; good enough for argmax
      #print('corrs : ', corrs.shape)
      #compute the correlations for the coreset pts (use fabs because we can decrease the weight of these)
      corecorrs = np.fabs(corevecs.dot(resid) / np.sqrt((corevecs**2).sum(axis=1))) / corevecs.shape[1] #up to a constant; good enough for argmax
      #print('corecorrs : ', corecorrs, 'corrs :', corrs, np.argmax(corrs))

      if corecorrs.size>0: print('corecorrs and corrs shape : ', corecorrs.shape, corrs.shape)
      #get the best selection; if it's an old coreset pt do nothing, if it's a new point expand and initialize storage for the new pt

      if corecorrs.shape[0]>self.initialized :
        maxcorecors = corecorrs[self.initialized:].max()
      else:
        maxcorecors = -np.inf
      if corecorrs.size == 0 or corrs.max() > maxcorecors:
        print('\n\n\nsub_idcs : ', sub_idcs)

        f = sub_idcs[self.groups[np.argmax(corrs)]] if sub_idcs is not None else np.argmax(corrs)
        print('f = ', f)
        if f not in self.selected_groups: self.selected_groups.append(f)
        #expand and initialize storage for new coreset pt
        #need to double-check that f isn't in self.idcs, since the subsample may contain some of the coreset pts
        if f not in self.idcs:
          newpoints = self.data[self.groups[f],:]
          self.wts.resize(self.wts.shape[0]+newpoints.shape[0], refcheck=False)
          self.idcs.resize(self.idcs.shape[0]+newpoints.shape[0], refcheck=False)
          self.pts.resize((self.pts.shape[0]+newpoints.shape[0], self.data.shape[1]), refcheck=False)
          self.wts[-newpoints.shape[0]:] = [0.]*newpoints.shape[0]
          self.idcs[-newpoints.shape[0]:] = self.groups[f]
          self.pts[-newpoints.shape[0]:,:] = newpoints
      print('idcs and pts shapes : ', self.idcs.shape, self.pts.shape)
    return

  def _optimize(self):
    def grd(w):
      vecs, sum_scaling, sub_idcs, corevecs = self._get_projection(self.n_subsample_opt, w, self.pts)
      resid = sum_scaling*vecs.sum(axis=0) - w.dot(corevecs)
      #output gradient of weights at idcs
      return -corevecs.dot(resid) / corevecs.shape[1]
    x0 = self.wts
    self.wts = nn_opt(x0, grd, opt_itrs=self.opt_itrs, step_sched = self.step_sched)

  def error(self):
    return 0. #TODO: implement KL estimate
