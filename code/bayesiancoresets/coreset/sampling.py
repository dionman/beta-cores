import numpy as np
from ..util.errors import NumericalPrecisionError
from .coreset import Coreset

class UniformSamplingCoreset(Coreset):
  def __init__(self, data, groups=None, selected_groups=None, **kw):
    super().__init__(**kw)
    self.data = data
    if 'wts' in kw: # if coreset has been initiliazed to dummy random subset
      self.cts = [1]*len(self.idcs.tolist())
      self.ct_idcs = self.idcs.tolist()
    else:
      self.cts = []
      self.ct_idcs = []
    self.groups = groups
    self.selected_groups = []

  def reset(self):
    self.cts = []
    self.ct_idcs = []
    super().reset()

  def _build(self, itrs, sz):
    if self.size()+itrs > sz:
      raise ValueError(self.alg_name + '._build(): # itrs + current size cannot exceed total desired size sz. # itr = ' + str(itrs) + ' cur sz: ' + str(self.size()) + ' desired sz: ' + str(sz))
    if self.groups is None:
      for i in range(itrs):
        f = np.random.randint(self.data.shape[0])
        if f in self.ct_idcs:
          self.cts[self.ct_idcs.index(f)] += 1
        else:
          self.ct_idcs.append(f)
          self.cts.append(1)
      self.wts = self.data.shape[0]*np.array(self.cts)/np.array(self.cts).sum()
      self.idcs = np.array(self.ct_idcs)
      self.pts = self.data[self.idcs]
    else:
      for i in range(itrs):
        f = np.random.randint(len(self.groups))
        #expand and initialize storage for new coreset pt
        #need to double-check that f isn't in self.idcs, since the subsample may contain some of the coreset pts
        if f not in self.selected_groups:
          newpoints = self.data[self.groups[f],:]
          self.ct_idcs.append(self.groups[f])
          self.cts+=[1]*newpoints.shape[0]
          self.wts.resize(self.wts.shape[0]+newpoints.shape[0], refcheck=False)
          self.idcs.resize(self.idcs.shape[0]+newpoints.shape[0], refcheck=False)
          self.pts.resize((self.pts.shape[0]+newpoints.shape[0], self.data.shape[1]), refcheck=False)
          self.wts = self.data.shape[0]*np.array(self.cts)/np.array(self.cts).sum()
          self.idcs[-newpoints.shape[0]:] = self.groups[f]
          self.pts[-newpoints.shape[0]:,:] = newpoints
          self.selected_groups.append(f)
