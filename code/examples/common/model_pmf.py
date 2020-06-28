import numpy as np
import scipy.linalg as sl
from scipy.special import gammaln

def pmf_log_likelihood(x, phis, psis):
  x = np.atleast_2d(x) # dim: [U, I]
  phis = np.atleast_3d(phis) # dim: [S, U, K]
  psis = np.atleast_3d(psis) # dim: [S, I, K]
  y = np.tile(x, (phis.shape[0],1,1)) # dim: [S, U, I]
  th = np.einsum('suk,sik -> sui', phis, psis) # dim: [S, U, I]
  thout = np.random.exponential(10**3, (y.shape[0], max(0,y.shape[1]-th.shape[1]), y.shape[2]))
  th = np.concatenate((th,thout), axis=1)
  return np.sum(np.multiply(y, np.log(th))- gammaln(y+1) - th, axis=2).T # dim: [U,S]

def pmf_log_prior(phis, psis):
  phis = np.atleast_3d(phis) # dim: [S, U, K]
  psis = np.atleast_3d(psis) # dim: [S, I, K]
  return -np.log(10**3)*10**3*phis.sum((1,2)) -np.log(10**3)*10**3*psis.sum((1,2))

def pmf_log_joint(x, phis, psis, wts):
  x = np.atleast_2d(x) # dim: [U, I]
  phis = np.atleast_3d(phis) # dim: [S, U, K]
  psis = np.atleast_3d(psis) # dim: [S, I, K]
  #wts dim: [U]
  wlls = np.einsum('u,us -> s', wts, pmf_log_likelihood(x, phis, psis))
  return wlls + pmf_log_prior(phis, psis)

def pmf_grad_log_likelihood(x, phis, psis):
  return

def test_pmf_log_likelihood(U=100, I=1000, S=50, K=10):
  x = np.random.randint(low=0, high=1, size=(U, I))
  phis = np.random.rand(S, U, K)
  psis = np.random.rand(S, I, K)
  print(pmf_log_likelihood(x, phis, psis).shape)
  return

def test_pmf_log_prior(U=100, I=1000, S=50, K=10):
  phis = np.random.rand(S, U, K)
  psis = np.random.rand(S, I, K)
  print(pmf_log_prior(phis, psis).shape)
  return

def test_pmf_log_joint(U=100, I=1000, S=50, K=10):
  x = np.random.randint(low=0, high=10, size=(U, I))
  phis = np.random.rand(S, U, K)
  psis = np.random.rand(S, I, K)
  w = np.random.rand(U)
  print(pmf_log_joint(x, phis, psis, w).shape)
  return

#test_pmf_log_likelihood()
#test_pmf_log_prior()
#test_pmf_log_joint()
