import numpy as np
import scipy.linalg as sl

def gaussian_potentials(Siginv, xSiginvx, xSiginv, logdetSig, x, samples):
  return -x.shape[1]/2*np.log(2*np.pi) - 1./2.*logdetSig - 1./2.*(xSiginvx[:, np.newaxis] - 2.*np.dot(xSiginv, samples.T) + (np.dot(samples, Siginv)*samples).sum(axis=1))

def gaussian_loglikelihood(x, th, Siginv, logdetSig):
  x = np.atleast_2d(x)
  th = np.atleast_2d(th)
  xSiginvx = (x*(x.dot(Siginv))).sum(axis=1)
  thSiginvth = (th*(th.dot(Siginv))).sum(axis=1)
  xSiginvth = x.dot(Siginv.dot(th.T))
  return -x.shape[1]/2*np.log(2*np.pi) - 1./2.*logdetSig - 1./2.*(xSiginvx[:, np.newaxis] + thSiginvth - 2*xSiginvth)

def gaussian_grad_x_loglikelihood(x, th, Siginv):
  x = np.atleast_2d(x)
  th = np.atleast_2d(th)
  return th.dot(Siginv)[np.newaxis, :, :] - x.dot(Siginv)[:, np.newaxis, :]
 
def gaussian_KL(mu0, Sig0, mu1, Sig1inv):
  t1 = np.dot(Sig1inv, Sig0).trace()
  t2 = np.dot((mu1-mu0),np.dot(Sig1inv, mu1-mu0))
  t3 = -np.linalg.slogdet(Sig1inv)[1] - np.linalg.slogdet(Sig0)[1]
  return 0.5*(t1+t2+t3-mu0.shape[0])

def weighted_post(th0, Sig0inv, Siginv, x, w): 
  LSigpInv = np.linalg.cholesky(Sig0inv + w.sum()*Siginv)
  LSigp = sl.solve_triangular(LSigpInv, np.eye(LSigpInv.shape[0]), lower=True, overwrite_b=True, check_finite=False)
  mup = np.dot(LSigp.dot(LSigp.T),  np.dot(Sig0inv,th0) + np.dot(Siginv, (w[:, np.newaxis]*x).sum(axis=0)))
  return mup, LSigp, LSigpInv

