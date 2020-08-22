import numpy as np
import scipy.linalg as sl
import jax.numpy as np
from jax import grad, jit, vmap, jacfwd, jacrev
from jax import random

def log_likelihood(x, mu, Sigma):
  x = np.atleast_2d(x)
  mu = np.atleast_2d(mu)
  Sigma = np.atleast_2d(Sigma)
  sign, Slogdet = sl.slogdet(Sigma)
  Sinv = sl.inv(Sigma)
  return -.5*x.shape[1]*np.log(2.*np.pi) - .5*Slogdet - .5*((x-mu).T)*Sinv*(x-mu)

def log_prior(mu, Sigma, nu0=None, kappa0=None, Sigma0=None, mu0=None, N=1000):
  nu0 = N+1
  kappa0=1
  Sigma0 = np.ones((mu.shape[0], mu.shape[0]))
  mu0 = np.ones((mu.shape[0],1))
  sign, Slogdet = sl.slogdet(Sigma)
  Sinv = sl.inv(Sigma)
  return -.5(nu0+mu.shape[0] + 2.)*Slogdet - .5*kappa0*(mu - mu0)^T*Sinv*(mu - mu0) - .5*np.trace(Sinv*Sigma0)

def log_joint(x, mu, Sigma, wts):
  return (wts[:, np.newaxis]*log_likelihood(z, mu, Sigma)).sum(axis=0) + log_prior(mu, Sigma)

def grad_th_log_likelihood(x, mu, Sigma):
  return grad(log_likelihood, (1,2))(x, mu, Sigma)

def grad_z_log_likelihood(x, mu, Sigma):
  return grad(log_likelihood, 0)(x, mu, Sigma)

def grad_th_log_prior(mu, Sigma):
  return grad(log_prior, (0,1))(mu, Sigma)

def grad_th_log_joint(x, mu, Sigma, wts):
  return grad_th_log_prior(mu, Sigma) + (wts[:, np.newaxis, np.newaxis]*grad_th_log_likelihood(x, mu, Sigma)).sum(axis=0)

def hessian(f):
  return jit(jacfwd(jacrev(f)))

def hess_th_log_likelihood(x, mu, Sigma):
  return hessian(log_likelihood, (1,2))(x, mu, Sigma)

def hess_th_log_prior(mu, Sigma):
  return hessian(log_prior, (0,1))(mu, Sigma)

def hess_th_log_joint(x, mu, Sigma, wts):
  return hess_th_log_prior(mu, Sigma) + (wts[:, np.newaxis, np.newaxis, np.newaxis]*hess_th_log_likelihood(x, mu, Sigma)).sum(axis=0)



 

