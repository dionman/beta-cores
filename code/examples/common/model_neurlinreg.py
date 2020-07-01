import numpy as np
import scipy.linalg as sl

def neurlinreg_loglikelihood(z, th, sigsq):
  z = np.atleast_2d(z)
  x = z[:, :-1]
  y = z[:, -1]
  th = np.atleast_2d(th)
  XST = x.dot(th.T)
  return -1./2.*np.log(2.*np.pi*sigsq) - 1./(2.*sigsq)*(y[:,np.newaxis]**2 - 2*XST*y[:,np.newaxis] + XST**2)

def neurlinreg_grad_x_loglikelihood(z, th, sigsq):
  pass

def neurlinreg_beta_likelihood():
  pass
