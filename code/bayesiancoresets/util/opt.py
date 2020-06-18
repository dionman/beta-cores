import numpy as np
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../../examples/common'))
from model_lr import *
import gaussian, iwg
import scipy.linalg as sl
from scipy.optimize import minimize, nnls

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


def partial_nn_opt_diagn(x0, grd, nn_idcs, opt_itrs=1000, step_sched=lambda i : 1./(i+1), b1=0.9, b2=0.999, eps=1e-8, verbose=False, m=None, d=None, Zmean=None, mup=None, SigpInv=None):
  x = x0.copy()
  m1 = np.zeros(x.shape[0])
  m2 = np.zeros(x.shape[0])
  kl =[]
  for i in range(opt_itrs):
    g = grd(x)
    if verbose:
      active_idcs = np.intersect1d(nn_idcs, np.where(x==0)[0])
      inactive_idcs = np.setdiff1d(np.arange(x.shape[0]), active_idcs)
      sys.stdout.write('itr ' + str(i+1) +'/'+str(opt_itrs)+': ||inactive constraint grads|| = ' + str(np.sqrt((g[inactive_idcs]**2).sum())) + '                \r')
      sys.stdout.flush()
    m1 = b1*m1 + (1.-b1)*g
    m2 = b2*m2 + (1.-b2)*g**2
    upd = step_sched(i)*m1/(1.-b1**(i+1))/(eps + np.sqrt(m2/(1.-b2**(i+1))))
    x -= upd
    #project onto x>=0
    x[nn_idcs] = np.maximum(x[nn_idcs], 0.)
    if True: #int(m)%2==0: # print diagnostics over optimization
      if True: #int(i+1)%10==1:
        #print('iteration :', i)
        #print('dimension d=',d, 'size m=',m)
        wts = x[:m]
        pts = x[m:].reshape((m, d))
        mul, LSigl, LSiglInv = get_laplace(wts, pts, Zmean)
        kld = gaussian.gaussian_KL(mul, LSigl.dot(LSigl.T), mup, SigpInv)
        #input()
        kl+=[kld]
  
  #print('m=', m, 'kl=', kl, '\n\n')
  if verbose:
    sys.stdout.write('\n')
    sys.stdout.flush()
  return x


def nn_opt_diagn(x0, grd, opt_itrs=1000, step_sched = lambda i : 1./(i+1), b1=0.9, b2=0.999, eps=1e-8, verbose=False, m=None, d=None, Zmean=None, mup=None, SigpInv=None, data=None):
  x = x0.copy()
  m1 = np.zeros(x.shape[0])
  m2 = np.zeros(x.shape[0])
  for i in range(opt_itrs):
    g = grd(x)
    if verbose:
      sys.stdout.write('itr ' + str(i+1) +'/'+str(opt_itrs)+': ||inactive constraint grads|| = ' + str(np.sqrt((g[x>0]**2).sum())) + '                \r')
      sys.stdout.flush()
    m1 = b1*m1 + (1.-b1)*g
    m2 = b2*m2 + (1.-b2)*g**2
    upd = step_sched(i)*m1/(1.-b1**(i+1))/(eps + np.sqrt(m2/(1.-b2**(i+1))))
    x -= upd
    #project onto x>=0
    x = np.maximum(x, 0.)
    if int(m)%10==0: # print diagnostics over optimization
      if int(i+1)%10==0:
        #print('iteration :', i)
        #print('dimension d=',d, 'size m=',m)
        wts = x
        pts = data
        #print(wts.shape, pts.shape, Zmean.shape)
        #exit()
        mul, LSigl, LSiglInv = get_laplace(wts, pts, Zmean)
        kld = gaussian.gaussian_KL(mul, LSigl.dot(LSigl.T), mup, SigpInv)
        print('i: ', i, ' kld=', kld, 'mean norm : ', np.linalg.norm(mul-mup))
        #input()
  if verbose:
    sys.stdout.write('\n')
    sys.stdout.flush()
  return x

def nn_opt(x0, grd, opt_itrs=1000, step_sched = lambda i : 1./(i+1), b1=0.9, b2=0.999, eps=1e-8, verbose=False):
  x = x0.copy()
  m1 = np.zeros(x.shape[0])
  m2 = np.zeros(x.shape[0])
  for i in range(opt_itrs):
    g = grd(x)
    if verbose:
      sys.stdout.write('itr ' + str(i+1) +'/'+str(opt_itrs)+': ||inactive constraint grads|| = ' + str(np.sqrt((g[x>0]**2).sum())) + '                \r')
      sys.stdout.flush()
    m1 = b1*m1 + (1.-b1)*g
    m2 = b2*m2 + (1.-b2)*g**2
    upd = step_sched(i)*m1/(1.-b1**(i+1))/(eps + np.sqrt(m2/(1.-b2**(i+1))))
    x -= upd
    #project onto x>=0
    x = np.maximum(x, 0.)
  if verbose:
    sys.stdout.write('\n')
    sys.stdout.flush()
  return x

def partial_nn_opt(x0, grd, nn_idcs, opt_itrs=1000, step_sched=lambda i : 1./(i+1), b1=0.9, b2=0.999, eps=1e-8, verbose=False):
  x = x0.copy()
  m1 = np.zeros(x.shape[0])
  m2 = np.zeros(x.shape[0])
  for i in range(opt_itrs):
    g = grd(x)
    if verbose:
      active_idcs = np.intersect1d(nn_idcs, np.where(x==0)[0])
      inactive_idcs = np.setdiff1d(np.arange(x.shape[0]), active_idcs)
      sys.stdout.write('itr ' + str(i+1) +'/'+str(opt_itrs)+': ||inactive constraint grads|| = ' + str(np.sqrt((g[inactive_idcs]**2).sum())) + '                \r')
      sys.stdout.flush()
    m1 = b1*m1 + (1.-b1)*g
    m2 = b2*m2 + (1.-b2)*g**2
    upd = step_sched(i)*m1/(1.-b1**(i+1))/(eps + np.sqrt(m2/(1.-b2**(i+1))))
    x -= upd
    #project onto x>=0
    x[nn_idcs] = np.maximum(x[nn_idcs], 0.)
  if verbose:
    sys.stdout.write('\n')
    sys.stdout.flush()
  return x


# no zero crossings allowed (for logreg with datapoints constrained to positive values) 
def nzc_opt(x0, grd, nn_idcs, opt_itrs=1000, step_sched=lambda i : 1./(i+1), b1=0.9, b2=0.99, eps=1e-8, verbose=False):
  x = x0.copy()
  m1 = np.zeros(x.shape[0])
  m2 = np.zeros(x.shape[0])
  for i in range(opt_itrs):
    g = grd(x)
    if verbose:
      active_idcs = np.intersect1d(nn_idcs, np.where(x==0)[0])
      inactive_idcs = np.setdiff1d(np.arange(x.shape[0]), active_idcs)
      sys.stdout.write('itr ' + str(i+1) +'/'+str(opt_itrs)+': ||inactive constraint grads|| = ' + str(np.sqrt((g[inactive_idcs]**2).sum())) + '                \r')
      sys.stdout.flush()
    m1 = b1*m1 + (1.-b1)*g
    m2 = b2*m2 + (1.-b2)*g**2
    upd = step_sched(i)*m1/(1.-b1**(i+1))/(eps + np.sqrt(m2/(1.-b2**(i+1))))
    #project elementwise updates to the halfspace where elem belongs
    x = (x-upd)*(np.sign(x)*(np.sign(x-upd))==1) + np.sign(x)*(1e-12) # apply update if it doesn't change the sign, otherwise project close to zero maintaining the sign
        
  if verbose:
    sys.stdout.write('\n')
    sys.stdout.flush()
  return x
