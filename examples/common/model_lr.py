import numpy as np
import sys
np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
import scipy.sparse as sp

def load_data(dnm, ttr=0.2):
  # read data to numpy arrays and
  # split to train and test dataset according to ratio ttr
  data = np.load(dnm, allow_pickle=True)
  X = data['X']
  Y = data['y']
  if not (data['Xt'].size in [0,1]) and not (data['yt'].size in [0,1]):
    Xt, Yt = data['Xt'], data['yt']
  else:
    test_size = int(ttr*X.shape[0])
    X, Y, Xt, Yt = X[:-test_size,:], Y[:-test_size], X[-test_size:,:], Y[-test_size:]
  data.close()
  return X, Y, Xt, Yt

def std_cov(X, Y, mean_=None, std_=None):
  #standardize the covariates; **last col is intercept**, so no stdization there
  if (mean_ is None) & (std_ is None): # train datapoints
    x_mean = X[:,:-1].mean(axis=0)
    x_std = np.cov(X[:,:-1], rowvar=False)+1e-12*np.eye(X[:,:-1].shape[1])
  else: # test datapoints
    x_mean = mean_
    x_std = std_
  X[:,:-1] = np.linalg.solve(np.linalg.cholesky(x_std), (X[:,:-1] - x_mean).T).T
  Z = Y[:, np.newaxis]*X
  return X, Y, Z, x_mean, x_std

def compute_accuracy(Xt, Yt, thetas):
  loglikep = log_likelihood(Xt,thetas)
  logliken = log_likelihood(-Xt,thetas)
  # make predictions based on max log likelihood under each sampled parameter
  # theta
  predictions = np.ones(loglikep.shape)
  predictions[logliken > loglikep] = -1
  # compute the distribution of the error rate using max LL on the test set
  # under the posterior theta distribution
  acc = np.mean(Yt[:, np.newaxis] == predictions)
  return acc

def perturb(X_train, y_train, noise_x=(0,5), f_rate=0.1, flip=True, structured=False, mean_val=0.1, std_val=1., theta_val=-1.):
  N, D = X_train.shape
  o = np.int(N*f_rate)
  print('o=',o)
  idxx = np.random.choice(N, size=o)
  if not structured: # random noise/mislabeling in input/output space
    idxy = np.random.choice(N, size=o)
    idcs =  np.random.choice(D,int(D/2.),replace=False) 
    for i in idcs: # replace half of the features with gaussian noise 
      X_train[idxx,i] = np.random.normal(noise_x[0], noise_x[1], size=o)
    if flip:       # flip the labels
      y_train[idxy] = -y_train[idxy]
  else: # structured perturbation for desirable decision boundary
    X_train[idxx,:], y_train[idxx], _, _ = gen_synthetic(o, d=D, mean_val=mean_val, std_val=std_val, theta_val=theta_val) 
  outidx = np.unique(np.concatenate([idxx, idxy]))
  print('after perturbations : ', X_train, y_train)
  return X_train, y_train, y_train[:, np.newaxis]*X_train, outidx

def gen_synthetic(n, d=2, mean_val=1., std_val=1., theta_val=1.):
  mu = mean_val*np.ones(d)
  cov = std_val*np.eye(d)
  th = theta_val*np.ones(d)
  X = np.random.multivariate_normal(mu, cov, n)
  ps = 1.0/(1.0+np.exp(-(X*th).sum(axis=1)))
  y = (np.random.rand(n) <= ps).astype(int)
  y[y==0] = -1
  return X, y, y[:, np.newaxis]*X, (y[:, np.newaxis]*X).mean(axis=0)

def log_likelihood(z, th):
  z = np.atleast_2d(z)
  th = np.atleast_2d(th)
  m = -z.dot(th.T)
  idcs = m < 100
  m[idcs] = -np.log1p(np.exp(m[idcs]))
  m[np.logical_not(idcs)] = -m[np.logical_not(idcs)]
  return m

def beta_likelihood(z, th, beta):
  z = np.atleast_2d(z)
  th = np.atleast_2d(th)
  m = -z.dot(th.T)
  m = -(((beta+1.)/beta)*(1+np.exp(m))**(-beta) - ((1+np.exp(m))**(-beta-1.) + (1+np.exp(-m))**(-beta-1.)))
  return m

def log_prior(th):
  th = np.atleast_2d(th)
  return -0.5*th.shape[1]*np.log(2.*np.pi) - 0.5*(th**2).sum(axis=1)

def log_joint(z, th, wts):
  return (wts[:, np.newaxis]*log_likelihood(z, th)).sum(axis=0) + log_prior(th)

def logistic_likelihood(z, th, wts):
  return (wts[:, np.newaxis]*log_likelihood(z, th)).sum(axis=0)

def grad_th_log_likelihood(z, th):
  z = np.atleast_2d(z)
  th = np.atleast_2d(th)
  m = -z.dot(th.T)
  idcs = m < 100
  m[idcs] = np.exp(m[idcs])/(1.+np.exp(m[idcs]))
  m[np.logical_not(idcs)] = 1.
  return m[:, :, np.newaxis]*z[:, np.newaxis, :]

def grad_z_log_likelihood(z, th):
  z = np.atleast_2d(z)
  th = np.atleast_2d(th)
  m = -z.dot(th.T)
  idcs = m < 100
  m[idcs] = np.exp(m[idcs])/(1.+np.exp(m[idcs]))
  m[np.logical_not(idcs)] = 1.
  return m[:, :, np.newaxis]*th[np.newaxis, :, :]

def grad_th_log_prior(th):
  th = np.atleast_2d(th)
  return -th

def grad_th_log_joint(z, th, wts):
  return grad_th_log_prior(th) + (wts[:, np.newaxis, np.newaxis]*grad_th_log_likelihood(z, th)).sum(axis=0)

def hess_th_log_likelihood(z, th):
  z = np.atleast_2d(z)
  th = np.atleast_2d(th)
  m = -z.dot(th.T)
  idcs = m < 100
  m[idcs] = np.exp(m[idcs])/(1.+np.exp(m[idcs]))**2
  m[np.logical_not(idcs)] = 0.
  return -m[:, :, np.newaxis, np.newaxis]*z[:, np.newaxis, :, np.newaxis]*z[:, np.newaxis, np.newaxis, :]

def hess_th_log_prior(th):
  th = np.atleast_2d(th)
  return np.tile(-np.eye(th.shape[1]), (th.shape[0], 1, 1))

def hess_th_log_joint(z, th, wts):
  return hess_th_log_prior(th) + (wts[:, np.newaxis, np.newaxis, np.newaxis]*hess_th_log_likelihood(z, th)).sum(axis=0)

def diag_hess_th_log_likelihood(z, th):
  z = np.atleast_2d(z)
  th = np.atleast_2d(th)
  m = -z.dot(th.T)
  idcs = m < 100
  m[idcs] = np.exp(m[idcs])/(1.+np.exp(m[idcs]))**2
  m[np.logical_not(idcs)] = 0.
  return -m[:, :, np.newaxis]*z[:, np.newaxis, :]**2

def diag_hess_th_log_prior(th):
  th = np.atleast_2d(th)
  return np.tile(-np.ones(th.shape[1]), (th.shape[0], 1))

def diag_hess_th_log_joint(z, th, wts):
  return diag_hess_th_log_prior(th) + (wts[:, np.newaxis, np.newaxis]*diag_hess_th_log_likelihood(z, th)).sum(axis=0)
