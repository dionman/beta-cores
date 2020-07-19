import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
import scipy.sparse as sp

def isinteger(x):
  return np.equal(np.mod(x, 1), 0)

def load_data(dnm, ttr=0.1):
  # read data to numpy arrays and
  # split to train and test dataset according to ratio ttr
  data = np.load(dnm)
  X = data['X']
  Y = data['y']
  if 'Xt' in data.keys()
    Xt, Yt = data['Xt'], data['yt']
  else:
    test_size = int(ttr*X.shape[0])
    X, Y, Xt, Yt = X[:-test_size,:], Y[:-test_size], X[-test_size:,:], Y[-test_size:]
  data.close()
  return X[:,:-1], Y, Xt[:,:-1], Yt

def std_cov(X, Y):
  #standardize the covariates; **last col is intercept**, so no stdization there
  x_mean = X.mean(axis=0)
  x_std = np.cov(X, rowvar=False)+1e-12*np.eye(X.shape[1])
  X = np.linalg.solve(np.linalg.cholesky(x_std), (X - x_mean).T).T
  Z = Y[:, np.newaxis]*X
  return X, Y, Z, x_mean, x_std

def get_predprobs(X_ts, thetas, x_mean, x_std):
  # normalize test dataset
  X_ts = np.atleast_2d(X_ts)
  thetas = np.atleast_2d(thetas)
  print(X_ts[:,:-1].shape, x_mean.shape)
  X_ts[:,:-1] = np.linalg.solve(np.linalg.cholesky(x_std), (X_ts[:,:-1] - x_mean).T).T
  return np.mean(1./(1+np.exp(X_ts.dot(thetas.T))), axis=1)

def compute_accuracy(Xt, Yt, thetas):
  loglikep = log_likelihood(Xt,thetas)
  logliken = log_likelihood(-Xt,thetas)
  # make predictions based on max log likelihood under each sampled parameter
  # theta
  predictions = np.ones(loglikep.shape)
  predictions[logliken > loglikep] = -1
  #compute the distribution of the error rate using max LL on the test set
  # under the posterior theta distribution
  acc = np.mean(Yt[:, np.newaxis] == predictions)
  return acc

def _compute_expected_ll(X_ts, thetas, py):
  logits = x @ theta
  ys = torch.ones_like(logits).type(torch.LongTensor) * torch.arange(self.linear.out_features)[None, :]
  ys = utils.to_gpu(ys).t()
  loglik = torch.stack([-self.cross_entropy(logits, y) for y in ys]).t()
  return torch.sum(py * loglik, dim=-1, keepdim=True)

def perturb(X_train, y_train, noise_x=(0,0,[6]), f_rate=0.1, flip=True):
  N, D = X_train.shape
  o = np.int(N*f_rate)
  idxx = np.random.choice(N, size=o)
  idxy = np.random.choice(N, size=o)
  for i in noise_x[2]: X_train[idxx,i] = np.random.normal(noise_x[0], noise_x[1], size=o)
  if flip: y_train[idxy] = -y_train[idxy]
  return X_train, y_train

def gen_synthetic(n):
  mu = np.array([0, 0])
  cov = np.eye(2)
  th = np.array([3, 3])
  X = np.random.multivariate_normal(mu, cov, n)
  ps = 1.0/(1.0+np.exp(-(X*th).sum(axis=1)))
  y = (np.random.rand(n) <= ps).astype(int)
  y[y==0] = -1
  return y[:, np.newaxis]*X, (y[:, np.newaxis]*X).mean(axis=0)

def log_likelihood(z, th):
  z = np.atleast_2d(z)
  th = np.atleast_2d(th)
  m = -z.dot(th.T)
  #idcs = m < 100
  #m[idcs] = -np.log1p(np.exp(m[idcs]))
  #m[np.logical_not(idcs)] = -m[np.logical_not(idcs)]
  m=-np.log1p(np.exp(m))
  return m

def beta_likelihood(z, th, beta):
  z = np.atleast_2d(z)
  th = np.atleast_2d(th)
  m = -z.dot(th.T)
  idcs = m < 100
  m[idcs] = -(1./beta*(1+np.exp(m[idcs]))**(-beta) - 1./(beta+1.)*((1+np.exp(m[idcs]))**(-beta-1.) + (1+np.exp(-m[idcs]))**(-beta-1.)))
  m[np.logical_not(idcs)] = -m[np.logical_not(idcs)]
  return m

def log_prior(th):
  th = np.atleast_2d(th)
  return -0.5*th.shape[1]*np.log(2.*np.pi) - 0.5*(th**2).sum(axis=1)

def log_joint(z, th, wts):
  return (wts[:, np.newaxis]*log_likelihood(z, th)).sum(axis=0) + log_prior(th)

def logistic_likelihood(z, th, wts):
  return (wts[:, np.newaxis]*log_likelihood(z, th)).sum(axis=0)

def predictive_log_likelihood(samples, Z_test, wts):
  ll = logistic_likelihood(samples, Z_test, wts).sum()
  return ll / (samples.shape[0] * Z_test.shape[0])

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
