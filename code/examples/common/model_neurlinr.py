import numpy as np
import scipy.linalg as sl
import pandas

def load_data(name, data_dir):
  """
  Return data from UCI sets
  :return: Inputs, outputs
  """
  if name in [ 'concrete']:
    data = np.array(pandas.read_excel('{}/{}.xls'.format(data_dir, name)))
  elif name in ['energy', 'power']:
    data = np.array(pandas.read_excel('{}/{}.xlsx'.format(data_dir, name)))
  elif name in ['kin8nm', 'protein']:
    data = np.array(pandas.read_csv('{}/{}.csv'.format(data_dir, name)))
  elif name in ['naval', 'yacht']:
    data = np.loadtxt('{}/{}.txt'.format(data_dir, name))
  elif name in ['wine']:
    data = np.array(pandas.read_csv('{}/{}.csv'.format(data_dir, name), delimiter=';'))
  elif name in ['boston']:
    from sklearn.datasets import load_boston
    data = load_boston()
  elif name in ['year']:
    data = np.loadtxt('{}/{}.txt'.format(data_dir, name), delimiter=',')
  else:
    raise ValueError('Unsupported dataset: {}'.format(data_dir, name))
  if name in ['energy', 'naval']:  # dataset has 2 response values
    X = data[:, :-2]
    Y = data[:, -2:-1]  # pick first response value
  elif name in ['boston']:
    X = data['data']
    Y = data['target'][:, np.newaxis]
  else:
    X = data[:, :-1]
    Y = data[:, -1:]
  return (X, Y)

def std_cov(X, Y, mean_=None, std_=None):
  #standardize the covariates; **last col is intercept**, so no stdization there
  if (mean_ is None) & (std_ is None): # train datapoints
    x_mean = X.mean(axis=0)
    x_std = np.cov(X, rowvar=False)+1e-12*np.eye(X.shape[1])
  else: # test datapoints
    x_mean = mean_
    x_std = std_
  X = np.linalg.solve(np.linalg.cholesky(x_std), (X - x_mean).T).T
  Z = Y[:, np.newaxis]*X
  return X, Y, Z, x_mean, x_std


def perturb(X_train, y_train, noise_x=(0,5), f_rate=0.1, structured=False, mean_val=0.1, std_val=1., theta_val=-1.):
  N, D = X_train.shape
  o = np.int(N*f_rate)
  idxx = np.random.choice(N, size=o)
  if not structured: # random noise/mislabeling in input/output space
    idxy = np.random.choice(N, size=o)
    idcs =  np.random.choice(D,int(D/2.),replace=False)
    for i in idcs: # replace half of the features with gaussian noise
      X_train[idxx,i] = np.random.normal(noise_x[0], noise_x[1], size=o)
    if o>0: y_train[idxy] = np.random.normal(noise_x[0], noise_x[1], size=o)
  else: # structured perturbation for desirable adversarial outcome
    NotImplementedError
  outidx = np.unique(np.concatenate([idxx, idxy]))
  return X_train, y_train

def build_synthetic_dataset(N=2000, noise_std=0.1, D=40):
  d = D+1 # dimensionality of w
  w = 10+np.random.randn(d)
  X = np.random.randn(N, d)
  X[:,-1] = 1.
  Y = (np.dot(X, w) + np.random.normal(0, noise_std, size=N))[:,np.newaxis]
  return X, Y

def neurlinr_loglikelihood(z, th, sigsq):
  z = np.atleast_2d(z)
  x = z[:, :-1]
  y = z[:, -1]
  th = np.atleast_2d(th)
  XST = x.dot(th.T)
  vals= -1./2.*np.log(2.*np.pi*sigsq) - 1./(2.*sigsq)*(y[:,np.newaxis]**2 - 2*XST*y[:,np.newaxis] + XST**2)
  print('vals = ', vals)
  return vals

def neurlinr_grad_x_loglikelihood(z, th, sigsq):
  pass

def neurlinr_beta_likelihood(z, th, beta, sigsq):
  z = np.atleast_2d(z)
  x = z[:, :-1]
  y = z[:, -1]
  th = np.atleast_2d(th)
  XST = x.dot(th.T)
  vals =  (-(beta+1.)/beta*(2.*np.pi*sigsq)**(-beta/2.)*
            (np.exp(-beta/(2.*sigsq)*(y[:,np.newaxis]**2 - 2*XST*y[:,np.newaxis] + XST**2))
                -1./(2*np.pi*sigsq**(0.5)*(1+beta))))
  print('vals : ', y[:,np.newaxis]**2 )
  return vals


def neurlinr_beta_gradient(z, th, beta, Siginv, logdetSig):
  pass

def weighted_post(th0, Sig0inv, sigsq, z, w):
  z = np.atleast_2d(z)
  X = z[:, :-1]
  Y = z[:, -1]
  LSigpInv = np.linalg.cholesky(Sig0inv + (w[:, np.newaxis]*X).T.dot(X)/sigsq)
  LSigp = sl.solve_triangular(LSigpInv, np.eye(LSigpInv.shape[0]), lower=True, overwrite_b=True, check_finite=False)
  mup = np.dot(LSigp.dot(LSigp.T),  np.dot(Sig0inv,th0) + (w[:, np.newaxis]*Y[:,np.newaxis]*X).sum(axis=0)/sigsq)
  return mup, LSigp, LSigpInv
