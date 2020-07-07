import numpy as np
import scipy.linalg as sl
import pandas

def load_data(name, data_dir):
  """
  Return data from UCI sets
  :return: Inputs, outputs
  """
  np.random.seed(seed)
  if name in ['boston', 'concrete']:
    data = np.array(pandas.read_excel('{}/{}.xls'.format(data_dir, name)))
  elif name in ['energy', 'power']:
    data = np.array(pandas.read_excel('{}/{}.xlsx'.format(data_dir, name)))
  elif name in ['kin8nm', 'protein']:
    data = np.array(pandas.read_csv('{}/{}.csv'.format(data_dir, name)))
  elif name in ['naval', 'yacht']:
    data = np.loadtxt('{}/{}.txt'.format(data_dir, name))
  elif name in ['wine']:
    data = np.array(pandas.read_csv('{}/{}.csv'.format(data_dir, name), delimiter=';'))
  elif name in ['year']:
    data = np.loadtxt('{}/{}.txt'.format(data_dir, name), delimiter=',')
  else:
    raise ValueError('Unsupported dataset: {}'.format(data_dir, name))
  if name in ['energy', 'naval']:  # dataset has 2 response values
    X = data[:, :-2]
    Y = data[:, -2:-1]  # pick first response value
  else:
    X = data[:, :-1]
    Y = data[:, -1:]
  return (X, Y)

def build_synthetic_dataset(N=2000, noise_std=0.1, D=10):
  d = D+1 # dimensionality of w
  w = np.random.randn(d)
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
  return -1./2.*np.log(2.*np.pi*sigsq) - 1./(2.*sigsq)*(y[:,np.newaxis]**2 - 2*XST*y[:,np.newaxis] + XST**2)

def neurlinr_grad_x_loglikelihood(z, th, sigsq):
  pass

def weighted_post(th0, Sig0inv, sigsq, z, w):
  z = np.atleast_2d(z)
  X = z[:, :-1]
  Y = z[:, -1]
  LSigpInv = np.linalg.cholesky(Sig0inv + (w[:, np.newaxis]*X).T.dot(X)/sigsq)
  LSigp = sl.solve_triangular(LSigpInv, np.eye(LSigpInv.shape[0]), lower=True, overwrite_b = True, check_finite = False)
  mup = np.dot(LSigp.dot(LSigp.T),  np.dot(Sig0inv,th0) + (w[:, np.newaxis]*Y[:,np.newaxis]*X).sum(axis=0)/sigsq )
  return mup, LSigp, LSigpInv
