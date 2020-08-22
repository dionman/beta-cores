import sklearn.datasets as skl_ds
from sklearn import preprocessing
import scipy.sparse as sp
import numpy as np

def _load_svmlight_data(path):
    X, y = skl_ds.load_svmlight_file(path)
    return X, y

def load_data(path, file_type, max_data=0, max_dim=0,
              preprocess=True, include_offset=True):
    """Load data from a variety of file types.

    Parameters
    ----------
    path : string
        Data file path.

    file_type : string
        Supported file types are: 'svmlight', 'npy' (with the labels y in the
        rightmost col), 'npz', 'hdf5' (with datasets 'x' and 'y'), and 'csv'
        (with the labels y in the rightmost col)

    max_data : int
        If positive, maximum number of data points to use. If zero or negative,
        all data is used. Default is 0.

    max_dim : int
        If positive, maximum number of features to use. If zero or negative,
        all features are used. Default is 0.

    preprocess : boolean or Transformer, optional
        Flag indicating whether the data should be preprocessed. For sparse
        data, the features are scaled to [-1, 1]. For dense data, the features
        are scaled to have mean zero and variance one. Default is True.

    include_offset : boolean, optional
        Flag indicating that an offset feature should be added. Default is
        False.

    Returns
    -------
    X : array-like matrix, shape=(n_samples, n_features)

    y : int ndarray, shape=(n_samples,)
        Each entry indicates whether each example is negative (-1 value) or
        positive (+1 value)

    pp_obj : None or Transformer
        Transformer object used on data, or None if ``preprocess=False``
    """
    if not isinstance(path, str):
        raise ValueError("'path' must be a string")

    if file_type in ["svmlight", "svm"]:
        X, y = _load_svmlight_data(path)
    else:
        raise ValueError("unsupported file type, %s" % file_type)

    y_vals = set(y)
    if len(y_vals) != 2:
        raise ValueError('Only expected y to take on two values, but instead'
                         'takes on the values ' + ', '.join(y_vals))
    if 1.0 not in y_vals:
        raise ValueError('y does not take on 1.0 as one on of its values, but '
                         'instead takes on the values ' + ', '.join(y_vals))
    if -1.0 not in y_vals:
        y_vals.remove(1.0)
        print('converting y values of %s to -1.0' % y_vals.pop())
        y[y != 1.0] = -1.0

    if preprocess is False:
        pp_obj = None
    else:
        if preprocess is True:
            if sp.issparse(X):
                pp_obj = preprocessing.MaxAbsScaler(copy=False)
            else:
                pp_obj = preprocessing.StandardScaler(copy=False)
            pp_obj.fit(X)
        else:
            pp_obj = preprocess
        X = pp_obj.transform(X)

    if include_offset:
        X = preprocessing.add_dummy_feature(X)
        X = np.flip(X, -1) # move intercept to the last column of the array

    if sp.issparse(X) and (X.nnz > np.prod(X.shape) / 10 or X.shape[1] <= 20):
        print("X is either low-dimensional or not very sparse, so converting "
              "to a numpy array")
        X = X.toarray()
    if isinstance(max_data, int) and max_data > 0 and max_data < X.shape[0]:
        X = X[:max_data,:]
        y = y[:max_data]
    if isinstance(max_dim, int) and max_dim > 0 and max_dim < X.shape[1]:
        X = X[:,:max_dim]

    return X, y, pp_obj




X, y, pp_obj = load_data('covtype_train.svm', 'svm')
D = X.shape[1]
# load testing data if it exists
X_test, y_test, _ = load_data('covtype_test.svm', 'svm')
print(X, y)
np.savez('covtype', X=X, y=y, Xt=X_test, yt=y_test)
