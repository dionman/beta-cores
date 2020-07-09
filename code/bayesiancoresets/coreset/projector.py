import numpy as np
from ..util.errors import NumericalPrecisionError
import torch

class Projector(object):
    def project(self, pts, grad=False):
        raise NotImplementedError

    def update(self, wts, pts):
        raise NotImplementedError

class BlackBoxProjector(Projector):
    def __init__(self, sampler, projection_dimension, loglikelihood, grad_loglikelihood=None, **kwargs):
        self.projection_dimension = projection_dimension
        self.sampler = sampler
        self.loglikelihood = loglikelihood
        self.grad_loglikelihood = grad_loglikelihood
        self.update(np.array([]), np.array([]))
        self.encoder = None
        if 'nl' in kwargs: # encode pts to a learned feature space
            self.encoder = kwargs['nl']

    def project(self, pts, grad=False):
        if self.encoder: lls = self.loglikelihood(pts, self.samples, self.encoder)
        else: lls = self.loglikelihood(pts, self.samples)
        lls -= lls.mean(axis=1)[:,np.newaxis]
        if grad:
            if self.grad_loglikelihood is None:
                raise ValueError('grad_loglikelihood was requested but not initialized in BlackBoxProjector.project')
            glls = self.grad_loglikelihood(pts, self.samples)
            glls -= glls.mean(axis=2)[:, :, np.newaxis]
            return lls, glls
        else:
            return lls

    def update(self, wts, pts):
        self.samples = self.sampler(self.projection_dimension, wts, pts)

class BetaBlackBoxProjector(Projector):
    def __init__(self, sampler, projection_dimension, beta_likelihood, loglikelihood, beta_gradient):
        self.projection_dimension = projection_dimension
        self.sampler = sampler
        self.beta_likelihood = beta_likelihood
        self.loglikelihood = loglikelihood
        self.beta_gradient = beta_gradient
        self.update(np.array([]), np.array([]))
        if 'nl' in kwargs: # encode pts to a learned feature space
            self.encoder = kwargs['nl']

    def project_f(self, pts, beta, grad=False):
        # projections using beta-divergence
        if self.encoder:
          pts = pts.astype(np.float32)
          pts = np.hstack((self.encoder.encode(torch.from_numpy(pts[:, :-1])).detach().numpy(), pts[:,-1][:,np.newaxis]))
        bls = self.beta_likelihood(pts, self.samples, beta)
        bls -= bls.mean(axis=1)[:,np.newaxis]
        if grad:
            if self.beta_gradient is None:
                raise ValueError('grad_loglikelihood was requested but not initialized in BlackBoxProjector.project')
            glls = self.beta_gradient(pts, self.samples, beta)
            glls -= glls.mean(axis=1)[:,np.newaxis]
            return bls, glls
        else:
            return bls

    def update(self, wts, pts):
        self.samples = self.sampler(self.projection_dimension, wts, pts)
