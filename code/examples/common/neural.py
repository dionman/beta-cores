import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal as MVN

class BayesianRegressionDense(nn.Module):
  def __init__(self, shape, sigmasq=1., s=1.):
    """
    Implements Bayesian linear regression with a dense linear layer.
    """
    super().__init__()
    self.in_features, self.out_features = shape
    self.y_var = sigmasq
    self.w_cov_prior = s * torch.eye(self.in_features)

  def forward(self, x, X_train, y_train):
    """
    Computes the predictive mean and variance for observations given train observations
    :return: (torch.tensor, torch.tensor) Predictive mean and variance.
    """
    theta_mean, theta_cov = self._compute_posterior(X_train, y_train)
    pred_mean = x @ theta_mean
    pred_var = self.y_var + torch.sum(x @ theta_cov * x, dim=-1)
    return pred_mean, pred_var[:, None]

  def _compute_posterior(self, X, y):
    """
    Computes the posterior distribution over the weights.
    :return: (torch.tensor, torch.tensor) Posterior mean and covariance for layer weights.
    """
    theta_cov = self.y_var * torch.inverse(X.t() @ X + self.y_var * self.w_cov_prior)
    theta_mean = theta_cov / self.y_var @ X.t() @ y
    return theta_mean, theta_cov


### MODELS ###
class NeuralLinear(torch.nn.Module):
  def __init__(self, Z, linear=BayesianRegressionDense, out_features=30, normalize=True):
    """
    Neural linear module. Implements a deep feature extractor with an (approximate) Bayesian layer on top.
    :param linear: (nn.Module) Defines the type of layer to implement approx. Bayes computation.
    :param out_features: (int) Dimensionality of model targets.
    """
    super().__init__()
    self.feature_extractor = nn.Sequential(
      nn.Linear(Z[:,:-1].shape[1], out_features),
      nn.BatchNorm1d(out_features),
      nn.ReLU(),
      nn.Linear(out_features, out_features),
      nn.BatchNorm1d(out_features),
      nn.ReLU()
      )
    self.linear = linear([out_features, 1])
    X, Y = Z[:, :-1], Z[:, -1]
    self.x_train, self.y_train = torch.from_numpy(X), torch.from_numpy(Y)
    self.normalize = True
    if self.normalize:
        self.output_mean = torch.FloatTensor([torch.mean(self.y_train)])
        self.output_std = torch.FloatTensor([torch.std(self.y_train)])

  def update_batch(self, Z):
    X, Y = Z[:, :-1], Z[:, -1]
    self.x_train, self.y_train = torch.from_numpy(X), torch.from_numpy(Y)
    self.normalize = True
    if self.normalize:
      self.output_mean = torch.FloatTensor([torch.mean(self.y_train)])
      self.output_std = torch.FloatTensor([torch.std(self.y_train)])

  def forward(self, x):
    """
    Make prediction with model
    :param x: (torch.tensor) Inputs.
    :return: (torch.tensor) Predictive distribution (may be tuple)
    """
    return self.linear(self.encode(x), self.encode(self.x_train), self.y_train)

  def encode(self, x):
    """
    Use feature extractor to get features from inputs
    :param x: (torch.tensor) Inputs
    :return: (torch.tensor) Feature representation of inputs
    """
    if x.shape[0]==1:
      self.feature_extractor.eval()
    return self.feature_extractor(x)

  def optimize(self, wts, pts, num_epochs=1000, initial_lr=1e-2, weight_decay=1e-1, **kwargs):
    """
    Internal functionality to train model
    :param num_epochs: (int) Number of epochs to train for
    :param initial_lr: (float) Initial learning rate
    :param weight_decay: (float) Weight-decay parameter for deterministic weights
    :param kwargs: (dict) Optional additional arguments for optimization
    :return: None
    """
    weights = [v for k, v in self.named_parameters() if k.endswith('weight')]
    other = [v for k, v in self.named_parameters() if k.endswith('bias')]
    optimizer = torch.optim.Adagrad([
        {'params': weights, 'weight_decay': weight_decay},
        {'params': other},
      ], lr=initial_lr)
    batch_size = self.get_batch_size(pts.shape[0])
    print('points shape : ', pts.shape)
    print('batch size = ', batch_size)
    dataloader = DataLoader(
            dataset=data.TensorDataset(wts, pts),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-5)
    for epoch in range(num_epochs):
      scheduler.step()
      losses, performances = [], []
      self.train()
      for (w, p) in dataloader:
        x,y=p[:,:-1],p[:,-1]
        optimizer.zero_grad()
        y_pred = self.forward(x)
        step_loss = -self._compute_log_likelihood(y, y_pred, w)
        step_loss.backward()
        optimizer.step()
        performance = self._evaluate_performance(y, y_pred)
        losses.append(step_loss.cpu().item())
        performances.append(performance.cpu().item())
        #if epoch % 10 == 0 or epoch == num_epochs - 1:
        #  print('#{} loss: {:.4f}, rmse: {:.4f}'.format(epoch, np.mean(losses), np.mean(performances)))

  def get_batch_size(self, num_points):
    # computes the closest power of two that is smaller or equal than num_points/2
    batch_sizes = 2**np.arange(10)
    if num_points in batch_sizes:
        return int(num_points / 2)
    else:
        return int(batch_sizes[np.sum((num_points / 2) > batch_sizes) - 1])


  def test(self, test_data, **kwargs):
    """
    Test model
    :param data: (Object) Data to use for testing
    :param kwargs: (dict) Optional additional arguments for testing
    :return: (np.array) Performance metrics evaluated for testing
    """
    print("Testing...")
    test_bsz = len(test_data)
    losses, performances = self._evaluate(test_data, test_bsz, **kwargs)
    print("predictive ll: {:.4f}, rmse: {:.4f}".format(
            -np.mean(losses), np.mean(performances)))
    return np.hstack(losses), np.hstack(performances)


  def _evaluate(self, data, batch_size, **kwargs):
    """
    Evaluate model with data
    :param data: (Object) Data to use for evaluation
    :param batch_size: (int) Batch-size for evaluation procedure (memory issues)
    :param data_type: (str) Data split to use for evaluation
    :param kwargs: (dict) Optional additional arguments for evaluation
    :return: (np.arrays) Performance metrics for model
    """
    losses, performances = [], []
    self.eval()
    with torch.no_grad():
      dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
      for p in dataloader:
        x,y = p[:,:-1],p[:,-1]
        y_pred = self.forward(x)
        pred_mean, pred_variance = y_pred
        loss = torch.sum(-self.gaussian_log_density(y, pred_mean, pred_variance))
        avg_loss = loss / len(x)
        performance = self._evaluate_performance(y, y_pred)
        losses.append(avg_loss.cpu().item())
        performances.append(performance.cpu().item())
    return losses, performances

  def _evaluate_prior(self, test_data, prior_mean, prior_variance, **kwargs):
    print("Testing...")
    test_bsz = len(test_data)
    losses, performances = [], []
    self.eval()
    dataloader = DataLoader(test_data, batch_size=test_bsz, shuffle=False)
    for p in dataloader:
      x,y = p[:,:-1],p[:,-1]
      prior_variance = prior_variance.unsqueeze_(-1)
      y_pred = (prior_mean, prior_variance)
      loss = torch.sum(-self.gaussian_log_density(y, prior_mean, prior_variance))
      avg_loss = loss / len(x)
      performance = self._evaluate_performance(y, y_pred)
      losses.append(avg_loss.cpu().item())
      performances.append(performance.cpu().item())
    print("predictive ll: {:.4f}, rmse: {:.4f}".format(
            -np.mean(losses), np.mean(performances)))
    return np.hstack(losses), np.hstack(performances)

  def _evaluate_performance(self, y, y_pred):
    """
    Evaluate performance metric for model
    """
    pred_mean, pred_variance = y_pred
    return self.rmse(self.get_unnormalized(pred_mean), self.get_unnormalized(y))


  def get_unnormalized(self, output):
    """
    Unnormalize predictions if data is normalized
    :param output: (torch.tensor) Outputs to be unnormalized
    :return: (torch.tensor) Unnormalized outputs
    """
    if not self.normalize:
      return output
    return output * self.output_std + self.output_mean


  def gaussian_log_density(self, inputs, mean, variance):
    """
    Compute the Gaussian log-density of a vector for a given distribution
    :param inputs: (torch.tensor) Inputs for which log-pdf should be evaluated
    :param mean: (torch.tensor) Mean of the Gaussian distribution
    :param variance: (torch.tensor) Variance of the Gaussian distribution
    :return: (torch.tensor) log-pdf of the inputs N(inputs; mean, variance)
    """
    d = inputs.shape[-1]
    xc = inputs - mean
    return -0.5 * (torch.sum((xc * xc) / variance, dim=-1)
                   + torch.sum(torch.log(variance), dim=-1) + d * np.log(2*np.pi))


  def _compute_log_likelihood(self, y, y_pred, w):
    """
    Compute log-likelihood of predictions
    :param y: (torch.tensor) Observations
    :param y_pred: (torch.tensor) Predictions
    :return: (torch.tensor) Log-likelihood of predictions
    """
    pred_mean, pred_variance = y_pred
    return torch.sum(w*self.gaussian_log_density(inputs=y, mean=pred_mean, variance=pred_variance), dim=0)

  def rmse(self, y1, y2):
    """
    Compute root mean square error between two vectors
    :param y1: (torch.tensor) first vector
    :param y2: (torch.tensor) second vector
    :return: (torch.scalar) root mean square error
    """
    return torch.sqrt(torch.mean((y1 - y2)**2))
