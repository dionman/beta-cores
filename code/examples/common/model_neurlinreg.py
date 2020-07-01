import numpy as np
import scipy.linalg as sl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal as MVN


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


class BayesianRegressionDense(nn.Module):
    def __init__(self, shape, sn2=1., s=1.):
        """
        Implements Bayesian linear regression with a dense linear layer.
        :param shape: (int) Number of input features for the regression.
        :param sn2: (float) Noise variable for linear regression.
        :param s: (float) Parameter for diagonal prior on the weights of the layer.
        """
        super().__init__()
        self.in_features, self.out_features = shape
        self.y_var = sn2
        self.w_cov_prior = s * torch.eye(self.in_features)

    def forward(self, x, X_train, y_train):
        """
        Computes the predictive mean and variance for test observations given train observations
        :param x: (torch.tensor) Test observations.
        :param X_train: (torch.tensor) Training inputs.
        :param y_train: (torch.tensor) Training outputs.
        :return: (torch.tensor, torch.tensor) Predictive mean and variance.
        """
        theta_mean, theta_cov = self._compute_posterior(X_train, y_train)
        pred_mean = x @ theta_mean
        pred_var = self.y_var + torch.sum(x @ theta_cov * x, dim=-1)
        return pred_mean, pred_var[:, None]

    def _compute_posterior(self, X, y):
        """
        Computes the posterior distribution over the weights.
        :param X: (torch.tensor) Observation inputs.
        :param y: (torch.tensor) Observation outputs.
        :return: (torch.tensor, torch.tensor) Posterior mean and covariance for layer weights.
        """
        theta_cov = self.y_var * torch.inverse(X.t() @ X + self.y_var * self.w_cov_prior)
        theta_mean = theta_cov / self.y_var @ X.t() @ y
        return theta_mean, theta_cov



### MODELS ###


class NeuralLinear(torch.nn.Module):
    def __init__(self, data, linear=BayesianRegressionDense, out_features=10, **kwargs):
        """
        Neural linear module. Implements a deep feature extractor with an (approximate) Bayesian layer on top.
        :param data: (ActiveLearningDataset) Dataset.
        :param linear: (nn.Module) Defines the type of layer to implement approx. Bayes computation.
        :param out_features: (int) Dimensionality of model targets.
        :param kwargs: (dict) Additional parameters.
        """
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(data.X.shape[1], out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU()
        )

        self.linear = linear([out_features, 1], **kwargs)
        self.normalize = data.normalize

        if self.normalize:
            self.output_mean = utils.to_gpu(torch.FloatTensor([data.y_mean]))
            self.output_std = utils.to_gpu(torch.FloatTensor([data.y_std]))

        dataloader = DataLoader(Dataset(data, 'train'), batch_size=len(data.index['train']), shuffle=False)
        for (x_train, y_train) in dataloader:
            self.x_train, self.y_train = utils.to_gpu(x_train, y_train)

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
        return self.feature_extractor(x)

    def optimize(self, data, num_epochs=1000, batch_size=64, initial_lr=1e-2, weight_decay=1e-1, **kwargs):
        """
        Internal functionality to train model
        :param data: (Object) Training data
        :param num_epochs: (int) Number of epochs to train for
        :param batch_size: (int) Batch-size for training
        :param initial_lr: (float) Initial learning rate
        :param weight_decay: (float) Weight-decay parameter for deterministic weights
        :param kwargs: (dict) Optional additional arguments for optimization
        :return: None
        """
        weights = [v for k, v in self.named_parameters() if k.endswith('weight')]
        other = [v for k, v in self.named_parameters() if k.endswith('bias')]
        optimizer = torch.optim.Adam([
            {'params': weights, 'weight_decay': weight_decay},
            {'params': other},
        ], lr=initial_lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-5)
        dataloader = DataLoader(
            dataset=Dataset(data, 'train', transform=kwargs.get('transform', None)),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        for epoch in range(num_epochs):
            scheduler.step()
            losses, performances = [], []
            self.train()
            for (x, y) in dataloader:
                optimizer.zero_grad()
                x, y = utils.to_gpu(x, y)
                y_pred = self.forward(x)
                step_loss = -self._compute_log_likelihood(y, y_pred)
                step_loss.backward()
                optimizer.step()

                performance = self._evaluate_performance(y, y_pred)
                losses.append(step_loss.cpu().item())
                performances.append(performance.cpu().item())

            if epoch % 100 == 0 or epoch == num_epochs - 1:
                print('#{} loss: {:.4f}, rmse: {:.4f}'.format(epoch, np.mean(losses), np.mean(performances)))

    def get_projections(self, data, J, projection='two'):
        """
        Get projections for ACS approximate procedure
        :param data: (Object) Data object to get projections for
        :param J: (int) Number of projections to use
        :param projection: (str) Type of projection to use (currently only 'two' supported)
        :return: (torch.tensor) Projections
        """
        projections = []
        with torch.no_grad():
            theta_mean, theta_cov = self.linear._compute_posterior(self.encode(self.x_train), self.y_train)
            jitter = utils.to_gpu(torch.eye(len(theta_cov)) * 1e-4)
            try:
                theta_samples = MVN(theta_mean.flatten(), theta_cov + jitter).sample(torch.Size([J]))
            except:
                import pdb
                pdb.set_trace()

            dataloader = DataLoader(Dataset(data, 'unlabeled'), batch_size=len(data.index['unlabeled']), shuffle=False)
            for (x, _) in dataloader:
                x = utils.to_gpu(x)
                if projection == 'two':
                    for theta_sample in theta_samples:
                        projections.append(self._compute_expected_ll(x, theta_sample))
                else:
                    raise NotImplementedError

        return utils.to_gpu(torch.sqrt(1 / torch.FloatTensor([J]))) * torch.cat(projections, dim=1), torch.zeros(len(x))

    def test(self, data, **kwargs):
        """
        Test model
        :param data: (Object) Data to use for testing
        :param kwargs: (dict) Optional additional arguments for testing
        :return: (np.array) Performance metrics evaluated for testing
        """
        print("Testing...")

        test_bsz = len(data.index['test'])
        losses, performances = self._evaluate(data, test_bsz, 'test', **kwargs)
        print("predictive ll: {:.4f}, N: {}, rmse: {:.4f}".format(
            -np.mean(losses), len(data.index['train']), np.mean(performances)))
        return np.hstack(losses), np.hstack(performances)

    def get_predictions(self, x, data):
        """
        Make predictions for data
        :param x: (torch.tensor) Observations to make predictions for
        :param data: (Object) Data to use for making predictions
        :return: (np.array) Predictive distributions
        """
        self.eval()
        dataloader = DataLoader(Dataset(data, 'prediction', x_star=x), batch_size=len(x), shuffle=False)
        for (x, _) in dataloader:
            x = utils.to_gpu(x)
            y_pred = self.forward(x)
            pred_mean, pred_var = y_pred
            if self.normalize:
                pred_mean, pred_var = self.get_unnormalized(pred_mean), self.output_std ** 2 * pred_var

        return pred_mean.detach().cpu().numpy(), pred_var.detach().cpu().numpy()

    def get_unnormalized(self, output):
        """
        Unnormalize predictions if data is normalized
        :param output: (torch.tensor) Outputs to be unnormalized
        :return: (torch.tensor) Unnormalized outputs
        """
        if not self.normalize:
            return output

        return output * self.output_std + self.output_mean

    def _compute_expected_ll(self, x, theta):
        """
        Compute expected log-likelihood for data
        :param x: (torch.tensor) Inputs to compute likelihood for
        :param theta: (torch.tensor) Theta parameter to use in likelihood computations
        :return: (torch.tensor) Expected log-likelihood of inputs
        """
        pred_mean, pred_var = self.forward(x)
        const = -0.5 * torch.log(2 * np.pi * self.linear.y_var)
        z = (self.encode(x) @ theta)[:, None]
        return const - 0.5 / self.linear.y_var * (z ** 2 - 2 * pred_mean * z + pred_var + pred_mean ** 2)

    def _compute_log_likelihood(self, y, y_pred):
        """
        Compute log-likelihood of predictions
        :param y: (torch.tensor) Observations
        :param y_pred: (torch.tensor) Predictions
        :return: (torch.tensor) Log-likelihood of predictions
        """
        pred_mean, pred_variance = y_pred
        return torch.sum(utils.gaussian_log_density(inputs=y, mean=pred_mean, variance=pred_variance), dim=0)

    def _evaluate_performance(self, y, y_pred):
        """
        Evaluate performance metric for model
        """
        pred_mean, pred_variance = y_pred
        return utils.rmse(self.get_unnormalized(pred_mean), self.get_unnormalized(y))

    def _evaluate(self, data, batch_size, data_type='test', **kwargs):
        """
        Evaluate model with data
        :param data: (Object) Data to use for evaluation
        :param batch_size: (int) Batch-size for evaluation procedure (memory issues)
        :param data_type: (str) Data split to use for evaluation
        :param kwargs: (dict) Optional additional arguments for evaluation
        :return: (np.arrays) Performance metrics for model
        """

        assert data_type in ['val', 'test']
        losses, performances = [], []

        if data_type == 'val' and len(data.index['val']) == 0:
            return losses, performances

        gt.pause()
        self.eval()
        with torch.no_grad():
            dataloader = DataLoader(
                Dataset(data, data_type, transform=kwargs.get('transform', None)),
                batch_size=batch_size,
                shuffle=True
            )
            for (x, y) in dataloader:
                x, y = utils.to_gpu(x, y)
                y_pred = self.forward(x)
                pred_mean, pred_variance = y_pred
                loss = torch.sum(-utils.gaussian_log_density(y, pred_mean, pred_variance))
                avg_loss = loss / len(x)
                performance = self._evaluate_performance(y, y_pred)
                losses.append(avg_loss.cpu().item())
                performances.append(performance.cpu().item())

        gt.resume()
        return losses, performances
