import random
import collections.abc
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Sequence, Union

import jax
from jax import jit
import jax.numpy as jnp
from jax.numpy.linalg import inv as jinv

import chex
import distrax
from distrax._src.utils import jittable

import numpy as np
from sklearn.base import BaseEstimator

random.seed(42)
np.random.seed(42)

PRNGKey = chex.PRNGKey
Array = Union[chex.Array, chex.ArrayNumpy]
IntLike = Union[int, np.int16, np.int32, np.int64]
FloatLike = Union[float, np.float16, np.float32, np.float64]


def convert_seed_and_sample_shape(
    seed: Union[IntLike, PRNGKey],
    sample_shape: Union[IntLike, Sequence[IntLike]]
) -> Tuple[PRNGKey, Tuple[int, ...]]:
  """
  Shared functionality to ensure that seeds and shapes are the right type.
  Ref: https://github.com/google-deepmind/distrax/blob/ee17707c419766252386da3337f24751a6d12905/distrax/_src/distributions/distribution.py#L312
  """

  if not isinstance(sample_shape, collections.abc.Sequence):
    sample_shape = (sample_shape,)
  sample_shape = tuple(map(int, sample_shape))

  if isinstance(seed, (int, np.signedinteger)):
    rng = jax.random.PRNGKey(seed)
  else:  # key is of type PRNGKey
    rng = seed

  return rng, sample_shape  # type: ignore[bad-return-type]


class PPCA(jittable.Jittable, BaseEstimator):

    @staticmethod
    def sample_W(seed: Union[IntLike, PRNGKey],
                 sample_shape: Union[IntLike, Sequence[IntLike]]) -> Array:
        """Samples an event.

        Parameters
        ----------
        - seed (Union[IntLike, PRNGKey]): PRNG key or integer seed.
        - sample_shape (Union[IntLike, Sequence[IntLike]]): Additional leading dimensions for sample.

        Returns
        -------
        - Array: A sample of shape.
        """
        rng, sample_shape = convert_seed_and_sample_shape(seed, sample_shape)
        return jax.random.uniform(rng, shape=sample_shape)

    def __init__(self,
                 q: IntLike = 2,
                 prior_sigma: FloatLike = 1.0,
                 seed: Union[IntLike, PRNGKey] = 42):
        r"""
        Initialize the Probabilistic PCA (PPCA) model.

        Parameters
        ----------
        - q (IntLike): Number of latent dimensions. Default ``2``.
        - prior_sigma (FloatLike): Prior variance. Default ``1.``.
        - seed (Union[IntLike, PRNGKey]): PRNG key or integer seed. Default ``42``.
        Attributes
        ---------
        - P (Array): input dataset.
        - sigma (FloatLike): Prior variance.
        - W (Array): Principal components matrix.
        - mu (Array): Mean vector.
        """
        self._sigma = prior_sigma
        self.q = q
        self.prior_sigma = prior_sigma
        self.seed = seed

    @property
    def P(self) -> Array:
        return self._P

    @property
    def W(self) -> Array:
        return self._W

    @property
    def sigma(self) -> FloatLike:
        return self._sigma

    @P.setter
    def P(self, value):
        self._P = value

    @W.setter
    def W(self, value):
        self._W = value

    @sigma.setter
    def sigma(self, value):
        self._sigma = value

    def data_reshape(self, data) -> Tuple[Array, Tuple[IntLike, IntLike]]:
        # if  (N << D), we have P as (DxN) a matrix whose columns are the data points --> Transpose the matrix
        if data.shape[0] < data.shape[1]:
            P = data.T if isinstance(data, jnp.ndarray) else jnp.asarray(data).T  # From NxD to DxN
            self.D, self.N = P.shape
            sample_shape = (self.N, self.q)
        else:
            P = data if isinstance(data, jnp.ndarray) else jnp.asarray(data)
            self.N, self.D = P.shape
            sample_shape = (self.N, self.q)
        return P, sample_shape

    def fit(self, data, use_em=True, max_iter: IntLike=20, verbose: IntLike=0):
        r"""
        Fit the Probabilistic PCA model to the input data.

        Parameters
        ----------
        - data (numpy.ndarray | Array): Input data matrix.
        - use_em (bool): Whether to use Expectation-Maximization (EM) algorithm.
        - max_iter (IntLike): Maximum number of iterations.
        - verbose (IntLike): Verbosity level.

        Returns
        -------
        - float or None: Negative log-likelihood if using EM, None if using Maximum Likelihood (ML) estimation.
        """
        self.P, sample_shape = self.data_reshape(data)

        if use_em:
            self.W: Array = self.sample_W(self.seed, sample_shape)
            n_iter = jnp.arange(max_iter).reshape(1, -1)
            ell = jax.vmap(self.__fit_em, in_axes=(0, None))(n_iter, verbose)
            return ell
        else:
            self.__fit_ml()

    def transform(self, 
                  P: Optional[Array] = None,
                  lower_dim_only: bool = False) -> Array:
        r"""
        Transform the input data into the latent space.

        Parameters
        ----------
        - P (Array): Input data matrix. If not provided,
                             the model's training data will be used.
        Returns
        -------
        - Array : Transformed data in the latent space.

        Notes
        -----
        This method transforms the input data into the latent space using the current parameters
        of the model. It calculates the latent variables based on the formula:

        z = inv((W.T @ W) + sigma * I_q) @ W.T @ {(X - mu)/(X - mu).T}  # depends on the dimension length.

        Where:
        - z is the matrix of latent variables.
        - W is the matrix of principal components.
        - sigma^2 is the noise variance parameter.
        - I_q is the q-dimensional identity matrix. (i.e., q represents `q`)
        - X is the input data matrix.
        - mu is the mean of the input data.
        """
        if P is None:
            P = self.P
        else:
            P, _ = self.data_reshape(P)

        M_inv = jinv(self.W.T @ self.W + self.sigma * np.eye(self.q))  # pylint-disable
        pmu = (P - self.mu)

        if (P is not None and P.shape[0] < P.shape[1]) or (self.N < self.D):
            if lower_dim_only:
                z = M_inv @ self.W.T @ pmu.T @ pmu
                z = z.T
            else:
                z = M_inv @ self.W.T @ pmu.T
        else:
            z = M_inv @ self.W.T @ pmu

        return z

    def fit_transform(self,
                      *args,
                      **kwargs
    ) -> Union[Tuple[Union[Array, FloatLike], Array], Array]:
        r"""
        Fit the model to the data and simultaneously transform the data.

        Returns
        -------
        - Tuple[Union[Array, FloatLike], Array]: Negative log-likelihood if using EM, None if using ML.
        - Array: Transformed data in the latent space.
        """
        ell = self.fit(*args, **kwargs)

        if isinstance(ell, Union[Array, FloatLike]):
            return ell, self.transform()
        else:
            return self.transform()

    def inverse_transform(self,
                          z: Optional[Array] = None,
                          add_noise: bool = False) -> Array:
        r"""
        Transform the latent data to the reconstructed data, optionally adding noise.

        Parameters
        ----------
        - z (Array, optional): Latent data matrix. If not provided,
            it will be inferred using the transform method.
        - add_noise (bool): Whether to add noise to the reconstructed data.

        Returns
        -------
        - Array: Transformed data in the original space,
         i.e., the reconstructed uncertainty.
        """
        if z is None:
            z = self.transform()

        if self.N < self.D:
            recon_data = self.W @ z + self.mu.T
        else:
            recon_data = self.W @ z + self.mu

        if add_noise:
            if self.N < self.D:
                d, n = recon_data.shape[0], recon_data.shape[1]
                d_iter = jnp.arange(d)
            else:
                n, d = recon_data.shape[0], recon_data.shape[1]
                d_iter = jnp.arange(d)

            k = jax.random.PRNGKey(42)
            keys = jax.random.split(k, d)

            dist = distrax.Normal(loc=0., scale=self.sigma)
            noises = jnp.stack([dist.sample(seed=keys[i], sample_shape=n)
                                for i in range(len(keys))])

            def add_noise(state, i):
                recon_data, noises = state
                if self.N < self.D:
                    recon_data = recon_data.at[i, ...].add(noises[i, ...])
                else:
                    recon_data = recon_data.at[:, i].add(noises[:, i])
                _ = 0.
                return (noises, recon_data), _

            (_, recon_data), _ = jax.lax.scan(add_noise, (recon_data, noises), d_iter)
        return recon_data

    def _ell(self,
             W: Array,
             mu: Array,
             sigma: FloatLike,
             lg_sigma: Array,
             ell_norm=True
             ) -> Array:
        r"""
        Calculate the Negative log-likelihood of the PPCA model.

        Parameters
        ----------
        - W (Array): Principal components matrix.
        - mu (Array): Mean vector.
        - sigma (FloatLike): Noise variance.
        - ell_norm (bool): Whether to normalize the log-likelihood.

        Returns
        -------
        - float: Negative Log-likelihood.
        """
        # E-step
        M_inv = jinv(W.T @ W + sigma * np.eye(W.shape[1]))
        M_inv_WT = M_inv @ W.T
        ell = jnp.float32(0.)

        def e_step(state, n):
            P, mu, W, sigma, ell = state
            p = P[:, n][:, None]
            pmu = p - mu

            def cond_N_and_cond_D(W, sigma, ell, N_lt_D, phi_shape):
                if N_lt_D:
                    phi = M_inv_WT[:, n][:, None] @ pmu.T
                    phi = jnp.diag(phi.T)
                    Phi = sigma * M_inv + phi @ phi.T
                    ell += 0.5 * jnp.trace(Phi)

                    ell += jnp.where(sigma > 1e-5,
                                    1/(2 * sigma) * jnp.float32((pmu.T @ pmu).reshape(-1,)[0]),
                                    0.0)
                    ell -= jnp.where(sigma > 1e-5,
                                    1/sigma * jnp.trace(
                                        (phi.T @ W.T[:, n][:,None] @ pmu.T).reshape(-1, 1)
                                    ), 0.0)
                else:
                    if phi_shape == 'cond_N':
                        phi = M_inv_WT @ pmu
                        Phi = sigma * M_inv + phi @ phi.T
                        ell += 0.5 * jnp.trace(Phi)
                        ell += jnp.where(sigma > 1e-5,
                                    1/(2 * sigma) * jnp.float32((pmu.T @ pmu).reshape(-1,)[0]), 0.0)
                        ell -= jnp.where(sigma > 1e-5,
                                    1/sigma * jnp.float32((phi.T @ W.T @ pmu).reshape(-1,)[0]), 0.0)
                    else:
                        # Handle the case where phi_shape is not compatible with cond_D
                        Phi = jnp.zeros_like(sigma, shape=(self.q, self.q))
                        ell = jnp.zeros_like(sigma)
                return Phi, ell

            Phi, ell = jax.lax.cond(self.N < self.D,
                                    lambda x: cond_N_and_cond_D(*x, N_lt_D=True, phi_shape='cond_N'),
                                    lambda x: cond_N_and_cond_D(*x, N_lt_D=False, phi_shape='cond_D'),
                                    (W, sigma, ell))

            ell += jnp.where(sigma > 1e-5,
                             1/(2 * sigma) * jnp.trace(W.T @ W @ Phi), 0.0) + ell

            chex.assert_shape(ell, ())  # Ensure ell has the expected shape
            return (P, mu, W, sigma, ell), ell


        n_iter = jnp.arange(self.N if self.N < self.D else self.D)
        (*params, ell), _ = jax.lax.scan(e_step, (self.P, mu, W, sigma, ell), n_iter)

        ell += jax.lax.cond(sigma > 1e-5,
                            lambda x: 0.5 * self.D * self.N * x,
                            lambda x: 0.,
                            lg_sigma)
        ell *= -1.0
        ell /= jax.lax.cond(ell_norm,
                            lambda: jnp.float32(self.D),
                            lambda: 1.)
        return ell

    def __fit_em(self, max_iter: Array, verbose: IntLike):
        r"""
        Fit the model using Expectation-Maximization (EM) algorithm.

        Parameters
        ----------
        - max_iter (Array): Maximum number of iterations.
        - verbose (IntLike): Verbosity level.

        Returns
        -------
        - Array: Negative log-likelihood.

        Notes
        -----
        This method fits the model to the input data using the Expectation-Maximization (EM) algorithm.
        It iteratively updates the parameters of the model until convergence to maximize the likelihood function.
        The algorithm consists of alternating E-steps and M-steps:

            1. E-step (Expectation Step): Estimate the latent variables based on
            the current parameters of the model.
            2. M-step (Maximization Step): Update the parameters of the model
            based on the estimated latent variables.

        During each iteration, the method updates the principal component matrix (W)
        and the noise variance parameter (sigma^2) using the current estimates
        of the latent variables. The algorithm terminates after reaching the maximum
        number of iterations.
        """
        def m_step(state, i):
            (W, sigma, S, mu, I_q, N, verb) = state

            inv_M = jinv(W.T @ W + sigma * I_q)  # q x q
            x = jinv(sigma * I_q + inv_M @ W.T @ S @ W)  # q x q

            W = S @ W @ x  # D x q
            sigma = jnp.float32(1/N * jnp.trace(S - S @ W @ inv_M @ W.T))
            lg_sigma = jnp.log(sigma)
            ell = self._ell(W=W, mu=mu, sigma=sigma, lg_sigma=lg_sigma)

            jax.lax.cond(verb == 1,
                         lambda: jax.debug.print("Iter: {}, Updated ell: {}", i+1, ell),
                         lambda: None)
            return (W, sigma, S, mu, I_q, N, verb), ell

        I_q = jnp.eye(self.q)
        self.mu = jnp.mean(self.P, axis=1)[:, np.newaxis]  # D x 1
        p_cent = self.P - self.mu

        # Sample Covariance Matrix
        if self.N < self.D:
            S = self.N**-1 * p_cent.T @ p_cent  # N x N
        else:
            S = self.D**-1 * p_cent @ p_cent.T  # N x N

        (self.W, self.sigma,
         *params), ell = jax.lax.scan(m_step, (self.W, self.sigma, S,
                                               self.mu, I_q, self.N, verbose),
                                      max_iter)
        return ell

    def __fit_ml(self):
        r"""
        Fit the model using Maximum Likelihood (ML) estimation.
        """
        self.mu = np.mean(self.P, axis=1)[:, np.newaxis]
        u, s, v = np.linalg.svd(self.P - self.mu)

        if self.q > len(s):
            ss = np.zeros(self.q)
            ss[:len(s)] = s
        else:
            ss = s[:self.q]

        ss = np.sqrt(np.maximum(0, ss**2 - self.sigma))
        self.W = u[:, :self.q].dot(np.diag(ss))

        if self.q < self.D:
            self.sigma = 1.0 / (self.D - self.q) * np.sum(s[self.q:]**2)
        else:
            self.sigma = 0.0