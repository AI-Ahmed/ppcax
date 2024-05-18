import jax
import jax.numpy as jnp
import jax.scipy.linalg
import distrax


def create_multivariate_equities_returns(n_samples,
                                         n_assets,
                                         rho=0.5,
                                         mean_return=0.0,
                                         volatility=0.1):
    r"""
    Generates synthetic equities returns data with stationary autocorrelation properties
    using a Toeplitz structured correlation matrix.
    This simulates the returns of multiple assets over time.
    
    Parameters
    --------------
    n_samples : int
        The number of time periods for which we are generating returns data.
    n_assets : int
        The number of assets for which returns are generated, representing the dimensionality of the data.
    rho : float, optional, default 0.5
        The autocorrelation coefficient which determines the decay of correlation between time periods
        for each asset. Values should be between 0 (no autocorrelation) and 1 (full autocorrelation).
    mean_return : float, optional, default 0.0
        The expected return for the assets. This can be a scalar or a jnp.ndarray of shape (n_assets,)
        in case each asset has a different expected return.
    volatility : float, optional, default 0.1
        The standard deviation of the returns (or volatility) for each asset. This can be a scalar or
        a jnp.ndarray of shape (n_assets,) in case each asset has a different volatility.
    
    Returns
    ---------
    returns : jnp.ndarray
        A (n_samples, n_assets) array representing the generated synthetic returns of the assets.
    """
    # Define the correlation matrix using the Toeplitz structure
    col = rho ** jnp.arange(n_assets)
    corr_matrix = jax.scipy.linalg.toeplitz(col)

    # Create a covariance matrix by scaling the correlation matrix
    if isinstance(volatility, (int, float)):
        volatility = jnp.ones(n_assets) * volatility
    cov_matrix = corr_matrix * volatility * volatility[:, None]

    # Define a multivariate normal distribution using distrax for the returns
    mvn_dist = distrax.MultivariateNormalFullCovariance(
        loc=mean_return * jnp.ones(n_assets)
         if isinstance(mean_return, (int, float)) else mean_return, 
        covariance_matrix=cov_matrix
    )

    # Sample from the distribution using JAX's random PRNG
    key = jax.random.PRNGKey(42)  # Use a different seed for different results
    returns = mvn_dist.sample(seed=key, sample_shape=n_samples)

    return returns


def create_sparse_equity_return_data(n_equities,
                                     n_bars,
                                     sparsity_level=0.57,
                                     mean_return=0.0,
                                     volatility=0.1):
    r"""
    Creates a dataset of equity returns that is sparse, accounting for the fact that
    equity returns do not necessarily occur at every bar.
    
    Parameters
    --------------
    n_equities : int
        The number of equities for which to generate return data.
    n_bars : int
        The number of bars which can potentially have return data. D >> N.
    sparsity_level : float, optional, default 0.57
        Controls the level of sparsity in the dataset, with 1 being completely sparse (no returns)
        and 0 being completely dense (a return at every bar for every equity).
    mean_return : float, optional, default 0.0
        The expected mean return of the assets.
    volatility : float, optional, default 0.1
        The standard deviation of the returns (or volatility) for each equity.
    
    Returns
    ---------
    sparse_returns : jnp.ndarray
        A (n_equities, n_bars) array containing the sparse dataset with the specified sparsity level.
    """
    key = jax.random.PRNGKey(42)
    
    # Simulate dense returns for each equity
    dense_returns = distrax.MultivariateNormalDiag(
        loc=jnp.ones(n_bars) * mean_return, 
        scale_diag=jnp.ones(n_bars) * volatility
    ).sample(seed=key, sample_shape=n_equities)
    
    # Generate random occurrences of return events for each equity at each bar
    sparsity_mask = jax.random.bernoulli(key, 1 - sparsity_level, (n_equities, n_bars))
    
    # Apply sparsity to the returns
    sparse_returns = jnp.where(sparsity_mask, dense_returns, 0.0)
    
    return sparse_returns