import jax
import jax.numpy as jnp
import distrax

def create_stationary_data(n_samples, n_features, rho=0.5):
    r"""
    Generates a dataset with stationary autocorrelation properties using a Toeplitz
    structured correlation matrix.
    
    Parameters
    --------------
    n_samples : int
        The number of samples to be generated.
    n_features : int
        The number of features for each sample, representing the dimensionality of the data.
    rho : float, optional, default 0.5
        The autocorrelation coefficient which determines the decay of correlation between features.
        Values should be between 0 (no autocorrelation) and 1 (full autocorrelation).
    
    Returns
    ---------
    samples : jnp.ndarray
        A (n_samples, n_features) array representing the generated dataset with stationary features.
    """
    # Define the correlation matrix using the Toeplitz structure
    col = rho ** jnp.arange(n_features)
    corr_matrix = jax.scipy.linalg.toeplitz(col)

    # Create a covariance matrix by scaling the correlation matrix
    # Here, you may tune the scale to your specific problem
    variance = jnp.ones(n_features)  # Assuming unit variance
    cov_matrix = corr_matrix * variance * variance[:, None]

    # Define a multivariate normal distribution using distrax
    mvn_dist = distrax.MultivariateNormalFullCovariance(
        loc=jnp.zeros(n_features), covariance_matrix=cov_matrix
    )

    # Sample from the distribution using JAX's random PRNG
    key = jax.random.PRNGKey(42)
    samples = mvn_dist.sample(seed=key, sample_shape=n_samples)

    return samples


def create_sparse_high_dim_data(n_samples, n_features, sparsity_level=0.57):
    r"""
    Creates a sparse dataset by applying a mask to a high-dimensional dataset that was
    originally sampled from a normal distribution.
    
    Parameters
    --------------
    n_samples : int
        The number of samples to generate.
    n_features : int
        The number of dimensions for each sample in the dataset.
    sparsity_level : float, optional, default 0.57
        Controls the level of sparsity in the dataset. A sparsity_level of 1.0 would mean all zeroes,
        while a level of 0.0 would mean a dense dataset. It should be between 0 and 1.
    
    Returns
    ---------
    sparse_data : jnp.ndarray
        A (n_samples, n_features) array containing the sparse dataset with the specified sparsity level.
    """
    key = jax.random.PRNGKey(42)
    
    # Generate dense data
    data = jax.random.normal(key, (n_samples, n_features))
    
    # Create a sparse mask
    mask = jax.random.uniform(key, (n_samples, n_features)) > sparsity_level
    
    # Apply mask to create a sparse matrix
    sparse_data = jnp.where(mask, 0, data)
    
    return sparse_data
