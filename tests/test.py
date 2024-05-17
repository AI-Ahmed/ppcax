import chex
import pytest
from src.ppcax import PPCA
from gen_data import create_stationary_data, create_sparse_high_dim_data

SEED = 42
ST_SHAPE = (10000, 10)
HD_SP_SHAPE = (50, 10000)
q = 2

@pytest.fixture
def ppca_package():
    return PPCA(q=q, prior_sigma=1.0, seed=SEED)


@pytest.fixture
def stationary_data():
    # Fixture to return stationary data
    return create_stationary_data(n_samples=ST_SHAPE[0], n_features=ST_SHAPE[1])


@pytest.fixture
def high_dim_sparse_data():
    # Fixture to return high-dimensional sparse data
    return create_sparse_high_dim_data(n_samples=HD_SP_SHAPE[0], n_features=HD_SP_SHAPE[1])


def test_init(ppca_package):
    ppca = ppca_package
    assert ppca.prior_sigma == 1.0
    assert ppca.seed == SEED
    # Add more assertions as needed


def test_fit_without_em(ppca_package, stationary_data):
    ppca = ppca_package
    ell, embedding = ppca.fit_transform(stationary_data, use_em=False)
    assert embedding is None and ell is None
    # Optionally, test parameters are learned correctly:
    # chex.assert_tree_all_finite(ppca.get_params())


def test_fit_with_em(ppca_package, stationary_data):
    ppca = ppca_package
    _, embedding = ppca.fit_transform(stationary_data, use_em=True, max_iter=20, verbose=1)
    chex.assert_shape(embedding, (q, ST_SHAPE[1]))
    # Here you might want to check if result meets certain conditions
    # for example, if the NLL reduced, or if parameters converged


def test_fit_em_high_dimensional(ppca_package, high_dim_sparse_data):
    ppca = ppca_package
    _, embedding = ppca.fit_transform(high_dim_sparse_data, use_em=True)
    chex.assert_shape(embedding, (q, HD_SP_SHAPE[1]))  # Substitute with expected shape


def test_high_dimensional_latent_dim(ppca_package, high_dim_sparse_data):
    ppca = ppca_package
    _ = ppca.fit_transform(high_dim_sparse_data, use_em=True)
    transformed_data = ppca.transform(lower_dim_only=True)
    chex.assert_shape(transformed_data, (HD_SP_SHAPE[0], q))  # Substitute with expected shape
    chex.assert_rank(transformed_data, q)
    # Further assertions to check if transform works correctly based on the expected outcome