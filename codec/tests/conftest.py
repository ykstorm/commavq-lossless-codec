import numpy as np
import pytest

@pytest.fixture
def rng():
    return np.random.default_rng(1234)

@pytest.fixture
def small_codebook(rng):
    # 16 codes in 4-D embedding space, well separated
    return rng.standard_normal((16, 4)).astype(np.float64)
