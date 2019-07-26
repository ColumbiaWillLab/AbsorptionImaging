import pytest

from numpy import testing

from config import config
from models import shots


@pytest.fixture(scope="module", params=[True, False])
def cfg(request):
    """Set config and return object."""
    config.read("./tests/config.ini")
    config.fit_optical_density = request.param  # Previously returning different results
    yield config


@pytest.fixture
def shot():
    """Create shot object."""
    path = "./tests/data/saturated/2019-07-26T162222"
    bmp_paths = [f"{path}-{i}.bmp" for i in range(1, 4)]
    yield shots.Shot(name=path, bmp_paths=bmp_paths)


def test_run_fit(cfg, shot):
    """Check that the run_fit method returns and sets the correct items."""
    assert isinstance(shot.run_fit(cfg), shots.ShotFit)  # return value
    assert isinstance(shot.fit, shots.ShotFit)  # set instance variable
    assert shot.fit is shot.fit_2D


def test_atom_number(cfg, shot):
    """Numerical sanity checks for atom number."""
    # Whole Image
    whole_image = shot.atom_number
    testing.assert_approx_equal(whole_image, 2.22e9, significant=2)

    # Sigma Masked Image
    shot.run_fit(cfg)
    sigma_masked = shot.atom_number
    testing.assert_approx_equal(sigma_masked, 2.22e9, significant=2)

    # Masked atom number should be less than whole atom number (removes noise)
    assert sigma_masked < whole_image
