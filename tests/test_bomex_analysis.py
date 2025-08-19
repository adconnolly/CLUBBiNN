from subgrid_parameterization.preprocess import SAM_helpers as sam
from subgrid_parameterization.preprocess.C14 import get_C14
from subgrid_parameterization.preprocess.mixing_length import get_mixing_length

import numpy as np
import pytest


def test_dispersion_stats(bomex_dataset):
    """Regression test based on analysis of BOMEX dataset.

    Based on the code in the `analysis_notebooks/plot_distribution_C14_BOMEX.ipynb`
    notebook.
    """
    dispersion = sam.get_disp(bomex_dataset)

    assert 9615 == np.sum(dispersion < 0)
    assert 3486 == np.sum(dispersion == 0)
    assert 219 == np.sum(dispersion > 0)


def test_total_kinetic_energy_stats(bomex_dataset):
    """Regression test based on analysis of BOMEX dataset.

    Based on the code in the `analysis_notebooks/plot_distribution_C14_BOMEX.ipynb`
    notebook.
    """
    e = sam.get_TKE(bomex_dataset)

    assert 0 == np.sum(e < 0)
    assert 0 == np.sum(e == 0)
    assert 13320 == np.sum(e > 0)


def test_C14_stats(bomex_dataset):
    """Regression test based on analysis of BOMEX dataset.

    Based on the code in the `analysis_notebooks/plot_distribution_C14_BOMEX.ipynb`
    notebook.
    """
    c14 = get_C14(bomex_dataset)

    assert 219 == np.sum(c14 < 0)
    assert 3486 == np.sum(c14 == 0)
    assert 9615 == np.sum(c14 > 0)
    assert 367 == np.sum(c14 > 2)

    c14_nonNeg = c14[c14 >= 0]
    c14_pos = c14[c14 > 0]

    assert pytest.approx(0.612029828700826) == np.mean(c14)
    assert pytest.approx(0.6222868171826276) == np.mean(c14_nonNeg)
    assert pytest.approx(0.8479021936463447) == np.mean(c14_pos)


def test_interactions_stats(bomex_dataset):
    """Regression test based on analysis of BOMEX dataset.

    Based on the code in the `analysis_notebooks/plot_distribution_C14_BOMEX.ipynb`
    notebook.

    Checks regression of statistics computed from all:
     - C14
     - dispersion
     - TKE
     - mixing length

    """
    e = sam.get_TKE(bomex_dataset)
    dispersion = sam.get_disp(bomex_dataset)
    c14 = get_C14(bomex_dataset)
    l, *_ = get_mixing_length(bomex_dataset)

    # Clipping the C14 value
    c14max = 2
    emin = (-1.5 * dispersion[dispersion < 0] * l[dispersion < 0] / c14max) ** (2 / 3)
    assert 367 == np.sum(e[dispersion < 0] < emin)

    # Stats of clipped C14
    c14_pos = c14[c14 > 0]
    c14_clip = c14_pos[e[dispersion < 0] > emin]

    assert pytest.approx(0.7743455787905932) == np.mean(c14_clip)
