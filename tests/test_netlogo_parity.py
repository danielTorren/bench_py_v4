"""Checks that the Python keeps specific NetLogo behaviours, defects included."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from bench_v4 import BENCHv4
from bench_v4.params import GRID_HALF, RECALL_PROB


def _model(case="ES", learning="Informative", seed=1):
    m = BENCHv4(case_study=case, seed=seed, learning=learning)
    m.setup()
    return m


def _place(m, xy):
    """Move agents 0..len(xy)-1 to the given patches; park the rest far away."""
    n = m.n_households
    m._grid_x = np.full(n, 0, dtype=np.int32)
    m._grid_y = np.full(n, 0, dtype=np.int32)
    # Park all other agents on a patch that no test agent touches.
    m._grid_x[:] = 20
    m._grid_y[:] = 20
    for k in range(len(xy), n):
        m._grid_x[k] = 20 + (k % 3) * 3
        m._grid_y[k] = 20 + (k // 3 % 3) * 3
    for k, (x, y) in enumerate(xy):
        m._grid_x[k], m._grid_y[k] = x, y
    m._build_neighbor_index()


def test_world_is_fixed_89_by_89():
    for case in ("ES", "NL"):
        m = _model(case)
        assert m._grid_min == -GRID_HALF and m._grid_max == GRID_HALF
        assert m.n_households == {"ES": 793, "NL": 759}[case]


def test_torus_neighbours_wrap_at_edges():
    m = _model()
    _place(m, [(-GRID_HALF, -GRID_HALF), (GRID_HALF, GRID_HALF)])
    # Opposite corners are adjacent on a torus.
    assert 1 in set(m._nbr_idx[0].tolist())
    assert 0 in set(m._nbr_idx[1].tolist())


def test_own_patch_is_not_a_neighbour():
    m = _model()
    _place(m, [(0, 0), (0, 0)])
    assert 1 not in set(m._nbr_idx[0].tolist())


def test_insulated_does_not_block_renovation():
    m = _model()
    m._U1[:] = 1.0
    m._invest1[:] = False
    m._dw_elab[:] = 3
    m._insulated[:] = True
    m._action()
    assert m._act1.all()


def test_es_group3_recall_is_effectively_1_7_percent():
    assert RECALL_PROB["ES"][3] == 1.7
    hits = trials = 0
    for seed in range(40):
        m = BENCHv4(case_study="ES", seed=seed)
        m.setup()
        m._h_group[:] = 3
        m._recall_memory()
        hits += int(m._invest1.sum())
        trials += m.n_households
    assert abs(100 * hits / trials - 1.7) < 0.3


def test_informative_broadcast_can_pass_6_6():
    m = _model(learning="Informative")
    m._act1[:] = False
    m._invest1[:] = False
    m._know[0] = 6.6
    m._learn()
    assert m._know[0] == pytest.approx(6.6 + 6.6 * 0.05)


def test_active_agent_pbci1_growth_is_not_clamped():
    m = _model(learning="No learning")
    m.learning = "Slow dynamics"
    m._act1[:] = False
    m._invest1[:] = False
    m._invest1[0] = True
    m._pbcI[0, 0] = 6.55
    m._learn()
    assert m._pbcI[0, 0] == pytest.approx(6.55 * 1.05)


def _ring(centre_xy, k):
    """Agent 0 at the centre, agents 1..k on k of its 8 neighbouring patches."""
    cx, cy = centre_xy
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    return [(cx, cy)] + [(cx + dx, cy + dy) for dx, dy in offs[:k]]


@pytest.mark.parametrize("k, expect_update", [(4, False), (5, True)])
def test_slow_dynamics_needs_more_than_four_neighbours(k, expect_update):
    m = _model(learning="Slow dynamics")
    _place(m, _ring((0, 0), k))
    m._act1[:] = False
    m._invest1[:] = False
    m._invest1[0] = True
    # Agent 0 has high values; neighbours have low values, so each
    # neighbour's neighbourhood statistic (which includes agent 0) is higher.
    m._know[0] = 7.0
    m._know[1:k + 1] = 3.0
    before = m._know[1:k + 1].copy()
    m._learn()
    changed = bool((m._know[1:k + 1] > before).any())
    assert changed is expect_update


def test_pbci1_neighbour_gate_is_6_5_in_fast_dynamics():
    m = _model(learning="Fast dynamics")
    _place(m, _ring((0, 0), 1))
    m._act1[:] = False
    m._invest1[:] = False
    m._invest1[0] = True
    m._pbcI[0, 0] = 7.0
    m._pbcI[1, 0] = 6.55          # above 6.5, below 6.6
    m._learn()
    assert m._pbcI[1, 0] == 6.55


def test_runs_with_different_seeds_differ():
    a = BENCHv4(case_study="NL", seed=1, learning="Informative")
    b = BENCHv4(case_study="NL", seed=2, learning="Informative")
    a.run()
    b.run()
    assert [s.n_renovated for s in a.history] != [s.n_renovated for s in b.history]


def test_run_is_2016_to_2050():
    m = BENCHv4(case_study="ES", seed=1)
    m.run()
    assert m.years() == list(range(2016, 2051))
