# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.layers.common.mamba_align_spec import (
    align_spec_copy_pairs, commit_align_spec_mamba_states, copy_state_slots)

BLOCK = 8
WIDTH = 6


def _pairs(rows):
    """rows: (num_computed, num_draft, num_accepted); block table row r maps
    column c to slot 100 * (r + 1) + c."""
    num_rows = len(rows)
    table = np.array([[100 * (r + 1) + c for c in range(WIDTH)]
                      for r in range(num_rows)],
                     dtype=np.int32)
    seq_lens = np.array(
        [p + d + 1 if d > 0 else (p + 1 if p >= 0 else 0) for p, d, _ in rows],
        dtype=np.int32)
    drafts = np.array([d for _, d, _ in rows], dtype=np.int32)
    accepted = np.array([a for _, _, a in rows], dtype=np.int32)
    src, dst = align_spec_copy_pairs(jnp.asarray(table), jnp.asarray(seq_lens),
                                     jnp.asarray(drafts),
                                     jnp.asarray(accepted), BLOCK)
    return np.asarray(src), np.asarray(dst)


@pytest.mark.parametrize(
    "row, boundary, accepted",
    [
        # Window 2..5 in block 0, 2 drafts accepted: state after position 4
        # (window column 0 + 2) back into block 0; no boundary.
        ((2, 3, 2), (0, 0), (102, 100)),
        # Nothing accepted, no boundary: checkpoint 0 already is block 0.
        ((2, 3, 0), (0, 0), (100, 100)),
        # Window 5..8 ends in block 1 (window column 1); 1 accepted stops at
        # position 6, before the boundary: roll back into block 0.
        ((5, 3, 1), (0, 0), (102, 100)),
        # 2 accepted reach position 7, completing block 0 exactly: the same
        # checkpoint is the boundary and the accepted state.
        ((5, 3, 2), (103, 100), (103, 100)),
        # 3 accepted pass the boundary: checkpoint 2 completes block 0,
        # checkpoint 3 (position 8) is the running state of block 1.
        ((5, 3, 3), (103, 100), (104, 101)),
        # Window 7..10: position 7 (checkpoint 0, window column 1) completes
        # block 0 whatever is accepted; phase 1 then overwrites block 1,
        # which phase 0 reads from - hence the phase order.
        ((7, 3, 2), (101, 100), (103, 101)),
        ((7, 3, 0), (101, 100), (101, 100)),
        # No drafts (prefill / plain decode): no copies.
        ((20, 0, 0), (0, 0), (0, 0)),
    ])
def test_align_spec_copy_pairs(row, boundary, accepted):
    src, dst = _pairs([row])
    assert (src[0, 0], dst[0, 0]) == boundary
    assert (src[1, 0], dst[1, 0]) == accepted


def test_align_spec_copy_pairs_padding_rows():
    # Padded rows have seq_len 0 and no drafts.
    src, dst = _pairs([(5, 3, 3), (-1, 0, 0)])
    assert (src[:, 1] == 0).all() and (dst[:, 1] == 0).all()


def _on_tpu():
    return jax.devices()[0].platform == "tpu"


@pytest.mark.skipif(not _on_tpu(), reason="Pallas TPU kernel")
def test_copy_state_slots_phases():
    rng = np.random.default_rng(0)
    conv = jnp.asarray(rng.standard_normal((32, 6, 256), dtype=np.float32))
    rec = jnp.asarray(rng.standard_normal((32, 4, 128, 128), dtype=np.float32))
    old = (np.asarray(conv), np.asarray(rec))
    # Phase 0 reads slot 11, phase 1 overwrites it; slot 20 -> 21 plus
    # no-op pairs (src == dst).
    src = jnp.array([[11, 20, 0, 5], [13, 0, 7, 5]], dtype=jnp.int32)
    dst = jnp.array([[10, 21, 0, 5], [11, 0, 7, 5]], dtype=jnp.int32)
    new = jax.jit(copy_state_slots, donate_argnums=(0, ))((conv, rec), src,
                                                          dst)
    for x_new, x_old in zip(new, old):
        expected = x_old.copy()
        expected[10] = x_old[11]
        expected[21] = x_old[20]
        expected[11] = x_old[13]
        np.testing.assert_array_equal(np.asarray(x_new), expected)


@pytest.mark.skipif(not _on_tpu(), reason="Pallas TPU kernel")
def test_commit_align_spec_mamba_states():
    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    mesh = jax.sharding.Mesh(devices, axis_names=("data", "model"))
    rng = np.random.default_rng(1)
    rows = [(5, 3, 3), (7, 3, 2), (2, 3, 0), (-1, 0, 0)]
    num_rows = len(rows)
    num_slots = 100 * (num_rows + 1) + WIDTH

    def layer():
        return (jnp.asarray(
            rng.standard_normal((num_slots, 6, 256), dtype=np.float32)),
                jnp.asarray(
                    rng.standard_normal((num_slots, 2, 128, 128),
                                        dtype=np.float32)))

    states = ((layer(), layer()), (layer(), ))
    old = jax.tree.map(np.asarray, states)
    table = np.array([[100 * (r + 1) + c for c in range(WIDTH)]
                      for r in range(num_rows)],
                     dtype=np.int32)
    seq_lens = np.array([p + d + 1 if d > 0 else 0 for p, d, _ in rows],
                        dtype=np.int32)
    drafts = np.array([d for _, d, _ in rows], dtype=np.int32)
    accepted = np.array([a for _, _, a in rows], dtype=np.int32)
    src, dst = (
        np.asarray(x)
        for x in align_spec_copy_pairs(jnp.asarray(table), jnp.asarray(
            seq_lens), jnp.asarray(drafts), jnp.asarray(accepted), BLOCK))

    with jax.set_mesh(mesh):
        new = commit_align_spec_mamba_states(
            states, (jnp.asarray(table.reshape(-1)), ) * 2,
            jnp.asarray(seq_lens),
            jnp.asarray(drafts[:3]),
            jnp.asarray(accepted),
            mesh=mesh,
            block_size=BLOCK)

    for group_new, group_old in zip(new, old):
        for layer_new, layer_old in zip(group_new, group_old):
            for x_new, x_old in zip(layer_new, layer_old):
                expected = x_old.copy()
                for phase in range(2):
                    snapshot = expected.copy()
                    for s, d in zip(src[phase], dst[phase]):
                        if s != d:
                            expected[d] = snapshot[s]
                np.testing.assert_array_equal(np.asarray(x_new), expected)
