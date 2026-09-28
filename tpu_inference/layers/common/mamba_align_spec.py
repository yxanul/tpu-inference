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
"""Speculative decoding with mamba prefix caching ("align" mode).

vLLM's align-mode MambaManager gives every request, besides the block of its
running state, `num_speculative_blocks` scratch blocks right after it. A
verify window over positions `P .. P + k` (P = tokens computed before the
step) writes checkpoint `t` (the state after position `P + t`) into block
column `(seq_len - 1) // block_size + t`, as on GPU.

After sampling, `commit_align_spec_mamba_states` resolves the rollback by
copying, per request with `a` accepted drafts:

  phase 0: if the accepted prefix crossed a block boundary, the boundary
           checkpoint into the block that boundary completes, so the block
           can be cached as a prefix state;
  phase 1: checkpoint `a` (the state after the last accepted token) into
           block `(P + a) // block_size`.

The next step reads its initial state from block
`(num_computed - 1) // block_size`, exactly like non-speculative align mode.
That invariant holds whatever happens between steps (batch reordering,
requests skipped by the scheduler, async scheduling), so no per-request
rollback state has to survive across steps. Phase 0 runs before phase 1
because a boundary checkpoint at window position 0 lives in the block that
phase 1 overwrites.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.ragged_paged_attention.v3.util import \
    get_tpu_version
from tpu_inference.layers.common.sharding import ShardingAxisName


def align_spec_copy_pairs(
    block_table: jax.Array,  # [num_reqs, max_blocks_per_req]
    seq_lens: jax.Array,  # [num_reqs], this step (computed + scheduled)
    draft_lengths: jax.Array,  # [num_reqs], draft tokens verified this step
    num_accepted_drafts: jax.Array,  # [num_reqs]
    block_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Returns (src, dst) slot ids of shape [2, num_reqs], one row per phase.

    Pairs with `src == dst` are no-ops; requests without draft tokens this
    step (prefill, plain decode, padding) get `(0, 0)`.
    """
    width = block_table.shape[-1]

    def slot(col):
        col = jnp.clip(col, 0, width - 1)
        return jnp.take_along_axis(block_table, col[:, None], axis=1)[:, 0]

    active = (draft_lengths > 0) & (seq_lens > 0)
    num_computed = seq_lens - (draft_lengths + 1)
    accepted = jnp.clip(num_accepted_drafts, 0, draft_lengths)
    window_col = jnp.maximum(seq_lens - 1, 0) // block_size

    # Phase 1: the state after the last accepted token.
    src_acc = slot(window_col + accepted)
    dst_acc = slot((num_computed + accepted) // block_size)
    # Phase 0: the checkpoint at a block boundary inside the accepted prefix.
    # A window is shorter than a block, so it crosses at most one boundary.
    aligned = (num_computed + accepted + 1) // block_size * block_size
    has_boundary = active & (aligned > num_computed)
    src_bnd = slot(window_col + (aligned - num_computed - 1))
    dst_bnd = slot(aligned // block_size - 1)

    zero = jnp.zeros_like(src_acc)
    src = jnp.stack([
        jnp.where(has_boundary, src_bnd, zero),
        jnp.where(active, src_acc, zero),
    ]).astype(jnp.int32)
    dst = jnp.stack([
        jnp.where(has_boundary, dst_bnd, zero),
        jnp.where(active, dst_acc, zero),
    ]).astype(jnp.int32)
    return src, dst


def copy_state_slots(states: tuple[jax.Array, ...], src: jax.Array,
                     dst: jax.Array) -> tuple[jax.Array, ...]:
    """In place, for every array `x` in `states`: `x[dst[p, i]] = x[src[p, i]]`.

    One single-slot HBM-to-HBM DMA per pair and array, so the cost is the
    number of real copies (pairs with `src == dst` issue empty DMAs). Phases
    (rows of `src`/`dst`) run in order; pairs within a phase must not read a
    slot another pair of the same phase writes.
    """
    num_states = len(states)
    num_phases, num_pairs = src.shape

    def kernel(src_ref, dst_ref, *refs):
        state_refs = refs[num_states:2 * num_states]
        sem = refs[2 * num_states]

        def copies(phase, i):
            s = src_ref[phase * num_pairs + i]
            d = dst_ref[phase * num_pairs + i]
            n = jnp.where(s != d, 1, 0)
            return [
                pltpu.make_async_copy(ref.at[pl.ds(s, n)], ref.at[pl.ds(d, n)],
                                      sem) for ref in state_refs
            ]

        for phase in range(num_phases):

            def start(i, carry, phase=phase):
                for c in copies(phase, i):
                    c.start()
                return carry

            def wait(i, carry, phase=phase):
                for c in copies(phase, i):
                    c.wait()
                return carry

            jax.lax.fori_loop(0, num_pairs, start, 0)
            jax.lax.fori_loop(0, num_pairs, wait, 0)

    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    if get_tpu_version() == 4:
        # Same reason as the GDN kernel: XLA may place operands in v4 CMEM,
        # but the DMAs address them as HBM refs.
        states = tuple(
            pltpu.with_memory_space_constraint(x, pltpu.HBM) for x in states)
    return pl.pallas_call(
        kernel,
        out_shape=tuple(
            jax.ShapeDtypeStruct(x.shape, x.dtype) for x in states),
        in_specs=(smem_spec, smem_spec) + (hbm_spec, ) * num_states,
        out_specs=(hbm_spec, ) * num_states,
        scratch_shapes=(pltpu.SemaphoreType.DMA(()), ),
        input_output_aliases={2 + i: i
                              for i in range(num_states)},
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name="mamba_copy_state_slots",
    )(src.reshape(-1), dst.reshape(-1), *states)


@functools.partial(jax.jit,
                   static_argnames=("mesh", "block_size"),
                   donate_argnums=(0, ))
def commit_align_spec_mamba_states(
    mamba_states: tuple[tuple[tuple[jax.Array, jax.Array], ...], ...],
    block_tables: tuple[jax.Array, ...],
    seq_lens: jax.Array,
    draft_lengths: jax.Array,
    num_accepted_drafts: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    block_size: int,
) -> tuple[tuple[tuple[jax.Array, jax.Array], ...], ...]:
    """Resolves this step's verify windows into the align-mode block layout.

    Args:
        mamba_states: per mamba kv-cache group, per layer `(conv_state,
            recurrent_state)`; donated and updated in place.
        block_tables: per group, that group's block table (flattened or
            `[max_num_reqs, max_blocks_per_req]`), rank-local slot ids.
        seq_lens: `[max_num_reqs]` this step's sequence lengths.
        draft_lengths: `[padded_num_reqs]` per-DP-rank draft token counts of
            this step (the spec decode metadata layout).
        num_accepted_drafts: `[max_num_reqs]` accepted draft tokens
            (`extract_last_sampled_tokens`' mamba read offsets).
    """
    num_rows = seq_lens.shape[0]
    data = ShardingAxisName.ATTN_DATA
    head = ShardingAxisName.ATTN_HEAD
    conv_spec = P(data, None, head)
    recurrent_spec = P(data, head, None, None)
    state_specs = tuple(
        tuple((conv_spec, recurrent_spec) for _ in group)
        for group in mamba_states)
    block_tables = tuple(bt.reshape(num_rows, -1) for bt in block_tables)

    def local(states, tables, seq_lens, draft_lengths, accepted):
        rows = seq_lens.shape[0]
        draft_lengths = jnp.pad(draft_lengths,
                                (0, rows - draft_lengths.shape[0]))
        out = []
        for group, table in zip(states, tables):
            src, dst = align_spec_copy_pairs(table, seq_lens, draft_lengths,
                                             accepted[:rows], block_size)
            flat = tuple(x for layer in group for x in layer)
            copied = copy_state_slots(flat, src, dst)
            out.append(
                tuple((copied[2 * i], copied[2 * i + 1])
                      for i in range(len(group))))
        return tuple(out)

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(state_specs, tuple(P(data, None) for _ in block_tables),
                  P(data), P(data), P(data)),
        out_specs=state_specs,
        check_vma=False,
    )(mamba_states, block_tables, seq_lens, draft_lengths, num_accepted_drafts)
