# SPDX-License-Identifier: Apache-2.0
"""Debug-only provenance tracer for mamba/GDN prefix caching (align mode).

Enable with PREFIX_TRACE=1 (use with --no-async-scheduling so token ids are
final on the host). For every step it derives, exactly like
`gdn_attention_op` does, the state slot each request reads its initial state
from and the slot it writes its new state to, and keeps a host-side record of
what every slot holds: (hash of the token prefix, prefix length, checksum of
the slot's recurrent state). Before each step it checks that every read slot
still holds exactly the requesting sequence's prefix at the expected length,
and that no slot changed without being written. Any violation is logged with
full context; the first one pinpoints the bookkeeping bug.
"""

import hashlib
import os

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("PREFIX_TRACE", "0") == "1"


def _prefix_hash(tokens: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(tokens,
                                             dtype=np.int32).tobytes()).hexdigest()[:10]


class PrefixTracer:

    def __init__(self, runner):
        self.runner = runner
        self.step = 0
        self.prov: dict[int, tuple[str, int, float, str, int]] = {}
        self.violations = 0
        self._gid = None
        self._cache_idx = None
        self._block_size = None
        self._checksum_fn = jax.jit(lambda s: jnp.sum(
            jnp.abs(s.astype(jnp.float32)), axis=tuple(range(1, s.ndim))))

    def _resolve(self) -> bool:
        if self._gid is not None:
            return True
        from vllm.v1.kv_cache_interface import MambaSpec
        cfg = self.runner.kv_cache_config
        for gid, group in enumerate(cfg.kv_cache_groups):
            spec = group.kv_cache_spec
            if isinstance(spec, MambaSpec):
                self._gid = gid
                self._block_size = self.runner.cache_config.mamba_block_size or spec.block_size
                self._cache_idx = self.runner.layer_name_to_kvcache_index[
                    group.layer_names[0]]
                logger.warning(
                    "[prefix-trace] tracing mamba group %d (layer %s, cache idx %d, "
                    "block size %d)", gid, group.layer_names[0], self._cache_idx,
                    self._block_size)
                return True
        return False

    def _checksums(self) -> np.ndarray:
        conv, rec = self.runner.kv_caches[self._cache_idx]
        return np.asarray(jax.device_get(self._checksum_fn(rec)))

    def _plan(self, scheduler_output):
        """[(req_id, num_computed, seq_len, read_slot, write_slot, row)]"""
        ib = self.runner.input_batch
        table = ib.block_table[self._gid].get_cpu_tensor()
        table = np.asarray(table)
        num_slots = self.runner.kv_caches[self._cache_idx][1].shape[0]
        out = []
        for req_id, n_sched in scheduler_output.num_scheduled_tokens.items():
            idx = ib.req_id_to_index.get(req_id)
            if idx is None:
                continue
            nc = int(ib.num_computed_tokens_cpu[idx])
            seq_len = nc + int(n_sched)
            row = table[idx]
            rcol = max(nc - 1, 0) // self._block_size
            wcol = max(seq_len - 1, 0) // self._block_size
            rslot = int(np.clip(row[rcol], 0, num_slots - 1))
            wslot = int(np.clip(row[wcol], 0, num_slots - 1))
            out.append((req_id, nc, seq_len, rslot, wslot, row, rcol, wcol))
        return out

    def before_step(self, scheduler_output) -> None:
        if not self._resolve():
            return
        self.step += 1
        self._plan_cache = self._plan(scheduler_output)
        self._pre = self._checksums()
        ib = self.runner.input_batch
        for (req, nc, seq_len, rslot, wslot, row, rcol, wcol) in self._plan_cache:
            nz = {int(i): int(b) for i, b in enumerate(row[:wcol + 2]) if b}
            logger.warning(
                "[prefix-trace] step %d req %s computed=%d -> %d | read col %d slot %d | "
                "write col %d slot %d | mamba row %s", self.step, req[-8:], nc,
                seq_len, rcol, rslot, wcol, wslot, nz)
            if nc == 0:
                continue  # fresh state, nothing read
            idx = ib.req_id_to_index[req]
            want = _prefix_hash(ib.token_ids_cpu[idx, :nc])
            got = self.prov.get(rslot)
            if got is None:
                self._violation("READ-UNWRITTEN", req, nc, rslot,
                                f"slot {rslot} never written by a traced step")
                continue
            h, length, csum, owner, wstep = got
            if length != nc or h != want:
                self._violation(
                    "READ-WRONG-PREFIX", req, nc, rslot,
                    f"slot holds len={length} hash={h} (written by {owner[-8:]} "
                    f"step {wstep}); request needs len={nc} hash={want}")
            elif abs(float(self._pre[rslot]) - csum) > 1e-3 * max(1.0, abs(csum)):
                self._violation(
                    "READ-MUTATED", req, nc, rslot,
                    f"slot checksum {float(self._pre[rslot]):.6g} != {csum:.6g} "
                    f"recorded when {owner[-8:]} wrote it at step {wstep}")

    def after_step(self) -> None:
        if self._gid is None or not getattr(self, "_plan_cache", None):
            return
        post = self._checksums()
        ib = self.runner.input_batch
        written = set()
        for (req, nc, seq_len, rslot, wslot, row, rcol, wcol) in self._plan_cache:
            idx = ib.req_id_to_index.get(req)
            if idx is None:
                continue
            if wslot in written:
                self._violation("DOUBLE-WRITE", req, seq_len, wslot,
                                "two requests wrote the same slot this step")
            written.add(wslot)
            self.prov[wslot] = (_prefix_hash(ib.token_ids_cpu[idx, :seq_len]),
                                seq_len, float(post[wslot]), req, self.step)
        changed = np.nonzero(
            np.abs(post - self._pre) > 1e-3 * np.maximum(1.0, np.abs(self._pre)))[0]
        for s in changed:
            s = int(s)
            if s not in written and s != 0:
                owner = self.prov.get(s)
                self._violation(
                    "UNEXPECTED-WRITE", "-", -1, s,
                    f"slot changed without a planned write; previously held "
                    f"{owner[:2] if owner else None}")
                self.prov.pop(s, None)
        self._plan_cache = None

    def _violation(self, kind, req, pos, slot, detail):
        self.violations += 1
        logger.warning("[prefix-trace] VIOLATION #%d %s step %d req %s pos %d slot %d: %s",
                       self.violations, kind, self.step, str(req)[-8:], pos, slot,
                       detail)
