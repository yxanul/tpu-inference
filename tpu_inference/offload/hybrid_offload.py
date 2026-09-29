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
"""HBM + host-DRAM prefix cache for hybrid (attention + GDN/mamba) models.

Built on vLLM's ``OffloadingConnector``, whose scheduler side is
device-agnostic and already handles hybrid KV-cache groups and mamba
"align" mode (it stores the exact boundary states the MambaManager hands
over). This module adds the TPU pieces:

- ``TPUHybridOffloadingSpec``: one host pool per slot size. The TPU keeps
  attention blocks and mamba states in separate device pools whose slots
  differ ~10x in size, so they cannot share vLLM's uniform CPU chunks.
- ``TPUHybridOffloadingManager``: routes offload keys by KV-cache group to one
  vLLM ``CPUOffloadingManager`` (LRU) per host pool.
- ``TPUHybridOffloadingWorker``: stores with a jitted gather of the device
  blocks and a background copy into preallocated host (numpy) pools; loads
  with a host->device copy and a jitted scatter into ``runner.kv_caches``,
  applied on the runner thread before the request runs.
- ``TPUHybridOffloadingConnector``: ``OffloadingConnector`` registering the
  TPU runner's JAX caches (``register_runner``) instead of torch tensors.

Enable with::

  --kv-transfer-config '{"kv_connector": "TPUHybridOffloadingConnector",
    "kv_connector_module_path": "tpu_inference.offload.hybrid_offload",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "spec_name": "TPUHybridOffloadingSpec",
      "spec_module_path": "tpu_inference.offload.hybrid_offload",
      "host_gb": 200, "mamba_host_fraction": 0.3,
      "offload_prompt_only": false}}'
"""

import concurrent.futures
import dataclasses
import functools
import threading
import time
from collections.abc import Collection, Iterable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import \
    OffloadingConnectorWorker
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import \
    OffloadingConnector
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.kv_offload.base import (GPULoadStoreSpec, LoadStoreSpec,
                                     LookupResult, OffloadingEvent,
                                     OffloadingManager, OffloadingSpec,
                                     OffloadingWorker, OffloadKey,
                                     PrepareStoreOutput, ReqContext,
                                     RequestOffloadingContext,
                                     ScheduleEndContext, TransferResult,
                                     get_offload_group_idx)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# Per-group host-pool info, set by TPUHybridOffloadingConnector before vLLM's
# connector constructor builds the spec's manager (both live in one process).
_GROUP_INFO: dict[int, "GroupInfo"] = {}


@dataclasses.dataclass(frozen=True)
class GroupInfo:
    """Host-pool routing for one KV-cache group."""
    is_mamba: bool
    # Bytes of one device block of this group, all of its layers.
    bytes_per_block: int

    @property
    def pool_key(self) -> tuple[bool, int]:
        # Groups with identical slots (e.g. mirrored mamba groups) share a
        # pool.
        return (self.is_mamba, self.bytes_per_block)


def group_info_from_kv_cache_config(kv_cache_config) -> dict[int, GroupInfo]:
    info = {}
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        is_mamba = isinstance(spec, MambaSpec)
        if is_mamba:
            page = dataclasses.replace(spec,
                                       page_size_padded=None).page_size_bytes
        else:
            page = spec.page_size_bytes
        info[group_id] = GroupInfo(is_mamba=is_mamba,
                                   bytes_per_block=page *
                                   len(group.layer_names))
    return info


def plan_host_pools(group_ids: Iterable[int], infos: dict[int, GroupInfo],
                    host_bytes: int,
                    mamba_fraction: float) -> dict[tuple[bool, int], int]:
    """Number of slots per host pool for a byte budget: `mamba_fraction` of
    it for mamba-state pools, the rest for attention pools, each split in
    proportion to slot size."""
    pools = {infos[g].pool_key for g in group_ids}
    counts = {}
    for is_mamba in (False, True):
        kind = [p for p in pools if p[0] == is_mamba]
        if not kind:
            continue
        share = mamba_fraction if is_mamba else 1.0 - mamba_fraction
        if not any(p[0] != is_mamba for p in pools):
            share = 1.0
        kind_bytes = int(host_bytes * share)
        total_slot = sum(p[1] for p in kind)
        for p in kind:
            counts[p] = max(1, kind_bytes // total_slot)
    return counts


@dataclasses.dataclass
class TPUHostLoadStoreSpec(LoadStoreSpec):
    """Host slots, in the order of the offload keys: (pool index, chunk)."""
    entries: list[tuple[int, int]]


class TPUHybridOffloadingManager(OffloadingManager):
    """One LRU (vLLM CPUOffloadingManager) per host pool; keys are routed by
    their KV-cache group."""

    def __init__(self, group_to_pool: dict[int, int], pool_sizes: list[int],
                 cache_policy: str, enable_events: bool):
        self._group_to_pool = group_to_pool
        self._pools = [
            CPUOffloadingManager(num_chunks=n,
                                 cache_policy=cache_policy,
                                 enable_events=enable_events)
            for n in pool_sizes
        ]

    def _pool(self, key: OffloadKey) -> int:
        return self._group_to_pool[get_offload_group_idx(key)]

    def _split(self, keys: Collection[OffloadKey]) -> dict[int, list]:
        by_pool: dict[int, list[OffloadKey]] = {}
        for key in keys:
            by_pool.setdefault(self._pool(key), []).append(key)
        return by_pool

    def _entries(self, keys: list[OffloadKey], specs: dict[int,
                                                           CPULoadStoreSpec],
                 subsets: dict[int, list[OffloadKey]]) -> list:
        position = {
            pool: {
                key: i
                for i, key in enumerate(subset)
            }
            for pool, subset in subsets.items()
        }
        entries = []
        for key in keys:
            pool = self._pool(key)
            entries.append(
                (pool, int(specs[pool].block_ids[position[pool][key]])))
        return entries

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        return self._pools[self._pool(key)].lookup(key, req_context)

    def prepare_load(self, keys: Collection[OffloadKey],
                     req_context: ReqContext) -> LoadStoreSpec:
        keys = list(keys)
        subsets = self._split(keys)
        specs = {
            pool: self._pools[pool].prepare_load(subset, req_context)
            for pool, subset in subsets.items()
        }
        return TPUHostLoadStoreSpec(self._entries(keys, specs, subsets))

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        for pool, subset in self._split(keys).items():
            self._pools[pool].touch(subset, req_context)

    def complete_load(self, keys: Collection[OffloadKey],
                      req_context: ReqContext):
        for pool, subset in self._split(keys).items():
            self._pools[pool].complete_load(subset, req_context)

    def prepare_store(self, keys: Collection[OffloadKey],
                      req_context: ReqContext) -> PrepareStoreOutput | None:
        keys = list(keys)
        subsets = self._split(keys)
        outputs = {}
        for pool, subset in subsets.items():
            output = self._pools[pool].prepare_store(subset, req_context)
            if output is None:
                # Undo the pools that already reserved slots.
                for done_pool, done in outputs.items():
                    if done.keys_to_store:
                        self._pools[done_pool].complete_store(
                            done.keys_to_store, req_context, success=False)
                return None
            outputs[pool] = output
        to_store = {k for out in outputs.values() for k in out.keys_to_store}
        keys_to_store = [k for k in keys if k in to_store]
        store_subsets = {
            pool: list(out.keys_to_store)
            for pool, out in outputs.items()
        }
        specs = {pool: out.store_spec for pool, out in outputs.items()}
        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=TPUHostLoadStoreSpec(
                self._entries(keys_to_store, specs, store_subsets)),
            evicted_keys=[
                k for out in outputs.values() for k in out.evicted_keys
            ],
        )

    def complete_store(self,
                       keys: Collection[OffloadKey],
                       req_context: ReqContext,
                       success: bool = True):
        for pool, subset in self._split(keys).items():
            self._pools[pool].complete_store(subset, req_context, success)

    def on_new_request(self,
                       req_context: ReqContext) -> RequestOffloadingContext:
        contexts = [p.on_new_request(req_context) for p in self._pools]
        return contexts[0]

    def on_request_finished(self, req_context: ReqContext) -> None:
        for pool in self._pools:
            pool.on_request_finished(req_context)

    def take_events(self) -> Iterable[OffloadingEvent]:
        for pool in self._pools:
            yield from pool.take_events()

    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        for pool in self._pools:
            pool.on_schedule_end(context)

    def has_pending_work(self) -> bool:
        return any(p.has_pending_work() for p in self._pools)

    def reset_cache(self) -> None:
        for pool in self._pools:
            pool.reset_cache()

    def get_stats(self):
        return None

    def shutdown(self) -> None:
        for pool in self._pools:
            pool.shutdown()


class TPUHybridOffloadingSpec(OffloadingSpec):
    """Host pools sized from `host_gb` (and `mamba_host_fraction`)."""

    def __init__(self, config):
        super().__init__(config)
        assert self.blocks_per_chunk == 1, (
            "TPUHybridOffloadingSpec stores whole device blocks "
            "(blocks_per_chunk must be 1)")
        host_bytes = int(float(self.extra_config.get("host_gb", 64)) * 2**30)
        mamba_fraction = float(
            self.extra_config.get("mamba_host_fraction", 0.3))
        self.group_ids = [g.group_id for g in config.groups]
        infos = _GROUP_INFO
        assert all(g in infos for g in self.group_ids), (
            "TPUHybridOffloadingSpec needs TPUHybridOffloadingConnector")
        counts = plan_host_pools(self.group_ids, infos, host_bytes,
                                 mamba_fraction)
        self.pool_keys = sorted(counts)
        self.pool_sizes = [counts[p] for p in self.pool_keys]
        # Offload keys carry the KV-cache group id.
        self.group_to_pool = {
            g: self.pool_keys.index(infos[g].pool_key)
            for g in self.group_ids
        }
        for key, size in zip(self.pool_keys, self.pool_sizes):
            logger.info(
                "[hybrid-offload] host pool %s: %d slots x %.1f MiB = %.1f GiB",
                "mamba" if key[0] else "attention", size, key[1] / 2**20,
                size * key[1] / 2**30)
        self._manager: TPUHybridOffloadingManager | None = None
        self._worker: TPUHybridOffloadingWorker | None = None

    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            self._manager = TPUHybridOffloadingManager(
                self.group_to_pool,
                self.pool_sizes,
                cache_policy=self.extra_config.get("eviction_policy", "lru"),
                enable_events=self.kv_events_config.enable_kv_cache_events)
        return self._manager

    def get_worker(self, kv_caches: Any) -> OffloadingWorker:
        # `kv_caches` is the TPU runner (see TPUOffloadingConnectorWorker).
        if self._worker is None:
            group_layers = [list(g.layer_names) for g in self.config.groups]
            group_pools = [self.group_to_pool[g] for g in self.group_ids]
            self._worker = TPUHybridOffloadingWorker(kv_caches, group_layers,
                                                     group_pools,
                                                     self.pool_sizes)
        return self._worker


def _bucket(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


@functools.partial(jax.jit, static_argnames=())
def _gather_blocks(arrays: tuple[jax.Array, ...],
                   block_ids: jax.Array) -> tuple[jax.Array, ...]:
    return tuple(a[block_ids] for a in arrays)


@functools.partial(jax.jit, donate_argnums=(0, ))
def _scatter_blocks(arrays: tuple[jax.Array, ...], block_ids: jax.Array,
                    values: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    return tuple(a.at[block_ids].set(v) for a, v in zip(arrays, values))


class TPUHybridOffloadingWorker(OffloadingWorker):
    """Moves whole device blocks of one KV-cache group at a time between
    `runner.kv_caches` and per-pool host arrays."""

    def __init__(self, runner, group_layers: list[list[str]],
                 group_pools: list[int], pool_sizes: list[int]):
        self._runner = runner
        # Per offloaded group: kv_caches positions of its components, as
        # (cache index, element index or None for a plain array).
        self._group_components: list[list[tuple[int, int | None]]] = []
        for layers in group_layers:
            seen, components = set(), []
            for name in layers:
                idx = runner.layer_name_to_kvcache_index[name]
                if idx in seen:
                    continue
                seen.add(idx)
                cache = runner.kv_caches[idx]
                if isinstance(cache, (tuple, list)):
                    components += [(idx, j) for j in range(len(cache))]
                else:
                    components.append((idx, None))
            self._group_components.append(components)
        self._group_pools = group_pools
        # Host storage: per pool, per component position, one lazily
        # committed array [num_slots, *block_shape].
        self._pools: list[list[np.ndarray] | None] = [None] * len(pool_sizes)
        self._pool_sizes = pool_sizes
        for gi, pool in enumerate(group_pools):
            if self._pools[pool] is None:
                self._pools[pool] = [
                    np.empty((pool_sizes[pool], ) + tuple(a.shape[1:]),
                             dtype=a.dtype) for a in self._arrays(gi)
                ]
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="tpu-hybrid-offload")
        self._lock = threading.Lock()
        # job_id -> (future, start time, bytes, pending scatter or None)
        self._jobs: dict[int, tuple[concurrent.futures.Future, float, int,
                                    Any]] = {}

    def _arrays(self, gi: int) -> list[jax.Array]:
        caches = self._runner.kv_caches
        return [
            caches[i] if j is None else caches[i][j]
            for i, j in self._group_components[gi]
        ]

    def _set_arrays(self, gi: int, arrays: tuple[jax.Array, ...]) -> None:
        caches = self._runner.kv_caches
        for (i, j), array in zip(self._group_components[gi], arrays):
            if j is None:
                caches[i] = array
            else:
                parts = list(caches[i])
                parts[j] = array
                caches[i] = type(caches[i])(parts)

    @staticmethod
    def _groups(spec: GPULoadStoreSpec):
        offset = 0
        for gi, n in enumerate(spec.group_sizes):
            if n:
                yield gi, offset, n
            offset += n

    def submit_store(self, job_id: int, src_spec: GPULoadStoreSpec,
                     dst_spec: LoadStoreSpec) -> bool:
        assert isinstance(dst_spec, TPUHostLoadStoreSpec)
        tasks = []
        for gi, offset, n in self._groups(src_spec):
            ids = np.zeros(_bucket(n), dtype=np.int32)
            ids[:n] = src_spec.block_ids[offset:offset + n]
            # Dispatched now, before later steps overwrite these blocks.
            gathered = _gather_blocks(tuple(self._arrays(gi)),
                                      jnp.asarray(ids))
            slots = [c for _, c in dst_spec.entries[offset:offset + n]]
            pool = dst_spec.entries[offset][0]
            tasks.append((pool, slots, n, gathered))

        def copy_out():
            nbytes = 0
            for pool, slots, n, gathered in tasks:
                for host, block in zip(self._pools[pool], gathered):
                    values = np.asarray(block)[:n]
                    host[slots] = values
                    nbytes += values.nbytes
            return nbytes

        with self._lock:
            self._jobs[job_id] = (self._executor.submit(copy_out),
                                  time.perf_counter(), 0, None)
        return True

    def submit_load(self, job_id: int, src_spec: LoadStoreSpec,
                    dst_spec: GPULoadStoreSpec) -> bool:
        assert isinstance(src_spec, TPUHostLoadStoreSpec)
        plan = []
        for gi, offset, n in self._groups(dst_spec):
            ids = np.zeros(_bucket(n), dtype=np.int32)
            ids[:n] = dst_spec.block_ids[offset:offset + n]
            pool = src_spec.entries[offset][0]
            slots = [c for _, c in src_spec.entries[offset:offset + n]]
            plan.append((gi, ids, pool, slots, n))
        shardings = {
            gi: [a.sharding for a in self._arrays(gi)]
            for gi, *_ in plan
        }

        def copy_in():
            loaded = []
            for gi, ids, pool, slots, n in plan:
                values = []
                for host, sharding in zip(self._pools[pool], shardings[gi]):
                    rows = np.zeros((len(ids), ) + host.shape[1:],
                                    dtype=host.dtype)
                    rows[:n] = host[slots]
                    values.append(jax.device_put(rows, sharding))
                loaded.append((gi, ids, tuple(values)))
            return loaded

        with self._lock:
            self._jobs[job_id] = (self._executor.submit(copy_in),
                                  time.perf_counter(), 0, "load")
        return True

    def _finish(self, job_id: int) -> TransferResult:
        future, start, _, kind = self._jobs.pop(job_id)
        try:
            result = future.result()
            if kind == "load":
                # On the runner thread, ahead of the next forward pass.
                nbytes = 0
                for gi, ids, values in result:
                    self._set_arrays(
                        gi,
                        _scatter_blocks(tuple(self._arrays(gi)),
                                        jnp.asarray(ids), values))
                    nbytes += sum(v.nbytes for v in values)
            else:
                nbytes = result
            return TransferResult(job_id, True, nbytes,
                                  time.perf_counter() - start)
        except Exception:
            logger.exception("[hybrid-offload] transfer job %d failed", job_id)
            return TransferResult(job_id, False)

    def get_finished(self) -> list[TransferResult]:
        with self._lock:
            done = [j for j, job in self._jobs.items() if job[0].done()]
        return [self._finish(j) for j in done]

    def wait(self, job_ids: set[int]) -> None:
        for job_id in job_ids:
            with self._lock:
                job = self._jobs.get(job_id)
            if job is not None:
                job[0].result()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


class TPUOffloadingConnectorWorker(OffloadingConnectorWorker):
    """vLLM's offloading worker logic with TPU cache registration."""

    def register_runner(self, runner) -> None:
        self._init_worker(runner)


class TPUHybridOffloadingConnector(OffloadingConnector):
    """vLLM's OffloadingConnector for TPU hybrid models."""

    def __init__(self, vllm_config, role, kv_cache_config=None):
        if kv_cache_config is not None:
            _GROUP_INFO.clear()
            _GROUP_INFO.update(
                group_info_from_kv_cache_config(kv_cache_config))
        super().__init__(vllm_config, role, kv_cache_config)
        worker = self.connector_worker
        if worker is not None:
            self.connector_worker = TPUOffloadingConnectorWorker(
                worker.spec, worker.vllm_config, worker.kv_cache_config)

    def register_runner(self, runner) -> None:
        assert self.connector_worker is not None
        self.connector_worker.register_runner(runner)
