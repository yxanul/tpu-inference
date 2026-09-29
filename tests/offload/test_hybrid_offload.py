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

import types

import jax.numpy as jnp
import numpy as np
from vllm.v1.kv_offload.base import (GPULoadStoreSpec, LookupResult,
                                     ReqContext, make_offload_key)

from tpu_inference.offload.hybrid_offload import (GroupInfo,
                                                  TPUHostLoadStoreSpec,
                                                  TPUHybridOffloadingManager,
                                                  TPUHybridOffloadingWorker,
                                                  plan_host_pools)


def test_plan_host_pools_splits_budget_by_kind():
    infos = {
        0: GroupInfo(is_mamba=False, bytes_per_block=16),
        1: GroupInfo(is_mamba=True, bytes_per_block=100),
        2: GroupInfo(is_mamba=True, bytes_per_block=100),
    }
    counts = plan_host_pools([0, 1, 2],
                             infos,
                             host_bytes=1000,
                             mamba_fraction=0.3)
    # Mirrored mamba groups share one pool.
    assert counts == {(False, 16): 700 // 16, (True, 100): 3}


def test_manager_routes_keys_by_group_and_keeps_key_order():
    manager = TPUHybridOffloadingManager({
        0: 0,
        1: 1,
        2: 1
    }, [4, 2],
                                         cache_policy="lru",
                                         enable_events=False)
    ctx = ReqContext(req_id="r")
    keys = [
        make_offload_key(b"a0", 0),
        make_offload_key(b"m0", 1),
        make_offload_key(b"a1", 0),
        make_offload_key(b"m0", 2),
    ]
    out = manager.prepare_store(keys, ctx)
    assert out.keys_to_store == keys
    entries = out.store_spec.entries
    assert [pool for pool, _ in entries] == [0, 1, 0, 1]
    assert len({e for e in entries}) == 4
    for key in keys:
        assert manager.lookup(key, ctx) is LookupResult.HIT_PENDING
    manager.complete_store(keys, ctx)
    for key in keys:
        assert manager.lookup(key, ctx) is LookupResult.HIT

    # Loads report the slots the keys were stored in, in request order.
    load = manager.prepare_load(list(reversed(keys)), ctx)
    assert load.entries == list(reversed(entries))
    manager.complete_load(keys, ctx)

    # The 2-slot mamba pool evicts its LRU entries; attention is untouched.
    more = [make_offload_key(b"m1", 1), make_offload_key(b"m1", 2)]
    out = manager.prepare_store(more, ctx)
    assert set(out.evicted_keys) == {keys[1], keys[3]}
    manager.complete_store(more, ctx)
    assert manager.lookup(keys[0], ctx) is LookupResult.HIT
    assert manager.lookup(keys[1], ctx) is LookupResult.MISS


def test_worker_store_then_load_round_trip():
    rng = np.random.default_rng(0)

    def rand(*shape, dtype=jnp.float32):
        return jnp.asarray(rng.standard_normal(shape), dtype=dtype)

    runner = types.SimpleNamespace(
        kv_caches=[
            rand(8, 16, 128, dtype=jnp.bfloat16),
            (rand(5, 3, 64), rand(5, 4, 8, 8)),
            rand(8, 16, 128, dtype=jnp.bfloat16),
        ],
        layer_name_to_kvcache_index={
            "a0": 0,
            "m0": 1,
            "a1": 2
        },
    )
    original = [
        np.asarray(runner.kv_caches[0]),
        tuple(np.asarray(x) for x in runner.kv_caches[1]),
        np.asarray(runner.kv_caches[2]),
    ]
    worker = TPUHybridOffloadingWorker(runner, [["a0", "a1"], ["m0"]],
                                       group_pools=[0, 1],
                                       pool_sizes=[6, 3])
    entries = [(0, 4), (0, 1), (1, 2)]
    worker.submit_store(
        1, GPULoadStoreSpec([2, 5, 3],
                            group_sizes=[2, 1],
                            block_indices=[0, 0]),
        TPUHostLoadStoreSpec(entries))
    worker.wait({1})
    [result] = worker.get_finished()
    assert result.job_id == 1 and result.success

    # Later forward passes overwrite the source blocks.
    runner.kv_caches[0] = runner.kv_caches[0].at[2].set(0)
    runner.kv_caches[1] = (runner.kv_caches[1][0].at[3].set(0),
                           runner.kv_caches[1][1].at[3].set(0))

    worker.submit_load(
        2, TPUHostLoadStoreSpec(entries),
        GPULoadStoreSpec([6, 7, 4], group_sizes=[2, 1], block_indices=[0, 0]))
    worker.wait({2})
    [result] = worker.get_finished()
    assert result.job_id == 2 and result.success

    kv = runner.kv_caches
    np.testing.assert_array_equal(np.asarray(kv[0][6]), original[0][2])
    np.testing.assert_array_equal(np.asarray(kv[0][7]), original[0][5])
    np.testing.assert_array_equal(np.asarray(kv[2][6]), original[2][2])
    np.testing.assert_array_equal(np.asarray(kv[1][0][4]), original[1][0][3])
    np.testing.assert_array_equal(np.asarray(kv[1][1][4]), original[1][1][3])
    # Untouched blocks keep their contents.
    np.testing.assert_array_equal(np.asarray(kv[0][5]), original[0][5])
    worker.shutdown()
