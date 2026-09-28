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

from unittest import mock

from tpu_inference.core.hybrid_coordinator import MambaPoolSyncExecutorMixin
from tpu_inference.executors.uniproc_executor import UniProcExecutor


def test_installs_hybrid_hooks_before_worker_init():
    """The engine core builds the scheduler right after the executor, so the
    hybrid KV-cache hooks must be in place before the base executor runs."""
    calls = []
    with mock.patch(
            "tpu_inference.executors.uniproc_executor."
            "maybe_install_hybrid_coordinator_hooks",
            side_effect=lambda cfg: calls.append(("hooks", cfg))), \
        mock.patch(
            "vllm.v1.executor.uniproc_executor.UniProcExecutor._init_executor",
            autospec=True,
            side_effect=lambda self: calls.append(("init", None))):
        executor = UniProcExecutor.__new__(UniProcExecutor)
        executor.vllm_config = mock.sentinel.vllm_config
        executor._init_executor()

    assert calls == [("hooks", mock.sentinel.vllm_config), ("init", None)]


def test_publishes_mamba_pool_size():
    assert issubclass(UniProcExecutor, MambaPoolSyncExecutorMixin)
