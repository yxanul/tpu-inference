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

from vllm.v1.executor.uniproc_executor import \
    UniProcExecutor as UniProcExecutorV1

from tpu_inference.core.hybrid_coordinator import (
    MambaPoolSyncExecutorMixin, maybe_install_hybrid_coordinator_hooks)


class UniProcExecutor(MambaPoolSyncExecutorMixin, UniProcExecutorV1):
    """Single-host executor that installs the hybrid KV-cache hooks.

    The engine core builds its scheduler (and with it the KV-cache
    coordinator) right after the executor. Installing the hooks here, like the
    multiproc and Ray executors do, makes mamba prefix caching use
    TPUHybridKVCacheCoordinator and its compact mamba pool instead of vLLM's
    default coordinator, which would hand out mamba block ids from the much
    larger attention pool.
    """

    def _init_executor(self) -> None:
        maybe_install_hybrid_coordinator_hooks(self.vllm_config)
        super()._init_executor()
