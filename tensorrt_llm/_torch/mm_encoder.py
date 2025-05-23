from pathlib import Path
from typing import Any, Literal, Optional, Union, List

from transformers import PreTrainedTokenizerBase
from tensorrt_llm.executor import GenerationExecutor
from tensorrt_llm.llmapi.llm import LlmArgs
from tensorrt_llm.llmapi.utils import exception_handler, get_device_count, print_colored_debug
from tensorrt_llm.llmapi.llm_utils import LlmBuildStats, CachedModelLoader, _ModelRuntimeContext
from tensorrt_llm.logger import logger
from tensorrt_llm.executor.utils import get_spawn_proxy_process_env, create_mpi_comm_session
from tensorrt_llm.llmapi.mpi_session import external_mpi_comm_available, MpiPoolSession
import tempfile
import os
import atexit
import weakref
from tensorrt_llm._utils import nvtx_range_debug
from tensorrt_llm.inputs import create_input_processor
from tensorrt_llm.executor.request import MultimodalRequest

from tensorrt_llm.bindings import executor as tllm

class MultimodalEncoder:

    def __init__(self,
                 model: Union[str, Path],
                 trust_remote_code: bool = False,
                 tensor_parallel_size: int = 1,
                 dtype: str = "auto",
                 revision: Optional[str] = None,
                 **kwargs: Any) -> None:

        self._executor_cls = kwargs.pop("executor_cls", GenerationExecutor)
        try:
            self.args = LlmArgs.from_kwargs(
                backend="pytorch",
                model=model,
                trust_remote_code=trust_remote_code,
                tensor_parallel_size=tensor_parallel_size,
                dtype=dtype,
                revision=revision,
                **kwargs)

        except Exception as e:
            logger.error(
                f"Failed to parse the arguments for the mm encoder constructor: {e}")
            raise e


        print_colored_debug(f"Encoder.args.mpi_session: {self.args.mpi_session}\n",
                            "yellow")
        self.mpi_session = self.args.mpi_session

        if self.args.parallel_config.is_multi_gpu:
            if get_device_count(
            ) < self.args.parallel_config.world_size_per_node:
                raise RuntimeError(
                    f"Only {get_device_count()} GPUs are available, but {self.args.parallel_config.world_size} are required."
                )

            logger.info(
                f'start MpiSession with {self.args.parallel_config.world_size} workers'
            )
            if not self.mpi_session:
                mpi_process_pre_spawned: bool = get_spawn_proxy_process_env()
                if not mpi_process_pre_spawned:
                    print_colored_debug(f"Encoder create MpiPoolSession\n",
                                        "yellow")
                    self.mpi_session = MpiPoolSession(
                        n_workers=self.args.parallel_config.world_size)
                else:
                    print_colored_debug(f"Encoder create MpiCommSession\n",
                                        "yellow")
                    self.mpi_session = create_mpi_comm_session(
                        self.args.parallel_config.world_size)

        try:
            # Due to the Executor can only accept a engine path, we need to save the engine to a directory
            self._engine_dir: Optional[Path] = None
            self._executor: Optional[GenerationExecutor] = None
            self._workspace = tempfile.TemporaryDirectory(
                suffix="-mm-encoder-workspace", dir=self.args.workspace)

            self._hf_model_dir: Optional[Path] = None

            self.runtime_context: Optional[_ModelRuntimeContext] = None
            self.llm_build_stats = LlmBuildStats()

            self._build_model()

        except Exception as e:
            if self.mpi_session is not None:
                self.mpi_session.shutdown()
            raise e

        exception_handler.register(self, 'shutdown')
        atexit.register(MultimodalEncoder._shutdown_wrapper, weakref.ref(self))

    @property
    def workspace(self) -> Path:
        return Path(self._workspace.name)

    def generate_from_mm_request(
        self,
        mm_requests: List[MultimodalRequest],
    ):

        futures = []
        for i, request_inputs in enumerate(mm_requests):
            # future is a
            future = self.generate_async(
                request_inputs,
            )
            futures.append(future)
        for future in futures:
            future.result()

        return futures

    @nvtx_range_debug("Encoder.generate_async", color="green", category="Encoder")
    def generate_async(
        self,
        mm_request: MultimodalRequest,
    ):

        result = self._executor.generate_multimodal_async(
            mm_request,
        )
        return result
        #return RequestOutput._from_generation_result(result, prompt,
        #                                             self.tokenizer)


    def _build_model(self):
        model_loader = CachedModelLoader(self.args,
                                         mpi_session=self.mpi_session,
                                         workspace=self.workspace,
                                         llm_build_stats=weakref.proxy(
                                             self.llm_build_stats))
        self._engine_dir, self._hf_model_dir = model_loader()
        # update the model_dir to a local dir for the runtime, such as tokenizer loading.
        if self._engine_dir is not None:
            self.args.model = self._engine_dir

        # Multimodal special handling:
        # 1. Default load_tokenizer may fail because MM has different tokenizer configuration. Hence we initialize it inside input processor
        # 2. May need to modify model weights for MM (e.g., resize vocab embedding). We must do such operation via input processor's __init__
        #self.input_processor = create_input_processor(self._hf_model_dir, None)

        max_batch_size = self.args.max_batch_size or self.args.build_config.max_batch_size
        # In _build_model method:
        executor_config = tllm.ExecutorConfig(1)
        executor_config.backend = "multimodal"
        executor_config.mapping = self.args.parallel_config.to_mapping()
        executor_config.build_config = self.args.build_config
        executor_config.hf_model_dir = self._hf_model_dir
        executor_config.trt_engine_dir = self._engine_dir
        executor_config.max_batch_size = max_batch_size
        executor_config.max_num_active_requests = 100


        self._executor = self._executor_cls.create(
            self._engine_dir,
            executor_config=executor_config,
            model_world_size=self.args.parallel_config.world_size,
            mpi_session=self.mpi_session,
            reuse_mpi_comm=external_mpi_comm_available(
                self.args.parallel_config.world_size),
            is_llm_executor=False)


    def shutdown(self) -> None:
        if hasattr(self, "_executor") and self._executor is not None:
            self._executor.shutdown()
            self._executor = None

        if hasattr(self, 'mpi_session') and self.mpi_session is not None:
            self.mpi_session.shutdown()
            self.mpi_session = None

    @staticmethod
    def _shutdown_wrapper(self_ref):
        # Retrieve the instance if it still exists
        instance = self_ref()
        if instance is not None:
            instance.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        del exc_value, traceback
        self.shutdown()
        return False  # propagate exceptions

    def __getstate__(self):
        raise RuntimeError("Encoder object can not be pickled.")

    def __del__(self):
        self.shutdown()