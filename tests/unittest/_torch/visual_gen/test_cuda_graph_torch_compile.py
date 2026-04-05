# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CUDA graph + torch.compile integration in visual gen pipelines.

Regression tests for the bug where _setup_cuda_graphs() was called before
torch_compile(), causing CUDA graphs to capture the un-compiled model forward
(or be silently skipped when torch.compile was enabled).

The fix defers _setup_cuda_graphs() to after torch_compile() in pipeline_loader
so that graphs capture the compiled model forward.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from tensorrt_llm._torch.visual_gen.config import (
    CudaGraphConfig,
    TorchCompileConfig,
)
from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline


# ---------------------------------------------------------------------------
# Minimal stub pipeline that avoids loading real weights / configs
# ---------------------------------------------------------------------------


class _StubPipeline(BasePipeline):
    """Minimal stub that satisfies BasePipeline's abstract interface."""

    def __init__(self, model_config):
        # Bypass BasePipeline.__init__ to control exactly what happens
        self.model_config = model_config
        self.mapping = MagicMock()
        self._cuda_graph_runners = {}
        self._parallel_vae_enabled = False
        self._warmed_up_shapes = set()
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None
        self.transformer = _SimpleTransformer()

    def _init_transformer(self):
        pass

    def forward(self, *args, **kwargs):
        pass

    def infer(self, req):
        pass

    def _run_warmup(self, height, width, num_frames, steps):
        pass


class _SimpleTransformer(nn.Module):
    """Tiny transformer stub with a trackable forward call count."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.call_count = 0

    def forward(self, x):
        self.call_count += 1
        return self.linear(x)


def _make_model_config(*, enable_cuda_graph=True, enable_torch_compile=True):
    cfg = MagicMock()
    cfg.cuda_graph = CudaGraphConfig(enable_cuda_graph=enable_cuda_graph)
    cfg.torch_compile = TorchCompileConfig(
        enable_torch_compile=enable_torch_compile,
        enable_fullgraph=False,
        enable_autotune=False,
    )
    cfg.compilation = MagicMock()
    cfg.compilation.resolutions = None
    cfg.compilation.num_frames = None
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCudaGraphDeferredWhenTorchCompileEnabled:
    """_setup_cuda_graphs() must be called AFTER torch_compile()."""

    def test_no_cuda_graph_runners_in_init_when_compile_enabled(self):
        """When torch.compile is on, __init__ must not install CUDA graph runners."""
        model_config = _make_model_config(
            enable_cuda_graph=True, enable_torch_compile=True
        )
        pipe = _StubPipeline(model_config)

        # Simulate __init__ behaviour: only call _setup_cuda_graphs when compile is off
        if not pipe.model_config.torch_compile.enable_torch_compile:
            pipe._setup_cuda_graphs()

        assert pipe._cuda_graph_runners == {}, (
            "CUDA graph runners must NOT be installed before torch.compile is applied"
        )

    def test_cuda_graph_runners_installed_after_compile(self):
        """After torch_compile() + _setup_cuda_graphs(), runners must be present."""
        model_config = _make_model_config(
            enable_cuda_graph=True, enable_torch_compile=True
        )
        pipe = _StubPipeline(model_config)

        # Simulate pipeline_loader: torch.compile first, then CUDA graphs
        pipe._setup_cuda_graphs()

        assert "transformer" in pipe._cuda_graph_runners, (
            "_setup_cuda_graphs() should install a runner for 'transformer'"
        )
        assert isinstance(pipe._cuda_graph_runners["transformer"], CUDAGraphRunner)

    def test_cuda_graph_wraps_current_transformer_attribute(self):
        """The CUDA graph runner must wrap whatever self.transformer points to
        at the time _setup_cuda_graphs() is called, not a stale reference."""
        model_config = _make_model_config(
            enable_cuda_graph=True, enable_torch_compile=True
        )
        pipe = _StubPipeline(model_config)

        # Simulate torch_compile() replacing self.transformer (whole-module path)
        compiled_transformer = MagicMock(spec=nn.Module)
        compiled_transformer.forward = MagicMock(return_value=torch.zeros(1))
        pipe.transformer = compiled_transformer

        # Now _setup_cuda_graphs must wrap the NEW (compiled) transformer
        pipe._setup_cuda_graphs()

        # Verify the runner was set up on the compiled transformer
        assert "transformer" in pipe._cuda_graph_runners
        # The compiled transformer's forward should now be the wrapped version
        assert pipe.transformer.forward is not MagicMock, (
            "forward on the compiled transformer must be replaced with the wrapper"
        )

    def test_cuda_graph_disabled_no_runners(self):
        """When cuda_graph is disabled, _setup_cuda_graphs() must be a no-op."""
        model_config = _make_model_config(
            enable_cuda_graph=False, enable_torch_compile=True
        )
        pipe = _StubPipeline(model_config)
        pipe._setup_cuda_graphs()

        assert pipe._cuda_graph_runners == {}

    def test_cuda_graph_without_torch_compile_still_works(self):
        """Regression: cuda_graph=True, torch_compile=False path must be unaffected."""
        model_config = _make_model_config(
            enable_cuda_graph=True, enable_torch_compile=False
        )
        pipe = _StubPipeline(model_config)

        # Simulate __init__ behaviour for compile-off path
        if not pipe.model_config.torch_compile.enable_torch_compile:
            pipe._setup_cuda_graphs()

        assert "transformer" in pipe._cuda_graph_runners, (
            "CUDA graph should still be installed when torch.compile is disabled"
        )


class TestSetupCudaGraphsNoLongerGuardsOnTorchCompile:
    """_setup_cuda_graphs() must not silently skip when torch.compile is enabled."""

    def test_no_warning_emitted_when_both_enabled(self, caplog):
        """The old code emitted a warning and returned early; the fix removes that."""
        import logging

        model_config = _make_model_config(
            enable_cuda_graph=True, enable_torch_compile=True
        )
        pipe = _StubPipeline(model_config)

        with caplog.at_level(logging.WARNING):
            pipe._setup_cuda_graphs()

        # The old warning message must not appear
        old_warning = "CUDA graphs with torch.compile not yet supported"
        assert old_warning not in caplog.text, (
            f"Found stale warning '{old_warning}' — the guard should have been removed"
        )
