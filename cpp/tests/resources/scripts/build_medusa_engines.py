#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse as _arg
import pathlib as _pl
import platform as _pf
import sys as _sys

from build_engines_utils import init_model_spec_module, run_command, wincopy

init_model_spec_module()
import model_spec

import tensorrt_llm.bindings as _tb


def build_engine(base_model_dir: _pl.Path, medusa_model_dir: _pl.Path,
                 engine_dir: _pl.Path, *args):

    covert_cmd = [_sys.executable, "examples/medusa/convert_checkpoint.py"] + (
        ['--model_dir', str(base_model_dir)] if base_model_dir else []) + [
            '--medusa_model_dir', str(medusa_model_dir), \
            '--output_dir', str(engine_dir), '--dtype=float16', '--num_medusa_heads=4'
        ] + list(args)

    run_command(covert_cmd)

    build_args = ["trtllm-build"] + (
        ['--checkpoint_dir', str(engine_dir)] if engine_dir else []) + [
            '--output_dir',
            str(engine_dir),
            '--gemm_plugin=float16',
            '--max_batch_size=8',
            '--max_input_len=12',
            '--max_seq_len=140',
            '--log_level=error',
            '--paged_kv_cache=enable',
            '--use_paged_context_fmha=enable',
            '--remove_input_padding=enable',
            '--speculative_decoding_mode=medusa',
        ]

    run_command(build_args)


def build_engines(model_cache: str):
    resources_dir = _pl.Path(__file__).parent.resolve().parent
    models_dir = resources_dir / 'models'
    model_name = 'vicuna-7b-medusa'
    base_model_name = 'vicuna-7b-v1.3'
    medusa_model_name = 'medusa-vicuna-7b-v1.3'

    if model_cache:
        print(f"Copy model from {model_cache}")
        base_model_cache_dir = _pl.Path(model_cache) / base_model_name
        medusa_head_cache_dir = _pl.Path(model_cache) / medusa_model_name
        assert base_model_cache_dir.is_dir(), base_model_cache_dir
        assert medusa_head_cache_dir.is_dir(), medusa_head_cache_dir

        if _pf.system() == "Windows":
            wincopy(source=str(base_model_cache_dir),
                    dest=base_model_name,
                    isdir=True,
                    cwd=models_dir)
            wincopy(source=str(medusa_head_cache_dir),
                    dest=medusa_model_name,
                    isdir=True,
                    cwd=models_dir)
        else:
            run_command(["rsync", "-rlptD",
                         str(base_model_cache_dir), "."],
                        cwd=models_dir)
            run_command(["rsync", "-rlptD",
                         str(medusa_head_cache_dir), "."],
                        cwd=models_dir)

    base_model_dir = models_dir / base_model_name
    medusa_model_dir = models_dir / medusa_model_name
    assert base_model_dir.is_dir()
    assert medusa_model_dir.is_dir()

    engine_dir = models_dir / 'rt_engine' / model_name

    model_spec_obj = model_spec.ModelSpec('input_tokens.npy', _tb.DataType.HALF)
    model_spec_obj.use_gpt_plugin()
    model_spec_obj.set_kv_cache_type(_tb.KVCacheType.PAGED)
    model_spec_obj.use_packed_input()
    model_spec_obj.use_medusa()

    full_engine_path = engine_dir / model_spec_obj.get_model_path(
    ) / 'tp1-pp1-cp1-gpu'
    print(f"\nBuilding fp16 engine at {str(full_engine_path)}")
    build_engine(base_model_dir, medusa_model_dir, full_engine_path)

    print("Done.")


if __name__ == "__main__":
    parser = _arg.ArgumentParser()
    parser.add_argument("--model_cache",
                        type=str,
                        help="Directory where models are stored")

    build_engines(**vars(parser.parse_args()))
