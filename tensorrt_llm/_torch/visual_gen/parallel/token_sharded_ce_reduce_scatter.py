# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Copy-engine reduce-scatter for token-sharded TP's FP8 block-scale row boundaries.

At a token-sharded row boundary (``to_out``, ``down_proj``) every TP rank holds the row GEMM's
input for all ``tp * m`` padded rows and needs the sum over ranks of its own ``m`` rows. The
NCCL path runs the GEMM on all rows, then ``reduce_scatter`` (SM kernels that contend with the
GEMM; a bf16 ring at ``tp >= 3``). Here the GEMM runs per destination block -- block ``d`` of
the padded stream is exactly destination ``d``'s rows (``TokenShardPlan.row_start = d * m``) --
and each remote block is *pushed* straight into the destination's region of a symmetric-memory
slot with one copy-engine ``cudaMemcpyAsync`` on a side stream (no SMs) as soon as its GEMM is
enqueued; the own block is computed last, directly into the own region, while the pushes are
in flight. After the ``tp - 1`` ready flags one kernel reduces the ``[tp, m, N]`` view of the
slot in fixed source-rank order (fp32 adds, one bf16 rounding). The whole boundary is one custom
op (``trtllm::token_sharded_fp8_ce_gemm_reduce_scatter``) so a compiled block stays one graph.

Protocol (rank ``r``, one boundary on slot ``s``)
--------------------------------------------------
Producer side, :meth:`CeReduceScatter.push`, per destination ``d`` in
:attr:`CeReduceScatter.dest_order` (``r - 1, r - 2, ...``: at every step the ranks target
distinct destinations), on ``d``'s side stream after an event recorded behind block ``d``'s
GEMM on the current stream::

    wait(d, consumed(s))        # skipped on the slot's first use: d finished reducing slot s
    d's slot s, region r: copy_(block d)      # one CE copy of the contiguous bf16 [m, N] block
    signal(d, ready(s))         # same stream: ordered after the copy

Consumer side, on the current stream: the own block's GEMM writes region ``r`` of MY slot
(:meth:`CeReduceScatter.own_view`); :meth:`CeReduceScatter.wait_all` waits ``ready(s)`` from
every source in :attr:`CeReduceScatter.wait_order` (``r + 1, r + 2, ...``, the arrival order);
:func:`fixed_order_reduce_bf16` reduces :meth:`CeReduceScatter.partials`;
:meth:`CeReduceScatter.done` joins every side stream into the current stream and signals
``consumed(s)`` to every peer. Channels: ``ready(s) = s``, ``consumed(s) = slots + s``. One slot
by default: the only cross-boundary dependency is the producer's ``consumed`` wait on its side
stream, and the destination signals ``consumed`` right after its reduce while the next row
boundary is an attention or an MLP away, so a slow destination stalls a producer's side stream,
never its compute stream.

Every wait is satisfiable: ``ready(s)`` of boundary ``b`` is put by every producer inside its
``push(b)``, enqueued on side streams that only wait for ``consumed(s)`` of ``b - slots``, which
the destination put in ``done(b - slots)`` after a reduce that depended on ``ready(s)`` of
``b - slots``. One put per wait per ``(channel, src -> dst)``. A rank skipping ``done`` or
ranks pushing different slot sequences end in the signal timeout (a device-side trap), never
in silent corruption.

Streams and CUDA graphs
-----------------------
By default one side stream per destination (``tp - 1`` streams; measured 750 vs 731 GB/s and
the better pipelined boundary against a single side stream). With the gather's side stream and
the compute stream this fits the default 8 CUDA connections up to ``tp = 7``; at ``tp = 8`` two
streams share a hardware queue (performance only, no correctness impact).
``per_peer_streams=False`` (``CeReduceScatterState(per_peer_streams=False)``) selects one side
stream for every destination; it is the only knob, there is no environment variable.
Each push forks from the current stream through an event recorded behind its block's GEMM and
every side stream is joined back in ``done`` of the same boundary, so after ``done`` nothing is
pending on any side stream: no event recorded in one forward is waited on in a later one (a
CUDA-graph capture boundary is safe), ``drain`` is a plain check and no ``Tensor.record_stream``
is needed (the staging tensor and the op's inputs are alive for the op call and the join
orders every later reuse of their memory on the current stream behind the copies). Per-boundary
``torch.cuda.Event`` objects become graph nodes under capture. The pool is sized eagerly
(:meth:`CeReduceScatterState.prepare`, from ``TokenShardedTP.begin`` in the pipeline's eager
warm-up forwards; the op grows it on demand in eager mode as a backstop); reserving under
capture raises. Two rules are enforced rather than assumed: a graph bakes in the pool's
addresses and flags, so once any boundary was captured the pool may not grow
(:meth:`CeReduceScatter.reserve` raises), and a slot's first use (which skips the ``consumed``
wait) may not be captured (:meth:`CeReduceScatter.begin` raises; run at least ``slots`` eager
boundaries after reserving, as the pipeline's warm-up forwards do). The Triton reduce kernel is
JIT-compiled per ``(tp, numel)``: warm every served shape eagerly before capturing.

Numerics
--------
The per-block GEMM is bitwise the whole-M GEMM (DeepGEMM computes each output row from its own
activation row in one block-ordered K loop; the scales of a block are re-strided with one small
copy, :func:`token_sharded_ce_gather.mn_major_scales`). The reduce computes, for every element,
``out = bf16(((p_0' + p_1) + p_2) + ... + p_{tp-1})`` with every partial upcast to fp32, the
adds in source-rank order ``0 .. tp - 1`` whatever arrived first, and ``p_0' = bf16(p_0 + bias)``
when the row Linear's bias is given (the bias of ``tp_rank`` 0 only, added in bf16 to source 0's
partial exactly where the NCCL path's rank-0 Linear adds it), else ``p_0``. No atomics, no
accumulate-as-they-land: :func:`reference_fixed_order_reduce` (the plain torch chain) reproduces
it bitwise, which is what the tests assert. It is the precision class of NCCL's NVLS
reduce-scatter (fp32 accumulation, one rounding) and strictly tighter than the bf16 ring
(``tp - 1`` roundings): bitwise with NCCL at ``tp = 2`` (one add), not at ``tp >= 3``.

Failure triage
--------------
A rank that raises between ``push`` and ``done`` leaves its peers spinning in ``wait_signal``
until the timeout traps their CUDA context: a "device-side trap" on three of four ranks means
another rank raised; read that rank's traceback. Shape errors in the op body are checked
against the plan ints and the weight shape only, which are identical on every rank, so they
raise on every rank together; a failed op call resets the transport so the next ``begin`` does
not raise. The reset waits only for the ready flags that were put: a rank-symmetric raise after
``i`` of the ``tp - 1`` GEMM + push steps (a DeepGEMM failure, say) means every rank received
exactly the pushes of ``wait_order[:i]``, and :meth:`CeReduceScatter.reset` trims its pending
set to those before draining it, so the repaired protocol continues without a device trap.
"""

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import pad_up

from ...distributed.symm_mem_pool import (
    DEFAULT_ALIGN,
    DEFAULT_TIMEOUT_MS,
    SymmMemPool,
    probe_symmetric_memory,
)
from .token_sharded_ce_gather import (
    fp8_block_gemm_out,
    mn_major_scales,
    validate_fp8_block_consumer,
)
from .token_sharded_tp import FP8_BLOCK_SIZE, fp8_scale_cols

if TYPE_CHECKING:
    from triton import JITFunction

    from .token_sharded_tp import TokenShardPlan

__all__ = [
    "DEFAULT_ALIGN",
    "DEFAULT_TIMEOUT_MS",
    "REDUCE_BLOCK",
    "REDUCE_NUM_WARPS",
    "SLOTS",
    "Bf16RowsRegionLayout",
    "CeReduceScatter",
    "CeReduceScatterState",
    "ce_gemm_reduce_scatter_impl",
    "fixed_order_reduce_bf16",
    "get_rs_state",
    "reference_fixed_order_reduce",
    "register_rs_state",
    "release_rs_state",
    "validate_fp8_block_row_consumer",
]

SLOTS = 1
"""int: Slots the reduce-scatter transport alternates over (1: the slot is 1.55 GB at TP4 B=2
and the producer's ``consumed`` wait sits on its side stream; see the module docstring)."""

REDUCE_BLOCK = 2048
"""int: Elements per program of the fixed-order reduce kernel (fixed launch config, no
autotune: 2048 x 4 warps measured 6.5 TB/s at [4, 37800, 5120], ahead of 4096 x 8 warps)."""

REDUCE_NUM_WARPS = 4
"""int: Warps per program of the fixed-order reduce kernel."""


# =============================================================================
# Region layout
# =============================================================================


@dataclass(frozen=True)
class Bf16RowsRegionLayout:
    """One source's region of a reduce-scatter slot: a bf16 ``[m, N]`` partial, payload only.

    Every rank holds the same ``m`` (``TokenShardPlan.local_rows``), so one layout describes
    every region of a slot; a slot is ``tp_size`` regions in source-rank order. The regions sit
    :attr:`region_nbytes` apart when the slot is carved tightly for this layout; inside a pool
    they sit the POOL's region stride apart, which is larger once the pool grew for a bigger
    layout (grow-only), so :meth:`partials_view` takes that stride. Python ints only (capture
    safe). Same slot-carving surface (``region_nbytes``, ``slot_nbytes``) as the gather's
    ``RegionLayout``.

    Attributes:
        rows: ``m``, rows per destination (and per source region).
        cols: ``N``, the row Linear's ``out_features``.
        align: Byte alignment of a region (a power of two >= 16; the pool's ``DEFAULT_ALIGN``,
            so a slot of ``tp_size`` regions is carved exactly).
    """

    rows: int
    cols: int
    align: int = DEFAULT_ALIGN

    def __post_init__(self) -> None:
        if self.rows < 1 or self.cols < 1:
            raise ValueError(
                f"Bf16RowsRegionLayout needs rows >= 1 and cols >= 1 (got rows={self.rows}, "
                f"cols={self.cols})."
            )
        if self.align < 16 or self.align & (self.align - 1):
            raise ValueError(
                f"Bf16RowsRegionLayout align must be a power of two >= 16 (got {self.align})."
            )

    @property
    def payload_nbytes(self) -> int:
        """Bytes of the bf16 ``[m, N]`` partial."""
        return 2 * self.rows * self.cols

    @property
    def region_nbytes(self) -> int:
        return pad_up(self.payload_nbytes, self.align)

    @property
    def src_stride(self) -> int:
        """bf16 elements between consecutive sources' regions in a slot carved tightly for
        this layout (``region_nbytes / 2``): the default stride of :meth:`partials_view`. A
        pool that grew past this layout places the regions further apart (its
        ``region_nbytes``); the transport passes that stride explicitly."""
        return self.region_nbytes // 2

    def slot_nbytes(self, tp_size: int) -> int:
        """Bytes one slot needs: ``tp_size`` regions."""
        return tp_size * self.region_nbytes

    def consumer_view(self, region: torch.Tensor) -> torch.Tensor:
        """The bf16 ``[m, N]`` partial over a region's uint8 bytes (what a GEMM writes into
        the own region and what the reduce reads per source)."""
        self._check_region(region)
        return region[: self.payload_nbytes].view(torch.bfloat16).view(self.rows, self.cols)

    def producer_view(self, region: torch.Tensor) -> torch.Tensor:
        """The uint8 ``[2 * m * N]`` payload span of a region: what one push copies into."""
        self._check_region(region)
        return region[: self.payload_nbytes]

    def partials_view(
        self, slot: torch.Tensor, tp_size: int, region_stride_nbytes: int | None = None
    ) -> torch.Tensor:
        """The bf16 ``[tp_size, m, N]`` view over a slot whose regions sit
        ``region_stride_nbytes`` apart (strides ``(region_stride_nbytes / 2, N, 1)``).

        ``slot`` is a 1-D uint8 view of the slot's first region or of the whole slot; the view
        reaches ``tp_size`` regions from its start, which must lie inside its storage.
        ``region_stride_nbytes`` is the pool's region stride (``SymmMemPool.region_nbytes``,
        >= :attr:`region_nbytes` once the pool grew for a larger layout); None means a slot
        carved tightly for this layout (:attr:`src_stride`).
        """
        self._check_region(slot)
        stride = self.region_nbytes if region_stride_nbytes is None else region_stride_nbytes
        if stride < self.region_nbytes or stride % self.align:
            raise ValueError(
                f"Bf16RowsRegionLayout.partials_view: region_stride_nbytes must be a multiple of "
                f"{self.align} >= {self.region_nbytes}; got {stride}."
            )
        need = slot.storage_offset() + (tp_size - 1) * stride + self.payload_nbytes
        if slot.untyped_storage().nbytes() < need:
            raise ValueError(
                f"Bf16RowsRegionLayout.partials_view: the storage holds "
                f"{slot.untyped_storage().nbytes()} bytes, {tp_size} regions {stride} bytes "
                f"apart from byte {slot.storage_offset()} need {need}."
            )
        return slot.view(torch.bfloat16).as_strided(
            (tp_size, self.rows, self.cols), (stride // 2, self.cols, 1)
        )

    def _check_region(self, region: torch.Tensor) -> None:
        if region.dtype != torch.uint8 or region.dim() != 1 or region.numel() < self.region_nbytes:
            raise ValueError(
                f"Bf16RowsRegionLayout: need a 1-D uint8 region of >= {self.region_nbytes} "
                f"bytes; got {region.dtype} {tuple(region.shape)}."
            )
        if region.data_ptr() % 16:
            raise RuntimeError(
                f"Bf16RowsRegionLayout: region base {region.data_ptr():#x} is not 16-byte "
                "aligned (the GEMM's TMA store and the reduce's vector loads need it); the "
                "pool's region stride is misaligned."
            )


# =============================================================================
# The deterministic reduce
# =============================================================================


def reference_fixed_order_reduce(
    partials: Sequence[torch.Tensor] | torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """The reduce's contract as the plain torch chain: ``bf16((((p_0' + p_1) + p_2) + ...))``.

    Every partial is upcast to fp32 and added in source order; ``p_0' = bf16(p_0 + bias)`` when
    ``bias`` is given (source 0 carries the bias, as the NCCL path's rank-0 Linear does), else
    ``p_0``; one bf16 rounding at the end. IEEE fp32 adds and RNE casts, so
    :func:`fixed_order_reduce_bf16` reproduces it bitwise. CPU-testable; several launches and
    fp32 intermediates, so it is the reference and the fallback, not the hot path.

    Args:
        partials: ``tp`` bf16 ``[m, N]`` tensors in source-rank order (a list, or a
            ``[tp, m, N]`` tensor, which iterates over its first dimension).
        bias: ``[N]`` bf16 or None.
    """
    parts = list(partials)
    if not parts:
        raise ValueError("reference_fixed_order_reduce needs at least one partial.")
    acc = parts[0].float()
    if bias is not None:
        acc = (acc + bias.float()).to(torch.bfloat16).float()
    for p in parts[1:]:
        acc = acc + p.float()
    return acc.to(torch.bfloat16)


@functools.lru_cache(maxsize=1)
def _reduce_kernel() -> "JITFunction":
    """The Triton kernel, built on first use (so this module imports without Triton)."""
    import triton
    import triton.language as tl

    @triton.jit
    def _fixed_order_reduce_bf16_kernel(
        base_ptr,
        out_ptr,
        bias_ptr,
        src_stride,
        numel,
        n_cols,
        TP: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # out[i] = bf16(((p0'[i] + p1[i]) + p2[i]) + ...), every p upcast to fp32; p_r lives at
        # base_ptr + r * src_stride, flat over [m, N]; p0' = bf16(p0 + bias[i % N]) when HAS_BIAS.
        # Source order is fixed by the unrolled loop whatever arrived first; one RNE rounding at
        # the end. Triton vectorizes the loads (16 B) from the pointer and stride alignment it
        # specializes on: the region base is 256-byte aligned and src_stride a multiple of 128.
        # All index arithmetic in int64: Triton types a Python int argument i32 when it fits,
        # and r * src_stride would wrap for regions over 2^31 / (TP - 1) elements.
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        stride = src_stride.to(tl.int64)
        mask = offs < numel
        acc = tl.load(base_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(bias_ptr + offs % n_cols, mask=mask, other=0.0).to(tl.float32)
            acc = (acc + b).to(tl.bfloat16).to(tl.float32)
        for r in tl.static_range(1, TP):
            acc += tl.load(base_ptr + r * stride + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, acc.to(tl.bfloat16), mask=mask)

    return _fixed_order_reduce_bf16_kernel


def fixed_order_reduce_bf16(
    partials: torch.Tensor, bias: torch.Tensor | None, out: torch.Tensor
) -> torch.Tensor:
    """``out = bf16(fixed-order fp32 sum over the sources of partials, bias on source 0)``.

    One Triton launch with a fixed configuration (:data:`REDUCE_BLOCK` x :data:`REDUCE_NUM_WARPS`,
    JIT-compiled per ``(tp, numel)``), bitwise :func:`reference_fixed_order_reduce`. On CPU
    tensors (tests) the torch chain runs instead. The kernel indexes in int64, so the only size
    limits are the tensors' own (``numel`` and ``S`` below 2^63).

    Args:
        partials: bf16 ``[tp, m, N]`` with strides ``(S, N, 1)`` (the slot's
            :meth:`Bf16RowsRegionLayout.partials_view` with the pool's region stride; any
            source stride ``S``).
        bias: ``[N]`` bf16 or None.
        out: Contiguous bf16 ``[m, N]``.

    Returns:
        ``out``.
    """
    if partials.dim() != 3 or partials.dtype != torch.bfloat16:
        raise ValueError(
            f"fixed_order_reduce_bf16: partials must be bf16 [tp, m, N]; got {partials.dtype} "
            f"{tuple(partials.shape)}."
        )
    tp, m, n = partials.shape
    if partials.stride(2) != 1 or partials.stride(1) != n:
        raise ValueError(
            f"fixed_order_reduce_bf16: partials must have strides (S, N, 1); got "
            f"{tuple(partials.stride())} for N={n}."
        )
    if out.dtype != torch.bfloat16 or tuple(out.shape) != (m, n) or not out.is_contiguous():
        raise ValueError(
            f"fixed_order_reduce_bf16: out must be a contiguous bf16 [{m}, {n}]; got {out.dtype} "
            f"{tuple(out.shape)}."
        )
    if bias is not None and (bias.dtype != torch.bfloat16 or tuple(bias.shape) != (n,)):
        raise ValueError(
            f"fixed_order_reduce_bf16: bias must be bf16 [{n}] or None; got {bias.dtype} "
            f"{tuple(bias.shape)}."
        )
    if partials.device.type != "cuda":
        out.copy_(reference_fixed_order_reduce(partials, bias))
        return out
    numel = m * n
    grid = ((numel + REDUCE_BLOCK - 1) // REDUCE_BLOCK,)
    _reduce_kernel()[grid](
        partials,
        out,
        out if bias is None else bias,
        partials.stride(0),
        numel,
        n,
        TP=tp,
        HAS_BIAS=bias is not None,
        BLOCK=REDUCE_BLOCK,
        num_warps=REDUCE_NUM_WARPS,
    )
    return out


# =============================================================================
# The consumer check
# =============================================================================


def validate_fp8_block_row_consumer(linear: object) -> None:
    """Check once that a row ``Linear``'s operands fit the fused GEMM + reduce-scatter.

    The same operand contract as the gather's ``validate_fp8_block_consumer`` (contiguous
    float8_e4m3fn ``[N, K_local]`` weight with ``K_local % 128 == 0`` and the packed int32
    MN-major ``weight_scale`` that ``FP8BlockScalesLinearMethod.transform_weights`` produces),
    plus the bias, when present, a bf16 ``[N]`` (added inside the reduce on source 0). Called
    once after the weights are loaded (``TokenShardedTP`` does so at the first effective
    ``begin()``); the op body re-checks only dtypes and shapes per call, rank-symmetrically.

    Raises:
        ValueError: ``linear`` is not an FP8 block-scale ``Linear`` or an operand does not fit.
    """
    try:
        validate_fp8_block_consumer(linear)
    except ValueError as e:
        raise ValueError(
            f"copy-engine reduce-scatter: the row consumer's operands do not fit the fused "
            f"GEMM: {e}"
        ) from None
    bias = getattr(linear, "bias", None)
    n = linear.weight.shape[0]
    if bias is not None and (bias.dtype != torch.bfloat16 or tuple(bias.shape) != (n,)):
        raise ValueError(
            f"copy-engine reduce-scatter: the row consumer's bias must be bf16 [{n}]; got "
            f"{bias.dtype} {tuple(bias.shape)}."
        )


# =============================================================================
# Transport
# =============================================================================


class CeReduceScatter:
    """Per-destination copy-engine reduce-scatter over a ``SymmMemPool`` (module docstring).

    Owns the pool for its lifetime (data channels ``0 .. 2 * slots - 1``). Per boundary:
    :meth:`begin`, a GEMM + :meth:`push` per destination in :attr:`dest_order`, the own GEMM
    into :meth:`own_view`, :meth:`wait_all`, the reduce over :meth:`partials`, :meth:`done`.
    :meth:`reserve` sizes the pool eagerly and collectively.

    Args:
        pool: The TP group's reduce-scatter pool (its ``signal`` / ``wait`` carry the timeout).
        tp_size: Ranks in the TP group.
        rank: This process's rank in the group.
        device: The CUDA device the side streams run on.
        slots: Slots to alternate over.
        per_peer_streams: One side stream per destination (default) or one for all.
    """

    def __init__(
        self,
        pool: SymmMemPool,
        tp_size: int,
        rank: int,
        device: torch.device,
        *,
        slots: int = SLOTS,
        per_peer_streams: bool = True,
    ) -> None:
        if tp_size < 2 or not 0 <= rank < tp_size:
            raise ValueError(
                f"CeReduceScatter needs tp_size >= 2 and 0 <= rank < tp_size (got {rank}/{tp_size})."
            )
        if slots < 1:
            raise ValueError(f"CeReduceScatter needs slots >= 1 (got {slots}).")
        self.pool = pool
        self.tp_size = tp_size
        self.rank = rank
        self.device = device
        self.slots = slots
        self.per_peer_streams = per_peer_streams
        self.dest_order: tuple[int, ...] = tuple((rank - i) % tp_size for i in range(1, tp_size))
        """Destinations in push order ``me - 1, me - 2, ...``: at step ``i`` every rank targets
        a distinct destination, whose ``wait_order`` (``me + 1, me + 2, ...``) is the arrival
        order."""
        self.wait_order: tuple[int, ...] = tuple((rank + i) % tp_size for i in range(1, tp_size))
        """Remote sources in the order their ready flags are waited: ``me + 1, me + 2, ...``."""
        n_streams = tp_size - 1 if per_peer_streams else 1
        self._streams = [torch.cuda.Stream(device=device) for _ in range(n_streams)]
        self._stream_of: dict[int, torch.cuda.Stream] = {
            dst: self._streams[i if per_peer_streams else 0]
            for i, dst in enumerate(self.dest_order)
        }
        self._layout: list[Bf16RowsRegionLayout | None] = [None] * slots
        self._own_views: dict[int, torch.Tensor] = {}
        self._partials_views: dict[int, torch.Tensor] = {}
        self._peer_views: dict[tuple[int, int], torch.Tensor] = {}
        self._uses = [0] * slots
        self._in_flight = [False] * slots
        self._pushed: list[list[torch.cuda.Event]] = [[] for _ in range(slots)]
        self._pushed_to: list[set[int]] = [set() for _ in range(slots)]
        self._ready_pending: list[set[int]] = [set() for _ in range(slots)]
        self._slot_nbytes = 0
        self._captured = False  # a CUDA graph holds the pool's addresses: no growth after it

    # --- channels / streams / capacity ------------------------------------------------------

    def _ready_channel(self, slot: int) -> int:
        return slot

    def _consumed_channel(self, slot: int) -> int:
        return self.slots + slot

    @property
    def min_channels(self) -> int:
        """Data channels the pool must offer this transport."""
        return 2 * self.slots

    @property
    def slot_nbytes(self) -> int:
        """Bytes per slot of the current reservation (0 before :meth:`reserve`)."""
        return self._slot_nbytes

    def side_stream(self, dst: int) -> torch.cuda.Stream:
        """The side stream that carries the pushes to destination ``dst`` (tests inject delays
        on it to perturb arrival order)."""
        if dst not in self._stream_of:
            raise ValueError(f"destination {dst} is not a peer of rank {self.rank}/{self.tp_size}.")
        return self._stream_of[dst]

    def covers(self, layout: Bf16RowsRegionLayout) -> bool:
        """Whether the current reservation holds ``layout`` on every slot."""
        return self._slot_nbytes >= layout.slot_nbytes(self.tp_size)

    def reserve(self, layout: Bf16RowsRegionLayout) -> bool:
        """Size the pool for ``layout`` on :attr:`slots` slots (collective, grow-only).

        Call eagerly, every rank in the same order, never under capture (the pool raises) and
        never for a larger layout once a boundary was captured into a CUDA graph (raises: the
        graph holds the current allocation's addresses and flags). Returns True when the pool
        (re)allocated: the signal pad is then zeroed, so the protocol restarts from "first use"
        and the cached views are dropped.
        """
        if any(self._in_flight):
            raise RuntimeError(
                "CeReduceScatter.reserve with a boundary in flight; call done() first."
            )
        if not self.covers(layout):
            if self._captured:
                raise RuntimeError(
                    f"CeReduceScatter.reserve: the symmetric pool would grow after a CUDA graph "
                    f"captured its buffers (the pool holds {self._slot_nbytes} bytes per slot, "
                    f"the layout needs {layout.slot_nbytes(self.tp_size)}); warm up the largest "
                    "token shape first, or run the warm-up with capture disabled before "
                    "capturing."
                )
            # Drop the views into the current allocation first so the pool can free it before
            # allocating the larger one (otherwise both are mapped at once).
            self._drop_views()
        realloc = self.pool.reserve(layout.slot_nbytes(self.tp_size), self.slots, self.min_channels)
        if realloc:
            self._uses = [0] * self.slots
            self._drop_views()
        self._slot_nbytes = self.pool.nbytes // self.pool.slots
        if self._slot_nbytes // self.tp_size < layout.region_nbytes or self.pool.slots < self.slots:
            raise RuntimeError(
                f"CeReduceScatter.reserve: the pool holds {self.pool.slots} x {self._slot_nbytes} "
                f"bytes after reserve(); need {self.slots} x {layout.slot_nbytes(self.tp_size)}."
            )
        return realloc

    def _drop_views(self) -> None:
        # Views into the pool's allocation are valid for one allocation and one slot layout.
        self._layout = [None] * self.slots
        self._own_views.clear()
        self._partials_views.clear()
        self._peer_views.clear()

    def _check_slot(self, slot: int) -> None:
        if not 0 <= slot < self.slots:
            raise ValueError(f"slot {slot} out of range (transport has {self.slots} slots).")

    def _require_in_flight(self, slot: int, op: str) -> None:
        self._check_slot(slot)
        if not self._in_flight[slot]:
            raise RuntimeError(
                f"CeReduceScatter.{op}(slot={slot}) needs a begin({slot}) in flight."
            )

    def _bind_layout(self, slot: int, layout: Bf16RowsRegionLayout) -> None:
        # Region views are valid for the layout a slot was begun with: drop that slot's cache
        # when its layout changes; another slot (possibly in flight) keeps its own.
        if layout != self._layout[slot]:
            self._layout[slot] = layout
            self._own_views.pop(slot, None)
            self._partials_views.pop(slot, None)
            for key in [key for key in self._peer_views if key[0] == slot]:
                del self._peer_views[key]

    def _peer_view(self, slot: int, peer: int) -> torch.Tensor:
        """MY region's payload span in ``peer``'s slot: what my push writes (cached)."""
        key = (slot, peer)
        view = self._peer_views.get(key)
        if view is None:
            layout = self._layout[slot]
            view = layout.producer_view(self.pool.peer_region(peer, slot, 0, layout.region_nbytes))
            self._peer_views[key] = view
        return view

    # --- the boundary -----------------------------------------------------------------------

    def begin(self, slot: int, layout: Bf16RowsRegionLayout) -> None:
        """Open a boundary on slot ``slot`` for ``layout`` (the plan's ``m`` x the consumer's
        ``N``): bind the layout, start the push bookkeeping. Enqueues nothing."""
        self._check_slot(slot)
        if self._in_flight[slot]:
            raise RuntimeError(
                f"CeReduceScatter.begin(slot={slot}) before done({slot}) of its previous use: the "
                "peers would wait for the consumed flag forever."
            )
        if not self.covers(layout):
            raise RuntimeError(
                f"CeReduceScatter.begin: the pool holds {self._slot_nbytes} bytes per slot, a "
                f"[{layout.rows}, {layout.cols}] boundary needs "
                f"{layout.slot_nbytes(self.tp_size)}; reserve() eagerly for the largest boundary "
                "first."
            )
        capturing = torch.cuda.is_current_stream_capturing()
        if self._uses[slot] == 0 and capturing:
            raise RuntimeError(
                f"CeReduceScatter.begin: first use of slot {slot} under CUDA-graph capture (its "
                f"consumed wait would be missing from every replay); run at least {self.slots} "
                "eager boundaries after reserving before capturing."
            )
        if capturing:
            self._captured = True
        self._bind_layout(slot, layout)
        self._pushed[slot] = []
        self._pushed_to[slot] = set()
        self._ready_pending[slot] = set(self.wait_order)
        self._in_flight[slot] = True

    def push(self, dst: int, slot: int, block: torch.Tensor) -> None:
        """Push ``block`` (this rank's K-partial of destination ``dst``'s rows, a contiguous
        bf16 ``[m, N]``) into ``dst``'s slot ``slot`` and raise its ready flag.

        Forks ``dst``'s side stream from the current stream behind the GEMM that produced
        ``block``; enqueues only. Each destination exactly once per boundary, in any order.
        """
        self._require_in_flight(slot, "push")
        if dst not in self._stream_of:
            raise ValueError(
                f"CeReduceScatter.push: destination {dst} is not a peer of rank "
                f"{self.rank}/{self.tp_size}."
            )
        if dst in self._pushed_to[slot]:
            raise RuntimeError(
                f"CeReduceScatter.push(dst={dst}, slot={slot}) twice in one boundary: the peer "
                "would see a second ready flag."
            )
        layout = self._layout[slot]
        if (
            block.dtype != torch.bfloat16
            or tuple(block.shape) != (layout.rows, layout.cols)
            or not block.is_contiguous()
        ):
            raise ValueError(
                f"CeReduceScatter.push: block must be a contiguous bf16 [{layout.rows}, "
                f"{layout.cols}]; got {block.dtype} {tuple(block.shape)}."
            )
        fork = torch.cuda.Event()
        fork.record(torch.cuda.current_stream())
        stream = self._stream_of[dst]
        stream.wait_event(fork)
        with torch.cuda.stream(stream):
            if self._uses[slot] > 0:
                self.pool.wait(dst, self._consumed_channel(slot))
            self._peer_view(slot, dst).copy_(block.view(torch.uint8).view(-1))
            self.pool.signal(dst, self._ready_channel(slot))
            pushed = torch.cuda.Event()
            pushed.record(stream)
        self._pushed[slot].append(pushed)
        self._pushed_to[slot].add(dst)

    def own_view(self, slot: int) -> torch.Tensor:
        """Region ``rank`` of MY slot as the bf16 ``[m, N]`` the own block's GEMM writes (only
        this rank ever writes it, so no peer races it)."""
        self._require_in_flight(slot, "own_view")
        view = self._own_views.get(slot)
        if view is None:
            layout = self._layout[slot]
            region = self.pool.region(slot, self.rank, 0, layout.region_nbytes)
            view = layout.consumer_view(region)
            self._own_views[slot] = view
        return view

    def wait_all(self, slot: int) -> None:
        """Make the current stream wait for every remote source's ready flag (in
        :attr:`wait_order`); after it the slot holds all ``tp`` partials."""
        self._require_in_flight(slot, "wait_all")
        for src in self.wait_order:
            if src in self._ready_pending[slot]:
                self.pool.wait(src, self._ready_channel(slot))
                self._ready_pending[slot].discard(src)

    def partials(self, slot: int) -> torch.Tensor:
        """The slot's ``[tp, m, N]`` bf16 view in source-rank order (strides
        ``(pool.region_nbytes / 2, N, 1)``): the reduce's input. Valid after :meth:`wait_all`.

        The sources' regions sit the POOL's region stride apart (where the pushes and the own
        GEMM land: ``pool.peer_region`` / ``pool.region``), not the layout's packed
        ``src_stride``: the pool is grow-only, so after a larger reservation (a bigger token
        shape in the warm-up, a consumer with a larger ``N``) its regions are further apart
        than this boundary's layout alone would carve them.
        """
        self._require_in_flight(slot, "partials")
        view = self._partials_views.get(slot)
        if view is None:
            layout = self._layout[slot]
            region0 = self.pool.region(slot, 0, 0, layout.region_nbytes)
            view = layout.partials_view(
                region0, self.tp_size, region_stride_nbytes=self.pool.region_nbytes
            )
            self._partials_views[slot] = view
        return view

    def done(self, slot: int) -> None:
        """Join every side stream into the current stream and release slot ``slot`` to the
        peers. Call after the reduce was issued on the current stream."""
        self._require_in_flight(slot, "done")
        missing = set(self.dest_order) - self._pushed_to[slot]
        if missing:
            raise RuntimeError(
                f"CeReduceScatter.done(slot={slot}) before pushing to destination(s) "
                f"{sorted(missing)}: they would wait for the ready flag forever."
            )
        self._release(slot)

    def _release(self, slot: int) -> None:
        # One put per wait: consume the ready flags no wait_all() waited for before releasing.
        for src in sorted(self._ready_pending[slot]):
            self.pool.wait(src, self._ready_channel(slot))
        self._ready_pending[slot].clear()
        current = torch.cuda.current_stream()
        for pushed in self._pushed[slot]:
            current.wait_event(pushed)
        for dst in self.dest_order:
            self.pool.signal(dst, self._consumed_channel(slot))
        self._pushed[slot] = []
        self._pushed_to[slot] = set()
        self._in_flight[slot] = False
        self._uses[slot] += 1

    def drain(self) -> None:
        """Check that no boundary is in flight; nothing is pending on the side streams after
        :meth:`done`, so this touches no stream (safe under capture)."""
        for slot in range(self.slots):
            if self._in_flight[slot]:
                raise RuntimeError(
                    f"CeReduceScatter.drain with a boundary in flight on slot {slot}."
                )

    def reset(self) -> None:
        """Recover from a failed boundary: release every slot in flight (consume the ready flags
        that were put, join the side streams, signal consumed), so the next ``begin`` does not
        raise. Repairs the protocol when every rank failed at the same point; a peer that is
        already trapped cannot be helped.

        Under rank symmetry a rank that pushed to its first ``i`` destinations received the
        pushes of exactly ``wait_order[:i]`` (source ``s`` reaches rank ``r`` at its step
        ``s - r - 1 mod tp``), so only those ready flags are waited for; waiting for the others
        would spin to the deadline. Signalling ``consumed`` to every destination stays right: a
        destination that did not push to us consumes the extra flag at its next push, in place
        of the ``done`` of the failed boundary that never ran."""
        for slot in range(self.slots):
            if self._in_flight[slot]:
                pushed = len(self._pushed_to[slot])
                self._ready_pending[slot] &= set(self.wait_order[:pushed])
                self._release(slot)
        current = torch.cuda.current_stream()
        for stream in self._streams:
            current.wait_stream(stream)

    def close(self) -> None:
        """Release the pool (collective by symmetry: every rank must call it)."""
        self.drain()
        self._drop_views()
        self._uses = [0] * self.slots
        self._slot_nbytes = 0
        self.pool.release()


# =============================================================================
# Per-group state and registry
# =============================================================================


class CeReduceScatterState:
    """The copy-engine reduce-scatter state of one TP group: its own pool (separate from the
    gather's: different region size, no slot aliasing hazard), transport, slot counter, probe
    verdict.

    Created and registered by ``TokenShardedTP`` (two instances over one group, e.g. the two
    Wan2.2 experts, share it), released by ``TokenShardedTP.close()``; a closed state refuses
    :meth:`prepare`. :meth:`prepare` runs eagerly from ``begin()``; the first call probes the
    group's eligibility, rank-agreed, and the verdict decides :attr:`effective` for the state's
    lifetime.

    Args:
        group: The TP process group.
        group_name: ``group.group_name`` (the registry key and the op's handle).
        timeout_ms: Device-side deadline of every signal wait (:data:`DEFAULT_TIMEOUT_MS`).
        per_peer_streams: One push side stream per destination (default) or one for all.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        group_name: str,
        *,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        per_peer_streams: bool = True,
    ) -> None:
        if group is None:
            raise ValueError(
                "CeReduceScatterState needs a torch.distributed process group; got None."
            )
        if timeout_ms < 0:
            raise ValueError(f"CeReduceScatterState needs timeout_ms >= 0 (got {timeout_ms}).")
        self.group = group
        self.group_name = group_name
        self.timeout_ms = timeout_ms
        self.per_peer_streams = per_peer_streams
        self.tp_size: int = dist.get_world_size(group)
        self.rank: int = dist.get_rank(group)
        self.consumer_ks: set[int] = set()
        """set[int]: The noted row consumers' ``K_local`` (``in_features``)."""
        self.consumer_ns: set[int] = set()
        """set[int]: The noted row consumers' ``N`` (``out_features``), which size the pool."""
        self.effective: bool = False
        """bool: Whether the copy-engine reduce-scatter is in use for this group (False before
        the probe and after a failed one; the TP then keeps the NCCL reduce-scatter)."""
        self.reason: str = "not probed yet"
        """str: Why the probe failed, or "" after a successful one."""
        self.pool: SymmMemPool | None = None
        self.transport: CeReduceScatter | None = None
        self._probed = False
        self._closed = False
        self._counter = 0
        self._device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    # --- conversion time ------------------------------------------------------------------

    def note_row_consumer(self, k_local: int, out_features: int | None = None) -> None:
        """Record a row consumer: ``k_local`` (``in_features``, a positive multiple of 128, the
        fused GEMM's K) and, when given, ``out_features`` (``N``), which lets :meth:`prepare`
        size the pool eagerly; without it the op sizes the pool at its first eager call."""
        if k_local < 1 or k_local % FP8_BLOCK_SIZE:
            raise ValueError(
                f"note_row_consumer: K_local must be a positive multiple of {FP8_BLOCK_SIZE}; got "
                f"{k_local}."
            )
        if out_features is not None and out_features < 1:
            raise ValueError(f"note_row_consumer: N must be >= 1; got {out_features}.")
        self.consumer_ks.add(int(k_local))
        if out_features is not None:
            self.consumer_ns.add(int(out_features))

    # --- begin() ----------------------------------------------------------------------------

    def prepare(self, plan: "TokenShardPlan") -> None:
        """Probe once, start the forward's slot sequence and size the pool for ``plan``.

        Eager, every rank in the same order (``TokenShardedTP.begin``). Checks that the
        previous boundary chain is complete and restarts the slots at 0 (:meth:`begin_forward`,
        so the caller need not call it), then sizes the pool for ``plan.local_rows`` x the
        largest noted consumer ``N``, grow-only. Under CUDA-graph capture only the slot counter
        is reset; a probe or an allocation under capture raises, as does a closed state.
        """
        if self._closed:
            raise RuntimeError(
                f"copy-engine reduce-scatter: the state of TP group {self.group_name!r} is closed "
                "(TokenShardedTP.close() ran); build a new TokenShardedTP for further forwards."
            )
        capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        if not self._probed:
            if capturing:
                raise RuntimeError(
                    "copy-engine reduce-scatter: the first TokenShardedTP.begin() of a group ran "
                    "under CUDA-graph capture; run an eager warm-up forward first."
                )
            self._probe()
        if not self.effective:
            return
        self.begin_forward()
        if not self.consumer_ns:
            return
        layout = Bf16RowsRegionLayout(plan.local_rows, max(self.consumer_ns))
        if self.transport.covers(layout):
            return
        if capturing:
            raise RuntimeError(
                f"copy-engine reduce-scatter: TokenShardedTP.begin() under CUDA-graph capture "
                f"needs a pool for {layout.slot_nbytes(self.tp_size)} bytes per slot, the pool "
                f"holds {self.transport.slot_nbytes}; run an eager forward of this shape first."
            )
        self.transport.reserve(layout)  # logs the size, the deadline and the triage hint
        logger.info_once(
            f"Token-sharded TP ({self.group_name}): copy-engine reduce-scatter on.",
            key=("token_sharded_ce_reduce_scatter", self.group_name),
        )

    def begin_forward(self) -> None:
        """Start a forward: the previous boundary chain must be complete; slots restart at 0."""
        if self.transport is not None:
            self.transport.drain()
        self._counter = 0

    def next_slot(self) -> int:
        """The slot of the next boundary (alternating over :data:`SLOTS`; reset by
        :meth:`begin_forward`)."""
        slot = self._counter % SLOTS
        self._counter += 1
        return slot

    # --- probe --------------------------------------------------------------------------------

    def _probe(self) -> None:
        # Rank-agreed: >= 2 ranks, a CUDA NCCL group, a working tiny rendezvous and enough
        # signal channels; the reason names the declining ranks. Probed once per state.
        ok, reason = probe_symmetric_memory(self.group, self._device, min_channels=2 * SLOTS)
        if ok:
            self.pool = SymmMemPool(self.group, self._device, self.timeout_ms, align=DEFAULT_ALIGN)
            self.transport = CeReduceScatter(
                self.pool,
                self.tp_size,
                self.rank,
                self._device,
                slots=SLOTS,
                per_peer_streams=self.per_peer_streams,
            )
        else:
            logger.warning(
                f"Token-sharded TP ({self.group_name}): copy-engine reduce-scatter unavailable "
                f"({reason}); keeping the NCCL reduce-scatter."
            )
        self.effective = ok
        self.reason = reason
        self._probed = True

    # --- teardown -----------------------------------------------------------------------------

    def close(self) -> None:
        """Release the pool and unregister (idempotent; collective by symmetry when a pool
        exists). Call on every rank before the process group is destroyed. The state is then
        closed: not effective, and :meth:`prepare` raises (a sibling ``TokenShardedTP`` that
        still holds it gets a clear error rather than a silent fallback)."""
        if self.transport is not None:
            self.transport.close()
        self.transport = None
        self.pool = None
        self.effective = False
        self.reason = "closed"
        self._closed = True
        if _STATES.get(self.group_name) is self:
            del _STATES[self.group_name]


_STATES: dict[str, CeReduceScatterState] = {}


def get_rs_state(group_name: str) -> CeReduceScatterState:
    """The registered state of ``group_name``; raises ``KeyError`` when there is none."""
    state = _STATES.get(group_name)
    if state is None:
        raise KeyError(group_name)
    return state


def register_rs_state(state: CeReduceScatterState) -> CeReduceScatterState:
    """Register ``state`` for its group name, or return the one already registered for it."""
    return _STATES.setdefault(state.group_name, state)


def release_rs_state(group_name: str) -> None:
    """Close and unregister the state of ``group_name`` (no-op when there is none)."""
    state = _STATES.get(group_name)
    if state is not None:
        state.close()


# =============================================================================
# The op body
# =============================================================================


def _check_op_operands(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    w_fp8: torch.Tensor,
    w_sf: torch.Tensor,
    bias: torch.Tensor | None,
    padded_rows: int,
) -> None:
    # Against the plan ints and the weight shape only: identical on every rank, so a raise here
    # is rank-symmetric and leaves no peer waiting.
    if w_fp8.dim() != 2 or w_fp8.dtype != torch.float8_e4m3fn or w_sf.dtype != torch.int32:
        raise ValueError(
            "token_sharded_fp8_ce_gemm_reduce_scatter: the weight must be a float8_e4m3fn "
            f"[N, K_local] with the packed int32 weight_scale (post_load_weights); got "
            f"{w_fp8.dtype} {tuple(w_fp8.shape)} / {w_sf.dtype}."
        )
    n, k = w_fp8.shape
    if x_fp8.dtype != torch.float8_e4m3fn or tuple(x_fp8.shape) != (padded_rows, k):
        raise ValueError(
            f"token_sharded_fp8_ce_gemm_reduce_scatter: expected the padded float8_e4m3fn "
            f"[{padded_rows}, {k}] row-path input; got {x_fp8.dtype} {tuple(x_fp8.shape)}."
        )
    if x_sf.dtype != torch.int32 or tuple(x_sf.shape) != (padded_rows, fp8_scale_cols(k)):
        raise ValueError(
            f"token_sharded_fp8_ce_gemm_reduce_scatter: expected int32 packed scales "
            f"[{padded_rows}, {fp8_scale_cols(k)}]; got {x_sf.dtype} {tuple(x_sf.shape)}."
        )
    if bias is not None and (bias.dtype != torch.bfloat16 or tuple(bias.shape) != (n,)):
        raise ValueError(
            f"token_sharded_fp8_ce_gemm_reduce_scatter: bias must be bf16 [{n}] or None; got "
            f"{bias.dtype} {tuple(bias.shape)}."
        )


def ce_gemm_reduce_scatter_impl(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    w_fp8: torch.Tensor,
    w_sf: torch.Tensor,
    bias: torch.Tensor | None,
    group_name: str,
    tp_rank: int,
    tp_size: int,
    batch_size: int,
    seq_len: int,
    padded_seq_len: int,
) -> torch.Tensor:
    """Body of ``trtllm::token_sharded_fp8_ce_gemm_reduce_scatter``: per-destination GEMM +
    push, own GEMM, wait, fixed-order reduce, done.

    Returns this rank's reduced ``[m, N]`` bf16 rows of the padded stream (``m = batch_size *
    padded_seq_len / tp_size``; the caller takes its ``local_view``). Order on the current
    stream: for each destination in ``transport.dest_order`` the GEMM of its row block into a
    transient staging tensor, then its push (side stream); the own block's GEMM straight into
    the own region; the ``tp - 1`` ready waits; the reduce with ``bias`` on source 0; the join
    and the consumed signals.

    Args:
        x_fp8: The padded ``[batch_size * padded_seq_len, K_local]`` float8_e4m3fn row-path
            input of all tokens (``quantize_fp8_block`` of ``pad_row_input(act)``).
        x_sf: Its ``(batch_size * padded_seq_len, P)`` int32 packed UE8M0 scales, any stride.
        w_fp8: The row Linear's ``[N, K_local]`` float8_e4m3fn weight.
        w_sf: Its packed int32 weight scale.
        bias: The row Linear's ``[N]`` bf16 bias on ``tp_rank`` 0, None elsewhere and when the
            Linear has none (added inside the reduce to source 0's partial).
        group_name: The TP group's c10d name (the registry key).
        tp_rank: This rank in the group.
        tp_size: Ranks in the group.
        batch_size: ``B``.
        seq_len: ``S`` (not used by the body: the output keeps the pad rows, as the NCCL
            reduce-scatter's does; part of the plan ints the graph specializes on).
        padded_seq_len: ``S_pad``.
    """
    del seq_len
    try:
        state = get_rs_state(group_name)
    except KeyError:
        raise RuntimeError(
            f"token_sharded_fp8_ce_gemm_reduce_scatter: no CeReduceScatterState is registered "
            f"for TP group {group_name!r}; TokenShardedTP(reduce_scatter_mode='copy_engine') "
            "registers one at its first begin()."
        ) from None
    transport = state.transport
    if transport is None:
        raise RuntimeError(
            f"token_sharded_fp8_ce_gemm_reduce_scatter: the copy-engine reduce-scatter is not "
            f"effective for TP group {group_name!r} ({state.reason}); the adapters must take the "
            "NCCL path."
        )
    if tp_size != transport.tp_size or tp_rank != transport.rank:
        raise ValueError(
            f"token_sharded_fp8_ce_gemm_reduce_scatter: plan rank {tp_rank}/{tp_size} does not "
            f"match the group's {transport.rank}/{transport.tp_size}."
        )
    padded_rows = batch_size * padded_seq_len
    if padded_rows % tp_size:
        raise ValueError(
            f"token_sharded_fp8_ce_gemm_reduce_scatter: B * S_pad = {padded_rows} is not a "
            f"multiple of tp_size={tp_size}."
        )
    m = padded_rows // tp_size
    _check_op_operands(x_fp8, x_sf, w_fp8, w_sf, bias, padded_rows)
    n = w_fp8.shape[0]
    layout = Bf16RowsRegionLayout(m, n)
    if not transport.covers(layout):
        # Eager backstop (rank-symmetric: every rank calls the op with the same plan ints and
        # N): a consumer whose N was not noted before begin(). Under capture the pool raises.
        logger.info(
            f"Token-sharded TP ({group_name}): sizing the reduce-scatter pool for [{m}, {n}] at "
            "the op's first eager call; note the consumer's out_features before begin() to "
            "reserve eagerly."
        )
        transport.reserve(layout)
    x_fp8 = x_fp8.contiguous()
    slot = state.next_slot()
    try:
        transport.begin(slot, layout)
        staging = torch.empty(((tp_size - 1) * m, n), dtype=torch.bfloat16, device=x_fp8.device)
        for i, dst in enumerate(transport.dest_order):
            rows = slice(dst * m, (dst + 1) * m)
            block = staging[i * m : (i + 1) * m]
            fp8_block_gemm_out(x_fp8[rows], mn_major_scales(x_sf[rows]), w_fp8, w_sf, block)
            transport.push(dst, slot, block)
        rows = slice(tp_rank * m, (tp_rank + 1) * m)
        fp8_block_gemm_out(
            x_fp8[rows], mn_major_scales(x_sf[rows]), w_fp8, w_sf, transport.own_view(slot)
        )
        transport.wait_all(slot)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x_fp8.device)
        fixed_order_reduce_bf16(transport.partials(slot), bias, out)
        transport.done(slot)
    except Exception:
        transport.reset()
        raise
    return out
