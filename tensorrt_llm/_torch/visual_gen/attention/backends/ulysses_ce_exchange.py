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
"""Head-chunked Ulysses attention exchange over copy-engine pushes.

This module replaces the synchronous, unfused ``UlyssesAttention`` exchange (three ``all_to_all_4d``
of q, k, v, one attention call, one ``all_to_all_4d`` of the output) with a per-head-group pipeline
over ``SymmMemPool``: the q | k | v blocks of one head group are pushed into the peers' input slots
by the copy engines, the attention of group g runs while the blocks of group g + 1 are still
arriving, and the output rows of group g land in the peers' final tensors under the attention of
group g + 1. This commit holds the head plan only: the protocol (per-group flag pool, per-workspace
data pools, the one-launch wait, the cuda-python copies), the workspaces, the state and the
custom-op body arrive in the next commit.

Why
---
Attention is independent per head, so a rank's local heads can be cut into ordered groups and the
exchange pipelined per group: only the first group's input transfer and the last group's output
transfer stay exposed. How the heads are cut is a trade: small first and last groups shorten the
exposed transfers, but every group is one attention launch, and a launch whose CTA count is not a
multiple of the SM count pays for a partial wave. :class:`HeadPlan` holds the cut; the wave-time
rule (:func:`auto_choice`) picks it from the geometry so that chunking never costs more wave-times
than the stock single call.

Plan
----
A :class:`HeadPlan` is the ordered head groups of one rank: ``groups[g]`` heads each,
``offsets[g]`` = first local head of group g. Named plans (:func:`named_groups`): ``per_head`` =
``[1] * H_local``; ``whole`` = ``[H_local]`` (the stock single call); ``first_small`` =
``[1, H_local - 1]``; ``ends_small`` = ``[1, H_local - 2, 1]``. An explicit comma list such as
``1,4,5`` (:func:`parse_groups`) may use ``rest`` once. ``auto`` applies the wave-time rule and
needs the GLOBAL sequence length ``S = P * S_local``, so it is resolved where S is known (the first
forward per shape). Every rank derives the plan from the same ``(H_local, S, B, sms)``, so every
rank cuts the same groups; the protocol checks the groups rank-symmetrically with its per-shape
gather before any pool is reserved.

Wave-time rule
--------------
cuDNN's sm100 flash fprop kernel launches one CTA per (query tile of ``CUDNN_Q_TILE`` rows, head,
batch element), one CTA per SM, and a partial wave costs a whole one, so a plan costs ``W = sum_g
ceil(B * h_g * ceil(S / tile) / sms)`` wave-times (:func:`wave_times`). ``auto`` takes the smallest
W over :func:`auto_candidates` (``whole``, ``per_head``, ``first_small``, ``ends_small``, equal
groups for every divisor of H_local, ``first_<k>``, ``ends_<k>``); ties go to the smallest first
group, then the smallest last group, then the fewest groups (:func:`rank_key`). A sum of ceilings
is at least the ceiling of the sum, so the whole-plan W is a lower bound and the chosen plan never
costs more wave-times than the stock single call. Measured on B200 (4 ranks, fused producer): at
Wan 720p/81f one head is exactly two waves and every plan ties (``ends_small``); at 480p/81f one
head is 0.86 wave, one-head groups leave 14 % of the SMs idle and ``per_head`` (W 10) ran 0.908x
against 0.85-0.86x for the W-9 plans.

Rules
-----
* Pure integer arithmetic: nothing here touches a stream, a pool or a device, except
  :func:`device_sm_count`, which reads the SM count of a CUDA device and answers B200's 148 for
  CPU devices or without CUDA (the rule is then evaluated for B200, which CPU tests rely on).
* ``auto`` without ``S`` raises: the rule's input is the global sequence length, never a guess.
* A plan is validated at construction: positive ``H_local`` and ``sms``, every group at least one
  head, groups summing to ``H_local``. An empty ``groups`` means one head per group.
* The ``auto`` decision (every candidate's W and the reason) is logged once per process per
  ``(H_local, S, B, sms)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from tensorrt_llm.logger import logger

__all__ = [
    "CUDNN_Q_TILE",
    "NAMED_SPECS",
    "SM_COUNT_B200",
    "AutoChoice",
    "HeadPlan",
    "auto_candidates",
    "auto_choice",
    "device_sm_count",
    "named_groups",
    "parse_groups",
    "rank_key",
    "wave_times",
]

SM_COUNT_B200 = 148
"""int: SM count the wave estimates assume for CPU devices and for the named plans' logging."""

CUDNN_Q_TILE = 256
"""int: Query rows per CTA of cuDNN's sm100 flash fprop kernel."""

NAMED_SPECS = ("auto", "per_head", "whole", "first_small", "ends_small")
"""tuple[str, ...]: Plan specs ``HeadPlan.from_spec`` accepts by name; anything else is a comma list."""

_LOGGED_PLANS: set[tuple] = set()  # auto decisions already logged in this process


def device_sm_count(device=None) -> int:
    """Return the SM count of a CUDA device (cuDNN runs one CTA per SM); B200's for CPU devices."""
    if device is not None and torch.device(device).type != "cuda":
        return SM_COUNT_B200
    if not torch.cuda.is_available():
        return SM_COUNT_B200
    dev = torch.cuda.current_device() if device is None else device
    return torch.cuda.get_device_properties(dev).multi_processor_count


def named_groups(spec: str, H_local: int) -> tuple[int, ...]:
    """Return the groups of a named spec other than ``auto``.

    ``per_head`` = [1] * H_local; ``whole`` = [H_local] (the stock single call); ``first_small``
    = [1, H_local - 1] (attention starts after one head's transfer); ``ends_small`` =
    [1, H_local - 2, 1] (small first and last group: the last group's output transfer is the
    exposed tail).
    """
    if spec == "per_head":
        return (1,) * H_local
    if spec == "whole":
        return (H_local,)
    if spec == "first_small":
        return (1, H_local - 1) if H_local > 1 else (1,)
    if spec == "ends_small":
        return (1, H_local - 2, 1) if H_local >= 3 else (1,) * H_local
    raise ValueError(f"HeadPlan: {spec!r} is not a named plan {NAMED_SPECS[1:]}")


def parse_groups(spec: str, H_local: int) -> tuple[int, ...]:
    """Parse an explicit comma list such as ``1,4,5``; ``rest`` is allowed once (``1,rest,1``)."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"HeadPlan.from_spec: empty spec {spec!r}")
    if parts.count("rest") > 1:
        raise ValueError(f"HeadPlan.from_spec: 'rest' may appear once, got {spec!r}")
    known = sum(int(p) for p in parts if p != "rest")
    return tuple(H_local - known if p == "rest" else int(p) for p in parts)


def wave_times(
    groups: tuple[int, ...], S: int, B: int = 1, sms: int = SM_COUNT_B200, tile: int = CUDNN_Q_TILE
) -> int:
    """Return W of a plan: per group ``ceil(CTAs / sms)`` with ``CTAs = B * h_g * ceil(S / tile)``.

    cuDNN's sm100 flash fprop kernel launches one CTA per (query tile, head, batch element), one
    CTA per SM, and a partial wave costs a whole one.
    """
    q_tiles = math.ceil(S / tile)
    return sum(math.ceil(B * h * q_tiles / sms) for h in groups)


def auto_candidates(H_local: int) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Return the plans ``auto`` chooses from, ``whole`` first, deduplicated by groups.

    whole, per_head, first_small, ends_small, equal groups of g heads for every divisor g of
    H_local with 1 < g < H_local (``pairs``, ``halves``, ``groups_of_<g>``), ``first_<k>`` =
    [k, H_local - k] for 2 <= k <= H_local / 2 and ``ends_<k>`` = [k, H_local - 2k, k] for k >= 2
    with a non-empty middle group.
    """
    H = H_local
    cands: list[tuple[str, tuple[int, ...]]] = [
        (name, named_groups(name, H)) for name in NAMED_SPECS[1:]
    ]
    cands[0], cands[1] = cands[1], cands[0]  # whole first: its W is the lower bound of the rule
    for g in range(2, H):
        if H % g == 0:
            name = "halves" if 2 * g == H else "pairs" if g == 2 else f"groups_of_{g}"
            cands.append((name, (g,) * (H // g)))
    for k in range(2, H // 2 + 1):
        cands.append((f"first_{k}", (k, H - k)))
    for k in range(2, (H - 1) // 2 + 1):
        cands.append((f"ends_{k}", (k, H - 2 * k, k)))
    seen: set[tuple[int, ...]] = set()
    out = []
    for name, groups in cands:
        if groups not in seen:
            seen.add(groups)
            out.append((name, groups))
    return tuple(out)


def rank_key(W: int, groups: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Return the rule's sort key: wave-times, smallest first group, smallest last, fewest groups."""
    return (W, groups[0], groups[-1], len(groups))


@dataclass(frozen=True)
class AutoChoice:
    """Decision of ``auto``: name, groups, W, the W of every candidate (``table``) and the reason."""

    name: str
    groups: tuple[int, ...]
    W: int
    table: tuple[tuple[str, int], ...]
    reason: str

    def table_text(self) -> str:
        """Return every candidate's W as ``name W, name W, ...`` in candidate order."""
        return ", ".join(f"{name} {W}" for name, W in self.table)


def auto_choice(
    H_local: int, S: int, B: int = 1, sms: int | None = None, tile: int = CUDNN_Q_TILE
) -> AutoChoice:
    """Apply the wave-time rule.

    Smallest W over ``auto_candidates``; ties go to the smallest first group (attention starts
    after the shortest input transfer), then the smallest last group (the shortest exposed output
    transfer), then the fewest groups (fewest launches, waits and copies): ends_small > per_head >
    first_small > whole on a full tie. A sum of ceilings is at least the ceiling of the sum, so
    the whole-plan W is a lower bound: plans above it are refused and the result never costs more
    wave-times than the stock single call. Measured on B200 (4 ranks, fused producer): at 720p one
    head is exactly two waves and every plan ties; at 480p one head is 0.86 wave, so one-head
    groups leave 14 % of the SMs idle and per_head (W 10) ran 0.908x against 0.85-0.86x for the
    W-9 plans.
    """
    if H_local < 1 or S < 1 or B < 1:
        raise ValueError(
            f"HeadPlan auto: need H_local, S, B >= 1, got H_local={H_local} S={S} B={B}"
        )
    sms = device_sm_count() if sms is None else int(sms)
    if sms < 1:
        raise ValueError(f"HeadPlan auto: need sms >= 1, got {sms}")
    cands = auto_candidates(H_local)
    table = tuple((name, wave_times(groups, S, B, sms, tile)) for name, groups in cands)
    w_whole = table[0][1]
    best = min(range(len(cands)), key=lambda i: rank_key(table[i][1], cands[i][1]))
    if table[best][1] > w_whole:  # cannot happen (sum of ceilings >= ceiling of the sum)
        best = 0
    name, groups = cands[best]
    W = table[best][1]
    refused = ", ".join(f"{n} {w}" for n, w in table if w > w_whole)
    if best == 0:
        why = (
            f", every chunked plan costs more wave-times than one launch ({refused} > {w_whole})"
            if refused
            else ""
        )
        reason = f"whole: no chunking{why}"
    else:
        tail = f" (refused: {refused})" if refused else " (every plan ties)"
        reason = f"{name}: {len(groups)} groups {list(groups)} at W {W} = whole's {w_whole}{tail}"
    return AutoChoice(name, groups, W, table, reason)


@dataclass(frozen=True)
class HeadPlan:
    """Ordered head groups of one rank: ``groups[g]`` heads each, ``offsets[g]`` = first local head.

    ``groups`` empty means one head per group. ``sms`` is the SM count the wave estimates use
    (``auto``: the device's); ``note`` records how the groups were chosen (``auto``'s reason).
    """

    H_local: int
    groups: tuple[int, ...] = ()
    sms: int = field(default=SM_COUNT_B200, compare=False)
    note: str = field(default="", compare=False, repr=False)
    offsets: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.H_local <= 0:
            raise ValueError(f"HeadPlan: H_local must be positive, got {self.H_local}")
        if self.sms < 1:
            raise ValueError(f"HeadPlan: need sms >= 1, got {self.sms}")
        groups = tuple(int(g) for g in self.groups) or (1,) * self.H_local
        if any(g <= 0 for g in groups):
            raise ValueError(f"HeadPlan: every group needs at least one head, got {groups}")
        if sum(groups) != self.H_local:
            raise ValueError(
                f"HeadPlan: groups {groups} sum to {sum(groups)}, not H_local={self.H_local}"
            )
        offsets = tuple(sum(groups[:g]) for g in range(len(groups)))
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "offsets", offsets)

    @classmethod
    def from_spec(
        cls, spec: str, H_local: int, S: int | None = None, B: int = 1, sms: int | None = None
    ) -> HeadPlan:
        """Build a plan from ``auto``, a named spec or an explicit comma list.

        ``auto`` is the wave-time rule and needs the GLOBAL sequence length ``S`` (``P *
        S_local``) and ``B``; ``sms`` defaults to the current CUDA device's SM count. The named
        and explicit plans do not depend on S; their ``sms`` defaults to B200's 148 (their wave
        estimate is for logging).
        """
        spec = spec.strip()
        if spec == "auto":
            if S is None:
                raise ValueError(
                    "HeadPlan.from_spec: 'auto' needs the global sequence length S (the wave-time "
                    "rule's input); resolve the plan where S is known, e.g. at the first forward "
                    "from P * S_local"
                )
            sms = device_sm_count() if sms is None else int(sms)
            choice = auto_choice(H_local, S, B, sms)
            key = (H_local, S, B, sms)
            if key not in _LOGGED_PLANS:
                _LOGGED_PLANS.add(key)
                logger.info(
                    f"HeadPlan auto (H_local={H_local}, S={S}, B={B}, {sms} SMs): "
                    f"W {choice.table_text()} -> {choice.reason}"
                )
            return cls(H_local, choice.groups, sms, f"auto -> {choice.reason}")
        sms = SM_COUNT_B200 if sms is None else int(sms)
        groups = named_groups(spec, H_local) if spec in NAMED_SPECS else parse_groups(spec, H_local)
        return cls(H_local, groups, sms)

    @property
    def num_groups(self) -> int:
        """Return the number of head groups (one attention launch each)."""
        return len(self.groups)

    @property
    def uniform(self) -> bool:
        """Whether all groups have the same head count (one copy each for pack / assembly)."""
        return len(set(self.groups)) == 1

    def heads(self, g: int) -> range:
        """Return the local head indices of group g."""
        return range(self.offsets[g], self.offsets[g] + self.groups[g])

    def ctas(self, S: int, B: int = 1, tile: int = CUDNN_Q_TILE) -> tuple[int, ...]:
        """Return cuDNN CTAs per group for a global sequence length S: B * heads * ceil(S / tile)."""
        return tuple(B * h * math.ceil(S / tile) for h in self.groups)

    def waves(
        self, S: int, B: int = 1, sms: int | None = None, tile: int = CUDNN_Q_TILE
    ) -> tuple[float, ...]:
        """Return waves (CTAs / SMs) per group; the fractional part is that group's tail waste."""
        sms = self.sms if sms is None else sms
        return tuple(c / sms for c in self.ctas(S, B, tile))

    def wave_times(self, S: int, B: int = 1, tile: int = CUDNN_Q_TILE) -> int:
        """Return W of this plan at a global sequence length S (whole waves summed over groups)."""
        return wave_times(self.groups, S, B, self.sms, tile)

    def candidate_table(self, S: int, B: int = 1) -> str:
        """Return the W of every ``auto`` candidate for this H_local at S."""
        return auto_choice(self.H_local, S, B, self.sms).table_text()

    def describe(self, S: int | None = None, B: int = 1) -> str:
        """Return groups and offsets; with ``S`` the waves per group, W and every candidate's W."""
        text = f"HeadPlan(H_local={self.H_local}, groups={list(self.groups)}, offsets={list(self.offsets)})"
        if S is not None:
            waves = ", ".join(f"{w:.2f}" for w in self.waves(S, B))
            text += (
                f" waves@S={S},B={B}: [{waves}]; W {self.wave_times(S, B)} wave-times on {self.sms} SMs "
                f"(candidates: {self.candidate_table(S, B)})"
            )
        if self.note:
            text += f"; {self.note}"
        return text
