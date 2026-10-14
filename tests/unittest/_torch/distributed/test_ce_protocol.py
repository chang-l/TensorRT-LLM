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
"""CPU tests for the copy-engine protocol base (no GPU): the boundary bookkeeping and the signal
balance of a minimal two-channel protocol on a fake pool and a fake fabric (every rank in one
process; one flag bit per ``(channel, src -> dst)``, one op queue per stream, a scheduler that
fails where the devices would trap and where a resync collective would hang), after normal
boundaries and after a rank-symmetric raise at every point of the boundary; the one-sided and
asymmetric raises the base cannot repair (pinned as the trap or the unmatched collective); a
release that raises; ``signal_write``'s lost put; the join of exactly the forked copy streams;
the growth guard; the per-slot capture note; the resync latch; the generic arrival rule; and
the per-group state's probe fallback, setup failure and registry lifecycle on a one-rank gloo
group.
"""

import contextlib
from collections import Counter

import pytest
import torch
import torch.distributed as dist

from tensorrt_llm._torch.distributed import ce_protocol
from tensorrt_llm._torch.distributed.ce_protocol import (
    CeResyncError,
    CeState,
    CeTransport,
    CopyStreams,
)
from tensorrt_llm._torch.distributed.symm_mem_pool import DEFAULT_TIMEOUT_MS, PoolLayout

pytestmark = pytest.mark.cpu_only

CPU = torch.device("cpu")
IN, OUT = 0, 1  # the minimal protocol's two channels


# =============================================================================
# The fabric: flags, streams and events of every rank in one process
# =============================================================================


class _FakeStream:
    """One CUDA stream of one rank: a FIFO of ops the fabric executes in order."""

    def __init__(self, fabric: "_Fabric", rank: int, name: str, priority: int = 0):
        self.fabric, self.rank, self.name, self.priority = fabric, rank, name, priority
        self.ops: list[tuple] = []
        self.done = 0

    def wait_event(self, event: "_FakeEvent") -> None:
        self.ops.append(("event", event))

    def wait_stream(self, other: "_FakeStream") -> None:
        event = _FakeEvent(self.fabric)
        event.record(other)
        self.wait_event(event)

    def __repr__(self) -> str:
        return self.name


class _FakeEvent:
    """A CUDA event: 'everything enqueued on ``stream`` before the record has completed'."""

    def __init__(self, fabric: "_Fabric"):
        self.fabric = fabric
        self.stream: _FakeStream | None = None
        self.index = 0

    def record(self, stream: _FakeStream | None = None) -> None:
        self.stream = stream if stream is not None else self.fabric.current_stream()
        self.index = len(self.stream.ops)

    def ready(self) -> bool:
        return self.stream is not None and self.stream.done >= self.index

    def __repr__(self) -> str:
        return f"event({self.stream}@{self.index})"


class _Fabric:
    """Single-process model of what the devices do with the work the transports of ALL ranks
    enqueue: the signal pads (one bit per ``(channel, src -> dst)``, kept the way ``put_signal``
    / ``wait_signal`` keep them), the streams of every rank (one op queue each; events order
    queues), the capture flag and the resync collective. :meth:`run` executes the queues as the
    devices would -- a put is runnable only while its flag is clear, a wait only once it is set,
    a copy and a ``signal_write`` always (a write onto a set flag is a LOST put, counted) -- and
    fails where the real pool traps the device after its deadline: when no queue can make
    progress. :meth:`assert_collectives_matched` fails where the device would hang: a resync
    round that some ranks entered and others did not. Puts, waits and pushes are counted per
    ``(channel, src, dst)``.
    """

    def __init__(self, world_size: int):
        self.world_size = world_size
        self.rank = 0  # the rank whose transport code is running
        self.active: _FakeStream | None = None  # set by the torch.cuda.stream context
        self.streams: list[_FakeStream] = []
        self.main = [self._new_stream(r, f"rank{r}/current") for r in range(world_size)]
        self.flags: dict[tuple[int, int, int], int] = {}  # (channel, src, dst) -> 0 / 1
        self.puts: Counter = Counter()
        self.waits: Counter = Counter()
        self.pushes: Counter = Counter()  # (channel, src, dst) -> pushes the protocol issued
        self.copies: Counter = Counter()  # (channel, src, dst) -> copies the devices executed
        self.lost: Counter = Counter()  # (channel, src, dst) -> signal_write puts onto a set flag
        self.capturing = False
        self.transports: list[CeTransport] = []  # every rank's transport, for the resync gather
        self.resyncs = 0
        self.reported: dict[int, int] = {}  # rank -> the count it reported in this round
        self.counts_override: dict[int, int] = {}  # rank -> boundary count the gather reports

    def _new_stream(self, rank: int, name: str, priority: int = 0) -> _FakeStream:
        stream = _FakeStream(self, rank, name, priority)
        self.streams.append(stream)
        return stream

    # -- the torch.cuda surface the transport uses --

    def Stream(self, device=None, priority: int = 0) -> _FakeStream:  # noqa: N802
        n = sum(1 for s in self.streams if s.rank == self.rank) - 1
        return self._new_stream(self.rank, f"rank{self.rank}/side{n}", priority)

    def Event(self) -> _FakeEvent:  # noqa: N802
        return _FakeEvent(self)

    def current_stream(self) -> _FakeStream:
        return self.active if self.active is not None else self.main[self.rank]

    @contextlib.contextmanager
    def stream(self, stream: _FakeStream):
        prev, self.active = self.active, stream
        try:
            yield
        finally:
            self.active = prev

    def is_capturing(self) -> bool:
        return self.capturing

    def gather_boundary_counts(self, group, count: int) -> list[int]:
        """The resync collective. The ranks run one after another in this process, so a rank
        that already completed the boundary would report a count one higher than a rank that
        has not opened it yet; the real all-gather sees every rank's count at its own ``open``.
        Hence each rank's report is kept for the round (:func:`_boundary` starts one) and a rank
        that has not reported yet contributes the count it will report."""
        self.resyncs += 1
        self.reported[self.rank] = count
        return [
            self.counts_override.get(r, self.reported.get(r, t._boundaries))
            for r, t in enumerate(self.transports)
        ]

    def assert_collectives_matched(self) -> None:
        """A resync round is matched when every rank gathered or none did. The answer above is
        given to a lone rank too (the ranks run one after another here), so the matching is
        checked at the end of the round: on the device a rank gathering alone blocks in
        ``all_gather_object`` until the process group's timeout while its peers, which did not
        reset, go on and trap at the deadline on the flags it never put."""
        assert not self.reported or len(self.reported) == self.world_size, (
            f"unmatched resync collective: ranks {sorted(self.reported)} of {self.world_size} "
            "gathered; the device would hang them"
        )

    def install(self, monkeypatch) -> None:
        for name in ("Stream", "Event", "current_stream", "stream"):
            monkeypatch.setattr(torch.cuda, name, getattr(self, name))
        monkeypatch.setattr(ce_protocol, "_is_capturing", self.is_capturing)
        monkeypatch.setattr(ce_protocol, "_gather_boundary_counts", self.gather_boundary_counts)

    # -- the pads and the copy engines --

    def put(self, channel: int, src: int, dst: int) -> None:
        self.current_stream().ops.append(("put", channel, src, dst))

    def wait(self, channel: int, src: int, dst: int) -> None:
        self.current_stream().ops.append(("wait", channel, src, dst))

    def copy(self, channel: int, src: int, dst: int) -> None:
        self.current_stream().ops.append(("copy", channel, src, dst))

    def write(self, channel: int, src: int, dst: int, stream: _FakeStream | None = None) -> None:
        """``signal_write``: the flag set on ``stream`` (the current one by default), no
        back-pressure."""
        (stream if stream is not None else self.current_stream()).ops.append(
            ("write", channel, src, dst)
        )

    def _step(self, op: tuple) -> bool:
        if op[0] == "event":
            return op[1].ready()
        kind, channel, src, dst = op
        key = (channel, src, dst)
        if kind == "copy":
            self.copies[key] += 1
        elif kind == "put":
            if self.flags.get(key, 0):
                return False  # put_signal spins while the previous flag is unconsumed
            self.flags[key] = 1
            self.puts[key] += 1
        elif kind == "write":
            if self.flags.get(key, 0):
                self.lost[key] += 1  # signal_write has no back-pressure: the put is lost
            self.flags[key] = 1
            self.puts[key] += 1
        else:
            if not self.flags.get(key, 0):
                return False  # wait_signal spins until the flag is set
            self.flags[key] = 0
            self.waits[key] += 1
        return True

    def run(self) -> None:
        """Execute every queue to its end, or fail where the devices would trap."""
        while True:
            progressed = pending = False
            for stream in self.streams:
                while stream.done < len(stream.ops):
                    if not self._step(stream.ops[stream.done]):
                        pending = True
                        break
                    stream.done += 1
                    progressed = True
            if not pending:
                return
            if not progressed:
                heads = "; ".join(
                    f"{s}: {s.ops[s.done]}" for s in self.streams if s.done < len(s.ops)
                )
                raise AssertionError(
                    f"the devices would trap at the signal deadline; blocked: {heads}"
                )


class _FakePool:
    """``SymmMemPool`` stand-in over a :class:`_Fabric`: the pool's own layout arithmetic and
    grow-only reserve (refused under capture, as the real one), signals that enqueue into the
    fabric instead of launching kernels (with the real pool's channel check), and a release
    counter."""

    def __init__(self, fabric: _Fabric, rank: int, align: int = 256, channels: int = 2):
        self.fabric = fabric
        self.rank = rank
        self.world_size = fabric.world_size
        self.align = align
        self.channels = channels  # data channels the pad offers (the real pool's property)
        self.group = None  # the resync gather is the fabric's
        self.group_name = f"fake{fabric.world_size}"
        self.generation = 0
        self.releases = 0
        self._layout: PoolLayout | None = None

    @property
    def reserved(self) -> bool:
        return self._layout is not None

    @property
    def nbytes(self) -> int:
        return self._layout.nbytes if self._layout else 0

    @property
    def slots(self) -> int:
        return self._layout.slots if self._layout else 0

    @property
    def slot_nbytes(self) -> int:
        return self._layout.slot_nbytes if self._layout else 0

    @property
    def region_nbytes(self) -> int:
        return self._layout.region_nbytes if self._layout else 0

    def reserve(self, slot_nbytes: int, slots: int, min_channels: int) -> bool:
        if ce_protocol._is_capturing():
            raise RuntimeError("SymmMemPool.reserve under CUDA-graph capture")
        want = PoolLayout.for_request(slot_nbytes, slots, self.world_size, self.align)
        if self._layout is not None:
            if self._layout.covers(want):
                return False
            want = self._layout.merged(want)
        self._layout = want
        self.generation += 1
        for key in self.fabric.flags:  # the real pool zeroes this rank's pad on every allocation
            if key[2] == self.rank:
                self.fabric.flags[key] = 0
        return True

    def _check_channel(self, channel: int, who: str) -> None:
        if not 0 <= channel < self.channels:  # what the real pool's signal / wait raise
            raise ValueError(f"SymmMemPool.{who}: channel {channel} out of range")

    def signal(self, peer: int, channel: int) -> None:
        self._check_channel(channel, "signal")
        self.fabric.put(channel, self.rank, peer)

    def signal_write(self, peer: int, channel: int, stream: _FakeStream) -> None:
        self._check_channel(channel, "signal_write")
        self.fabric.write(channel, self.rank, peer, stream)

    def wait(self, src: int, channel: int) -> None:
        self._check_channel(channel, "wait")
        self.fabric.wait(channel, src, self.rank)

    def release(self) -> None:
        if self._layout is not None:
            self.releases += 1
        self._layout = None


# =============================================================================
# A minimal protocol on the base
# =============================================================================


class _Exchange(CeTransport):
    """The smallest protocol with the base's shape, in the Ulysses direction: push channel
    ``IN`` to ``me + 1, me + 2, ...`` (so it arrives from ``me - 1, me - 2, ...``), wait every
    ``IN`` flag, a consumer step, push channel ``OUT`` in the same order, wait every ``OUT`` flag,
    ``done``. Each push is one copy and the put on the destination's forked copy stream."""

    def __init__(self, pool, world_size, rank, device, *, slots=1, per_peer_streams=True):
        arrival = tuple((rank - i) % world_size for i in range(1, world_size))
        super().__init__(
            pool,
            world_size,
            rank,
            device,
            slots=slots,
            arrival_order=arrival,
            per_peer_streams=per_peer_streams,
        )
        self.push_order = tuple((rank + i) % world_size for i in range(1, world_size))

    def expected(self) -> list[tuple[int, int]]:
        return [(src, ch) for ch in (IN, OUT) for src in self.arrival_order]

    def push(self, slot: int, dst: int, channel: int) -> None:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        stream = self.streams.fork(dst, event)
        with torch.cuda.stream(stream):
            self.pool.fabric.copy(channel, self.rank, dst)
            self.mark_pushed(slot, dst, channel)
        self.pool.fabric.pushes[(channel, self.rank, dst)] += 1


def _transports(fabric: _Fabric, slots: int = 1, per_peer_streams: bool = True) -> list[_Exchange]:
    transports = []
    for r in range(fabric.world_size):
        fabric.rank = r
        transports.append(
            _Exchange(
                _FakePool(fabric, r),
                fabric.world_size,
                r,
                CPU,
                slots=slots,
                per_peer_streams=per_peer_streams,
            )
        )
    for r, transport in enumerate(transports):
        fabric.rank = r
        assert transport._reserve(1024, slots, 2) is True
    fabric.transports = transports
    return transports


def _raise_points(world_size: int) -> list:
    """Every point of the boundary a rank-symmetric raise can hit: before ``open``; inside the
    IN push loop after ``k`` pushes (``k = 0``: before any push); after every IN push, before
    the IN waits; inside the consumer (after the IN waits, before any OUT push); inside the OUT
    push loop after ``k`` pushes; after every OUT push, before the final waits; after the final
    waits, before ``done``."""
    return [
        "before_open",
        *(("in", k) for k in range(world_size - 1)),
        "before_in_wait",
        "in_consumer",
        *(("out", k) for k in range(world_size - 1)),
        "before_out_wait",
        "before_done",
    ]


def _raise_cases(world_sizes=(2, 3, 4, 8)) -> list:
    return [
        pytest.param(
            p, fail, id=f"P{p}-{fail if isinstance(fail, str) else fail[0] + str(fail[1])}"
        )
        for p in world_sizes
        for fail in _raise_points(p)
    ]


def _boundary(fabric: _Fabric, transports: list[_Exchange], slot: int, fail=None) -> None:
    """One boundary on every rank, enqueued in rank order as the op body does, then run on the
    fabric. ``fail`` injects the op's raise (:func:`_raise_points`) and its except path
    (``reset``): one point for every rank (rank-symmetric), or ``{rank: point}`` for the ranks
    that raise (one-sided / asymmetric; the other ranks complete the boundary)."""
    world_size = len(transports)
    fabric.reported = {}  # a new round of the resync collective, if any rank runs it
    for r, t in enumerate(transports):
        fabric.rank = r
        point = fail.get(r) if isinstance(fail, dict) else fail
        try:
            if point == "before_open":
                raise RuntimeError("injected before open")
            t.open(slot, t.expected())
            for i, dst in enumerate(t.push_order):
                if point == ("in", i):
                    raise RuntimeError(f"injected in the IN push loop after {i} pushes")
                t.push(slot, dst, IN)
            if point == "before_in_wait":
                raise RuntimeError("injected before the IN waits")
            for src in t.arrival_order:
                t.wait_ready(slot, src, IN)
            if point == "in_consumer":
                raise RuntimeError("injected inside the consumer")
            for i, dst in enumerate(t.push_order):
                if point == ("out", i):
                    raise RuntimeError(f"injected in the OUT push loop after {i} pushes")
                t.push(slot, dst, OUT)
            if point == "before_out_wait":
                raise RuntimeError("injected before the final waits")
            for src in t.arrival_order:
                t.wait_ready(slot, src, OUT)
            if point == "before_done":
                raise RuntimeError("injected before done")
            t.done(slot)
        except RuntimeError as e:
            assert point is not None and str(e).startswith("injected"), e
            t.reset()  # the op body's except path
    assert world_size == fabric.world_size
    fabric.assert_collectives_matched()
    fabric.run()


def _assert_balanced(fabric: _Fabric, transports: list[_Exchange]) -> None:
    """The invariant the base documents: after every completed boundary (``done`` or ``reset``)
    no slot is in flight, no copy stream has a pending op, and for every ``(channel, p -> d)``
    the flag is clear and ``puts = waits`` = the pushes ``p`` issued to ``d`` on that channel;
    every rank counts the same completed boundaries; no resync round was left unmatched and no
    ``signal_write`` put was lost."""
    fabric.assert_collectives_matched()
    assert not fabric.lost, f"signal_write puts onto a set flag (lost): {dict(fabric.lost)}"
    assert not any(any(t._in_flight) for t in transports)
    assert all(s.done == len(s.ops) for s in fabric.streams)
    assert len({t._boundaries for t in transports}) == 1
    world = fabric.world_size
    for ch in (IN, OUT):
        for p in range(world):
            for d in range(world):
                if p == d:
                    continue
                key = (ch, p, d)
                assert fabric.flags.get(key, 0) == 0, ("flag set", key)
                assert fabric.puts[key] == fabric.waits[key] == fabric.pushes[key], key
                assert fabric.copies[key] == fabric.pushes[key], key


# =============================================================================
# Signal balance: normal boundaries and recoveries
# =============================================================================


@pytest.mark.parametrize("per_peer_streams", [True, False], ids=["per-peer", "one-stream"])
@pytest.mark.parametrize("slots", [1, 2])
@pytest.mark.parametrize("world_size", [2, 3, 4, 8])
def test_boundaries_keep_the_signals_balanced(world_size, slots, per_peer_streams, monkeypatch):
    fabric = _Fabric(world_size)
    fabric.install(monkeypatch)
    transports = _transports(fabric, slots, per_peer_streams)
    assert all(len(t.streams) == (world_size - 1 if per_peer_streams else 1) for t in transports)
    for index in range(3 * slots):
        _boundary(fabric, transports, index % slots)
        _assert_balanced(fabric, transports)
    assert all(t._boundaries == 3 * slots for t in transports)
    assert fabric.resyncs == 0  # no reset, no collective
    # drain between forwards and a covered reserve touch no flag.
    for r, t in enumerate(transports):
        fabric.rank = r
        t.drain()
        assert t._reserve(1024, slots, 2) is False
    fabric.run()
    _assert_balanced(fabric, transports)


@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
@pytest.mark.parametrize("slots", [1, 2])
@pytest.mark.parametrize("world_size,fail", _raise_cases())
def test_reset_after_a_rank_symmetric_raise_keeps_the_signals_balanced(
    world_size, fail, slots, warm, monkeypatch
):
    """A rank-symmetric raise at every point of the boundary, on a transport's first boundary
    and after good ones, at both slot counts: ``reset`` completes the boundary like ``done``
    (the balance invariant holds right after it), no device would trap, the next eager open
    runs the resync collective exactly once per rank, and the following boundaries are plain
    ones."""
    fabric = _Fabric(world_size)
    fabric.install(monkeypatch)
    transports = _transports(fabric, slots)
    index = 0
    for _ in range(0 if not warm else 2 * slots):
        _boundary(fabric, transports, index % slots)
        _assert_balanced(fabric, transports)
        index += 1
    before = transports[0]._boundaries
    _boundary(fabric, transports, index % slots, fail=fail)
    _assert_balanced(fabric, transports)
    completed = fail != "before_open"  # nothing was in flight: reset() had nothing to release
    assert all(t._boundaries == before + int(completed) for t in transports)
    assert all(t._needs_resync for t in transports)
    index += 1
    for _ in range(2 * slots):
        _boundary(fabric, transports, index % slots)
        _assert_balanced(fabric, transports)
        index += 1
    assert fabric.resyncs == world_size  # one gather per rank, on the first open after the reset
    assert not any(t._needs_resync for t in transports)


@pytest.mark.parametrize(
    "pair",
    [(("in", 0), "before_out_wait"), ("before_done", ("out", 0)), ("in_consumer", "before_open")],
    ids=["in0+before_out_wait", "before_done+out0", "in_consumer+before_open"],
)
@pytest.mark.parametrize("slots", [1, 2])
@pytest.mark.parametrize("world_size", [2, 3, 4, 8])
def test_back_to_back_recoveries_keep_the_signals_balanced(world_size, slots, pair, monkeypatch):
    """Two rank-symmetric raises on every slot with no good boundary between them: the IDLE
    state ``reset`` leaves is the one ``done`` leaves, so the second recovery starts from it as
    a plain boundary would, and the boundaries after both are plain ones."""
    fabric = _Fabric(world_size)
    fabric.install(monkeypatch)
    transports = _transports(fabric, slots)
    index = 0
    for _ in range(slots):
        _boundary(fabric, transports, index % slots)
        index += 1
    for i in range(2 * slots):
        _boundary(fabric, transports, index % slots, fail=pair[i // slots])
        _assert_balanced(fabric, transports)
        index += 1
    resyncs = fabric.resyncs  # the first failed boundary's open after a reset did one gather
    for _ in range(2 * slots):
        _boundary(fabric, transports, index % slots)
        _assert_balanced(fabric, transports)
        index += 1
    assert fabric.resyncs == resyncs + world_size


def test_fabric_detects_the_trap():
    """The fabric fails where the pool traps: a second put onto a set flag, a wait with no put."""
    fabric = _Fabric(2)
    fabric.put(0, 0, 1)
    fabric.put(0, 0, 1)
    with pytest.raises(AssertionError, match=r"would trap .* \('put', 0, 0, 1\)"):
        fabric.run()
    fabric = _Fabric(2)
    fabric.wait(1, 1, 0)
    with pytest.raises(AssertionError, match=r"would trap .* \('wait', 1, 1, 0\)"):
        fabric.run()


def test_a_missing_wait_is_caught_by_the_fabric_on_the_next_boundary(monkeypatch):
    """Negative control: a protocol that forgets a flag in ``expected`` leaves it set, and the
    producer's next put on that pair blocks -- the fabric reports the trap instead of passing."""
    fabric = _Fabric(3)
    fabric.install(monkeypatch)
    transports = _transports(fabric)
    for index in range(2):
        for r, t in enumerate(transports):
            fabric.rank = r
            owed = t.expected()
            if r == 0:
                owed = owed[1:]  # rank 0 never waits (arrival_order[0], IN)
            t.open(0, owed)
            for dst in t.push_order:
                t.push(0, dst, IN)
            for src in t.arrival_order:
                t.wait_ready(0, src, IN)
            for dst in t.push_order:
                t.push(0, dst, OUT)
            t.done(0)
        if index == 0:
            fabric.run()
            assert fabric.flags[(IN, transports[0].arrival_order[0], 0)] == 1
        else:
            with pytest.raises(AssertionError, match="would trap"):
                fabric.run()


# =============================================================================
# What the base cannot repair, pinned: one-sided and asymmetric raises; a release that raises;
# the lost put of signal_write; a channel the pool lacks
# =============================================================================


def test_a_one_sided_benign_raise_leaves_the_resync_collective_unmatched(recorder, monkeypatch):
    """Rank 0 alone raises after its last wait (every flag balanced, every count equal): the
    boundary itself passes the balance check and only rank 0 latches, so at the next boundary
    its resync gather has no partner. The fabric pins that as the unmatched collective (on the
    device rank 0 blocks in ``all_gather_object`` until the process group's timeout, announced
    by the warning) rather than passing; a change that drops the latch or gathers on another
    predicate fails here."""
    fabric = _Fabric(3)
    fabric.install(monkeypatch)
    transports = _transports(fabric)
    _boundary(fabric, transports, 0)
    _boundary(fabric, transports, 0, fail={0: "before_done"})
    _assert_balanced(fabric, transports)
    assert [t._needs_resync for t in transports] == [True, False, False]
    with pytest.raises(AssertionError, match=r"unmatched resync collective: ranks \[0\] of 3"):
        _boundary(fabric, transports, 0)
    assert fabric.resyncs == 1 and fabric.reported == {0: 2}
    assert len(recorder.warnings) == 1
    assert recorder.warnings[0].startswith("_Exchange[fake3]: resync after a failed boundary")


def test_a_one_sided_raise_before_the_last_put_traps_the_peers(monkeypatch):
    """Rank 0 alone raises inside the consumer (its IN flags consumed, its OUT pushes never
    issued): ``reset`` repairs rank 0's own books, but the peers wait rank 0's OUT flags, which
    no retry can put -- the fabric reports the trap (the pool's deadline on the device)."""
    fabric = _Fabric(3)
    fabric.install(monkeypatch)
    transports = _transports(fabric)
    _boundary(fabric, transports, 0)
    with pytest.raises(AssertionError, match=r"would trap .* \('wait', 1, 0, [12]\)"):
        _boundary(fabric, transports, 0, fail={0: "in_consumer"})


@pytest.mark.parametrize(
    "points,outcome",
    [((2, 1, 2, 1), "repaired"), ((1, 1, None), "trap")],
    ids=["P4-2121-repaired", "P3-11-trap"],
)
def test_asymmetric_raise_points_in_the_in_push_loop(points, outcome, monkeypatch):
    """Ranks raising after different numbers of IN pushes (``None``: the rank completes). The
    arrival rule is exact under rank symmetry only: at P = 4 the pattern 2, 1, 2, 1 happens to
    leave every rank's trimmed waits equal to the flags it received (all four latch, the
    resync is matched and the next boundaries are plain); at P = 3 the rank that completes
    waits an IN flag its peer never pushed, and the fabric reports the trap."""
    fabric = _Fabric(len(points))
    fabric.install(monkeypatch)
    transports = _transports(fabric)
    _boundary(fabric, transports, 0)
    fail = {r: ("in", k) for r, k in enumerate(points) if k is not None}
    if outcome == "trap":
        with pytest.raises(AssertionError, match="would trap"):
            _boundary(fabric, transports, 0, fail=fail)
        return
    _boundary(fabric, transports, 0, fail=fail)
    _assert_balanced(fabric, transports)
    assert all(t._needs_resync for t in transports)
    _boundary(fabric, transports, 0)
    _assert_balanced(fabric, transports)
    assert fabric.resyncs == len(points) and not any(t._needs_resync for t in transports)


def test_open_and_mark_pushed_refuse_a_channel_the_pool_lacks(monkeypatch):
    """A channel beyond the pool's is refused where the protocol declares it -- at ``open``,
    before anything is in flight, and at ``mark_pushed`` before the put and the record -- not
    by the pool's ``wait`` inside the release, where a raise would strand the slot."""
    fabric = _Fabric(3)
    fabric.install(monkeypatch)
    transports = _transports(fabric)
    t = transports[0]
    fabric.rank = 0
    assert t.pool.channels == 2
    with pytest.raises(ValueError, match=r"open: channel 7 out of range \(the pool offers 2"):
        t.open(0, [(1, IN), (1, 7)])
    assert not t._in_flight[0] and t._pending[0] == set()
    t.open(0, t.expected())
    with pytest.raises(ValueError, match=r"mark_pushed: channel 2 out of range"):
        t.mark_pushed(0, 1, 2)
    assert t._pushed_to[0] == set() and all(s.ops == [] for s in fabric.streams)
    t.reset()
    fabric.run()


@pytest.mark.parametrize("where", ["wait", "join"])
def test_a_release_that_raises_leaves_the_slot_idle_and_the_pool_releasable(where, monkeypatch):
    """A raise inside the release (the pool refusing a wait; the capture refusing a join)
    propagates, but the slot is IDLE afterwards with its bookkeeping dropped: ``reset`` does not
    raise again and waits no pair twice, the boundary is not counted (the resync latch covers a
    rank that fell behind), and ``close`` releases the pool instead of the transport staying
    wedged ('before done' on every open, 'in flight' on every close) until process exit."""
    fabric = _Fabric(3)
    fabric.install(monkeypatch)
    transports = _transports(fabric)
    t = transports[0]
    fabric.rank = 0
    t.open(0, t.expected())  # owes (1, IN), (1, OUT), (2, IN), (2, OUT) in the release's order
    for dst in t.push_order:
        t.push(0, dst, IN)
    if where == "wait":
        real_wait = t.pool.wait

        def failing_wait(src, channel):
            if (src, channel) == (2, IN):
                raise RuntimeError("injected: the pool refused the wait")
            real_wait(src, channel)

        t.pool.wait = failing_wait
        with pytest.raises(RuntimeError, match="injected: the pool refused the wait"):
            t.done(0)
        del t.pool.wait
        waited = [op for op in fabric.main[0].ops if op[0] == "wait"]
        assert waited == [("wait", IN, 1, 0), ("wait", OUT, 1, 0)]  # enqueued before the raise
    else:

        def failing_join():
            raise RuntimeError("injected: wait_stream refused (capture isolation)")

        t.streams.join_forked = failing_join
        with pytest.raises(RuntimeError, match="injected: wait_stream refused"):
            t.done(0)
        del t.streams.join_forked
        assert len([op for op in fabric.main[0].ops if op[0] == "wait"]) == 4
    assert not t._in_flight[0] and t._pending[0] == set() and t._pushed_to[0] == set()
    assert t._boundaries == 0 and not t._needs_resync  # done() raised: the op body resets next
    before = len(fabric.main[0].ops)
    t.reset()  # nothing in flight: no wait enqueued (none twice); the resync latched
    assert not [op for op in fabric.main[0].ops[before:] if op[0] == "wait"]
    assert t._needs_resync and t.streams._forked == set()
    t.close()
    assert t.pool.releases == 1 and not t.pool.reserved


class _WriteExchange(_Exchange):
    """The Ulysses form of the put: ``signal_write`` on the copy stream (no back-pressure)."""

    def _put(self, dst: int, channel: int) -> None:
        self.pool.signal_write(dst, channel, torch.cuda.current_stream())


def test_signal_write_puts_balance_and_a_lost_put_is_counted(monkeypatch):
    """A protocol whose ``_put`` is the pool's ``signal_write`` (the Ulysses override) passes
    the same invariant over plain boundaries, and a protocol bug that puts one pair twice in a
    boundary is not a trap at the put (the fabric's model of ``put_signal``) but a LOST put:
    counted per pair and failed by the balance check (on the device the pair's next wait spins
    to the deadline one boundary later)."""
    fabric = _Fabric(3)
    fabric.install(monkeypatch)
    transports = []
    for r in range(3):
        fabric.rank = r
        transports.append(_WriteExchange(_FakePool(fabric, r), 3, r, CPU))
        transports[-1]._reserve(1024, 1, 2)
    fabric.transports = transports
    for _ in range(2):
        _boundary(fabric, transports, 0)
        _assert_balanced(fabric, transports)
    assert sum(fabric.puts.values()) == 2 * 2 * 6 and not fabric.lost
    for r, t in enumerate(transports):  # the bug: rank 0 pushes IN to rank 1 twice
        fabric.rank = r
        t.open(0, t.expected())
        for dst in t.push_order:
            t.push(0, dst, IN)
        if r == 0:
            t.push(0, 1, IN)
        for src in t.arrival_order:
            t.wait_ready(0, src, IN)
        for dst in t.push_order:
            t.push(0, dst, OUT)
        for src in t.arrival_order:
            t.wait_ready(0, src, OUT)
        t.done(0)
    fabric.run()  # no trap: the second write found the flag set and was lost
    assert fabric.lost == {(IN, 0, 1): 1}
    assert fabric.puts[(IN, 0, 1)] == 4 and fabric.waits[(IN, 0, 1)] == 3
    with pytest.raises(AssertionError, match="lost"):
        _assert_balanced(fabric, transports)


# =============================================================================
# Streams: fork, join exactly the forked ones, join all only when eager
# =============================================================================


def test_join_covers_exactly_the_forked_streams(monkeypatch):
    fabric = _Fabric(4)
    fabric.install(monkeypatch)
    fabric.rank = 1
    streams = CopyStreams(4, 1, CPU, per_peer=True, priority=-1)
    assert streams.peers == (0, 2, 3) and len(streams) == 3
    assert all(streams.stream(p).priority == -1 for p in streams.peers)
    with pytest.raises(ValueError, match="not a peer"):
        streams.stream(1)
    with pytest.raises(ValueError, match="not a peer"):
        streams.fork(4, _FakeEvent(fabric))
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream())
    forked = streams.fork(2, event)
    assert forked is streams.stream(2) and forked.ops == [("event", event)]
    assert streams.fork(2, event) is forked  # forking twice records two waits, joins once
    streams.join_forked()
    joins = [op[1].stream for op in fabric.main[1].ops if op[0] == "event"]
    assert joins == [streams.stream(2)]  # one edge, to the forked stream only
    assert streams.stream(0).ops == [] and streams.stream(3).ops == []
    streams.join_forked()  # nothing forked since: no edge
    assert len(fabric.main[1].ops) == 1
    # join_all_eager joins every stream, forked or not, and is refused under capture.
    streams.join_all_eager()
    joins = [op[1].stream for op in fabric.main[1].ops[1:] if op[0] == "event"]
    assert joins == [streams.stream(p) for p in streams.peers]
    fabric.capturing = True
    streams.fork(3, event)
    with pytest.raises(RuntimeError, match="join_all_eager under CUDA-graph capture"):
        streams.join_all_eager()
    streams.join_forked()  # the forked stream can be joined into the capture
    assert fabric.main[1].ops[-1][1].stream is streams.stream(3)
    fabric.run()
    # The single-stream form carries every destination on one stream.
    fabric.rank = 0
    one = CopyStreams(4, 0, CPU, per_peer=False)
    assert len(one) == 1 and len({one.stream(p) for p in one.peers}) == 1
    with pytest.raises(ValueError, match="world_size >= 2"):
        CopyStreams(1, 0, CPU)
    with pytest.raises(ValueError, match="0 <= rank < world_size"):
        CopyStreams(2, 2, CPU)


def test_a_partial_destination_set_joins_only_its_streams(monkeypatch):
    """A protocol that pushes to a subset of its peers joins only those streams in ``done``
    (under capture a join of an un-forked stream would be a cross-capture dependency), and
    ``reset`` joins all of them when eager."""
    fabric = _Fabric(4)
    fabric.install(monkeypatch)
    transports = _transports(fabric)
    for r, t in enumerate(transports):  # ring: push IN to me + 1 only, expect it from me - 1
        fabric.rank = r
        ahead, behind = (r + 1) % 4, (r - 1) % 4
        t.open(0, [(behind, IN)])
        t.push(0, ahead, IN)
        t.wait_ready(0, behind, IN)
        t.done(0)
        joins = [op[1].stream for op in fabric.main[r].ops if op[0] == "event"]
        assert joins == [t.streams.stream(ahead)]
    fabric.run()
    _assert_balanced(fabric, transports)
    for r, t in enumerate(transports):
        fabric.rank = r
        ahead, behind = (r + 1) % 4, (r - 1) % 4
        t.open(0, [(behind, IN)])
        t.push(0, ahead, IN)
        t.reset()  # the release joins the forked stream, then every stream (eager)
        joins = [op[1].stream for op in fabric.main[r].ops if op[0] == "event"]
        every = [t.streams.stream(p) for p in t.streams.peers]
        assert joins[1:] == [t.streams.stream(ahead)] + every
    fabric.run()
    _assert_balanced(fabric, transports)


# =============================================================================
# Reserve: in flight, the scoped growth guard, view cache
# =============================================================================


def test_reserve_rules(monkeypatch):
    fabric = _Fabric(2)
    fabric.install(monkeypatch)
    t = _Exchange(_FakePool(fabric, 0), 2, 0, CPU)
    assert not t._covers(1024, 1)  # never reserved
    assert t._reserve(1024, 1, 2) is True and t.pool.generation == 1
    made: list[int] = []

    def make():
        made.append(len(made))
        return torch.zeros(1)

    assert t.view((0, "x"), make) is t.view((0, "x"), make) and made == [0]
    assert t._reserve(512, 1, 1) is False and made == [0]  # covered: no-op, views kept
    assert t._covers(1024, 1) and not t._covers(1025, 1) and not t._covers(1024, 2)
    assert t._reserve(4096, 1, 1) is True and t.pool.generation == 2
    t.view((0, "x"), make)
    assert made == [0, 1]  # the reallocation dropped the cached views
    t.view((1, "y"), make)
    t._drop_views(0)
    assert t.view((1, "y"), make) is not None and made == [0, 1, 2]  # slot 1's view survived
    t.view((0, "x"), make)
    assert made == [0, 1, 2, 3]
    # A boundary in flight refuses any reserve.
    t.open(0, t.expected())
    with pytest.raises(RuntimeError, match="in flight"):
        t._reserve(4096, 1, 1)
    t.reset()
    fabric.run()
    # The pool refuses under capture (the fake mirrors the real pool); the transport adds nothing.
    fabric.capturing = True
    with pytest.raises(RuntimeError, match="under CUDA-graph capture"):
        t._reserve(8192, 1, 1)
    fabric.capturing = False


def test_growth_guard_is_scoped_to_growth_after_a_captured_boundary(monkeypatch):
    fabric = _Fabric(2)
    fabric.install(monkeypatch)
    transports = _transports(fabric)  # 1024 B per slot, 1 slot
    _boundary(fabric, transports, 0)  # eager first
    fabric.capturing = True
    _boundary(fabric, transports, 0)  # captured and completed: done() promotes the latch
    fabric.capturing = False
    assert all(t._captured and not any(t._capturing) for t in transports)
    for r, t in enumerate(transports):
        fabric.rank = r
        assert t._reserve(1024, 1, 2) is False  # covered: allowed after the capture
        assert t._reserve(512, 1, 2) is False
        with pytest.raises(RuntimeError, match="would grow after a CUDA graph"):
            t._reserve(4096, 1, 2)  # bigger regions
        with pytest.raises(RuntimeError, match="would grow after a CUDA graph"):
            t._reserve(1024, 2, 2)  # more slots
        assert t.pool.generation == 1  # nothing was freed
    # A never-reserved pool's first reservation frees nothing: allowed after captures.
    fabric.rank = 0
    fresh = _Exchange(_FakePool(fabric, 0), 2, 0, CPU)
    fresh._captured = True
    assert fresh._reserve(4096, 2, 2) is True
    assert fresh._reserve(2048, 2, 2) is False  # covered
    with pytest.raises(RuntimeError, match="would grow"):
        fresh._reserve(8192, 2, 2)  # from now on the guard applies: the pool is reserved


# =============================================================================
# Capture latch: noted at open, promoted by done, dropped by reset
# =============================================================================


def test_capture_latch_is_promoted_by_done_and_dropped_by_reset(monkeypatch):
    fabric = _Fabric(3)
    fabric.install(monkeypatch)
    transports = _transports(fabric)
    _boundary(fabric, transports, 0)
    fabric.capturing = True
    # A boundary that raises under capture (its capture is discarded) leaves no mark, and
    # reset() under capture joins only the forked streams (join_all_eager is not called).
    _boundary(fabric, transports, 0, fail=("out", 1))
    _assert_balanced(fabric, transports)
    assert all(not t._captured and not any(t._capturing) and t._needs_resync for t in transports)
    # A latched transport refuses open() under capture, leaving the slot idle.
    fabric.rank = 0
    with pytest.raises(RuntimeError, match="run one eager call before capturing"):
        transports[0].open(0, transports[0].expected())
    assert not transports[0]._in_flight[0] and fabric.resyncs == 0
    fabric.capturing = False
    _boundary(fabric, transports, 0)  # eager: the resync runs and the latch clears
    assert fabric.resyncs == 3 and not any(t._needs_resync for t in transports)
    assert all(not t._captured for t in transports)
    fabric.capturing = True
    for r, t in enumerate(transports):
        fabric.rank = r
        t.open(0, t.expected())
        assert t._capturing == [True] and not t._captured  # noted, not yet promoted
    for r, t in enumerate(transports):
        fabric.rank = r
        for dst in t.push_order:
            t.push(0, dst, IN)
        for src in t.arrival_order:
            t.wait_ready(0, src, IN)
        for dst in t.push_order:
            t.push(0, dst, OUT)
        t.done(0)
        assert t._captured and t._capturing == [False]  # promoted by the completed boundary
    fabric.run()
    _assert_balanced(fabric, transports)
    fabric.capturing = False
    _boundary(fabric, transports, 0)
    assert all(t._captured for t in transports)  # stays: a graph holds the addresses


def test_capture_note_is_per_slot(monkeypatch):
    """With two slots in flight, an eager ``open`` on the second slot does not erase the note a
    captured boundary made on the first: its ``done`` still promotes the transport to captured
    (so a later growing reserve is refused while a graph holds the pool's buffers), and
    ``reset`` clears the note of every slot."""
    fabric = _Fabric(2)
    fabric.install(monkeypatch)
    transports = _transports(fabric, slots=2)
    for r, t in enumerate(transports):
        fabric.rank = r
        fabric.capturing = True
        t.open(0, t.expected())
        fabric.capturing = False
        t.open(1, t.expected())
        assert t._capturing == [True, False] and not t._captured
    for slot in (0, 1):
        for r, t in enumerate(transports):
            fabric.rank = r
            for dst in t.push_order:
                t.push(slot, dst, IN)
            for src in t.arrival_order:
                t.wait_ready(slot, src, IN)
            for dst in t.push_order:
                t.push(slot, dst, OUT)
            for src in t.arrival_order:
                t.wait_ready(slot, src, OUT)
            t.done(slot)
            assert t._captured and t._capturing == [False, False]
    fabric.run()
    _assert_balanced(fabric, transports)
    for r, t in enumerate(transports):
        fabric.rank = r
        fabric.capturing = True
        t.open(1, t.expected())
        fabric.capturing = False
        t.reset()
        assert t._capturing == [False, False] and t._captured
    fabric.run()


# =============================================================================
# Resync latch
# =============================================================================


def test_resync_latch_agrees_once_and_refuses_a_disagreement(recorder, monkeypatch):
    fabric = _Fabric(3)
    fabric.install(monkeypatch)
    transports = _transports(fabric)
    _boundary(fabric, transports, 0)
    assert fabric.resyncs == 0 and recorder.warnings == []
    _boundary(fabric, transports, 0, fail="in_consumer")
    assert all(t._needs_resync for t in transports)
    _boundary(fabric, transports, 0)
    assert fabric.resyncs == 3 and not any(t._needs_resync for t in transports)
    # Every gather is announced first: after a one-sided raise the log line is what names the
    # rank blocked in it (the peers' trap names a push / consume imbalance).
    assert len(recorder.warnings) == 3
    assert all(
        w.startswith("_Exchange[fake3]: resync after a failed boundary (this rank completed 2)")
        for w in recorder.warnings
    )
    _boundary(fabric, transports, 0)
    assert fabric.resyncs == 3  # one collective per recovery, nothing on the hot path
    # A disagreement raises naming the ranks and leaves the slot idle and the latch set.
    _boundary(fabric, transports, 0, fail="before_open")  # a raise before open latches too
    assert all(t._needs_resync for t in transports)
    n = transports[0]._boundaries
    assert n == 4  # one good, one recovered, two good; the raise before open completed none
    fabric.counts_override = {1: 99}
    fabric.rank = 0
    with pytest.raises(CeResyncError, match=rf"rank 0: {n}, rank 1: 99, rank 2: {n}"):
        transports[0].open(0, transports[0].expected())
    assert not transports[0]._in_flight[0] and transports[0]._needs_resync
    assert fabric.resyncs == 4
    fabric.counts_override = {}
    _boundary(fabric, transports, 0)
    assert fabric.resyncs == 7 and not any(t._needs_resync for t in transports)
    _assert_balanced(fabric, transports)


# =============================================================================
# The generic arrival rule
# =============================================================================


@pytest.mark.parametrize("world_size", [2, 3, 4, 8])
def test_arrived_mirrors_the_push_order_per_channel(world_size, monkeypatch):
    """For every rank and both push directions (Ulysses ``me + 1, ...`` arriving from
    ``me - 1, ...``; the TP pair's ``me - 1, ...`` arriving from ``me + 1, ...``), after ``m``
    pushes of one channel and ``n`` of another the flags that reached this rank are the first
    ``m`` / ``n`` sources of ``arrival_order``: the same set the senders' pushes produce under
    rank symmetry (checked against the senders' push prefixes)."""
    fabric = _Fabric(world_size)
    fabric.install(monkeypatch)
    for direction in (1, -1):
        push = {
            r: tuple((r + direction * i) % world_size for i in range(1, world_size))
            for r in range(world_size)
        }
        arrival = {
            r: tuple((r - direction * i) % world_size for i in range(1, world_size))
            for r in range(world_size)
        }
        for r in range(world_size):
            fabric.rank = r
            t = CeTransport(
                _FakePool(fabric, r), world_size, r, CPU, slots=1, arrival_order=arrival[r]
            )
            t._reserve(256, 1, 2)
            t.open(0, [(s, ch) for ch in (IN, OUT) for s in arrival[r]])
            for m in range(world_size):
                for n in range(m + 1):
                    t._pushed_to[0] = {(d, IN) for d in push[r][:m]} | {
                        (d, OUT) for d in push[r][:n]
                    }
                    # The senders that pushed to r within their first m (n) pushes of the channel.
                    senders_in = {(s, IN) for s in range(world_size) if s != r and r in push[s][:m]}
                    senders_out = {
                        (s, OUT) for s in range(world_size) if s != r and r in push[s][:n]
                    }
                    assert t._arrived(0) == senders_in | senders_out, (direction, r, m, n)
                    assert t._arrived(0) == {(s, IN) for s in arrival[r][:m]} | {
                        (s, OUT) for s in arrival[r][:n]
                    }
            t._pushed_to[0] = set()
            t.reset()
    fabric.run()


# =============================================================================
# Transport API checks
# =============================================================================


def test_transport_validation_and_refusals(monkeypatch):
    fabric = _Fabric(3)
    fabric.install(monkeypatch)
    pool = _FakePool(fabric, 0)
    with pytest.raises(ValueError, match="needs a SymmMemPool"):
        CeTransport(None, 3, 0, CPU, slots=1, arrival_order=(1, 2))
    with pytest.raises(ValueError, match="world_size >= 2"):
        CeTransport(pool, 1, 0, CPU, slots=1, arrival_order=())
    with pytest.raises(ValueError, match="0 <= rank < world_size"):
        CeTransport(pool, 3, 3, CPU, slots=1, arrival_order=(1, 2))
    with pytest.raises(ValueError, match="slots >= 1"):
        CeTransport(pool, 3, 0, CPU, slots=0, arrival_order=(1, 2))
    for bad in [(1,), (1, 1), (0, 1), (1, 2, 2), (2, 1, 0)]:
        with pytest.raises(ValueError, match="every peer .* exactly once"):
            CeTransport(pool, 3, 0, CPU, slots=1, arrival_order=bad)
    t = _Exchange(pool, 3, 0, CPU, slots=2)
    assert t.arrival_order == (2, 1) and t.push_order == (1, 2) and t.slots == 2
    assert t._reserve(1024, 2, 2)
    with pytest.raises(ValueError, match="slot 2 out of range"):
        t.open(2, [])
    with pytest.raises(ValueError, match="not a peer"):
        t.open(0, [(0, IN)])  # a wait on oneself
    with pytest.raises(ValueError, match="not a peer"):
        t.open(0, [(3, IN)])
    for op in (
        lambda: t.mark_pushed(0, 1, IN),
        lambda: t.wait_ready(0, 1, IN),
        lambda: t._consume(0, [(1, IN)]),
        lambda: t.done(0),
    ):
        with pytest.raises(RuntimeError, match=r"needs an open\(0\) in flight"):
            op()
    t.open(0, t.expected())
    with pytest.raises(RuntimeError, match=r"open\(slot=0\) before done\(0\)"):
        t.open(0, t.expected())
    with pytest.raises(ValueError, match="not a peer"):
        t.mark_pushed(0, 0, IN)
    with pytest.raises(RuntimeError, match="drain with a boundary in flight on slot 0"):
        t.drain()
    with pytest.raises(RuntimeError, match="drain with a boundary in flight"):
        t.close()
    # Slot 1 opens independently while slot 0 is in flight.
    t.open(1, t.expected())
    assert t._in_flight == [True, True]
    t.reset()
    fabric.run()
    assert t._in_flight == [False, False] and t._boundaries == 2
    t.close()
    t.close()  # idempotent
    assert pool.releases == 1 and not pool.reserved and t._views == {}


def test_wait_ready_waits_once_and_consume_books_without_a_wait(monkeypatch):
    fabric = _Fabric(4)
    fabric.install(monkeypatch)
    transports = _transports(fabric)
    t = transports[0]
    fabric.rank = 0
    t.open(0, t.expected())
    t.wait_ready(0, 3, IN)
    t.wait_ready(0, 3, IN)  # already consumed: no second wait
    assert fabric.main[0].ops == [("wait", IN, 3, 0)]
    owed = t._consume(0, [(2, IN), (3, IN), (1, IN)])
    assert owed == [(2, IN), (1, IN)]  # the given order, the owed pairs only
    assert fabric.main[0].ops == [("wait", IN, 3, 0)]  # booked, nothing enqueued
    for src, ch in owed:  # the protocol's own wait over the booked pairs (one launch on GPU)
        fabric.wait(ch, src, 0)
    assert t._pending[0] == {(s, OUT) for s in t.arrival_order}
    t.wait_ready(0, 3, OUT)
    t.done(0)  # the release waits the two owed OUT flags in sorted order, then joins nothing
    assert fabric.main[0].ops == [
        ("wait", IN, 3, 0),
        ("wait", IN, 2, 0),
        ("wait", IN, 1, 0),
        ("wait", OUT, 3, 0),
        ("wait", OUT, 1, 0),
        ("wait", OUT, 2, 0),
    ]
    assert t._pending[0] == set() and not t._in_flight[0]
    # The peers put every flag rank 0 waited and expect nothing from rank 0 (which pushed
    # nothing), so the fabric completes without a trap and every pair is balanced.
    for r in (1, 2, 3):
        fabric.rank = r
        peer = transports[r]
        peer.open(0, [(s, ch) for ch in (IN, OUT) for s in peer.arrival_order if s != 0])
        for ch in (IN, OUT):
            for dst in peer.push_order:
                peer.push(0, dst, ch)
        peer.done(0)
    fabric.run()
    _assert_balanced(fabric, transports)


# =============================================================================
# CeState on a one-rank gloo group: probe fallback, one warning, registry lifecycle
# =============================================================================


class _Recorder:
    """Stands in for ``tensorrt_llm.logger.logger``."""

    def __init__(self):
        self.warnings: list[str] = []

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(str(msg))

    def info(self, msg, *args, **kwargs):
        pass

    debug = info


class _ToyState(CeState):
    def _make_transport(self) -> None:
        fabric = self._fabric  # set by the test before the probe
        self.pool = _FakePool(fabric, self.rank)
        self.transport = _Exchange(self.pool, max(self.world_size, 2), self.rank, CPU)
        self.transport._reserve(256, 1, 2)


class _OtherState(CeState):
    def _make_transport(self) -> None:
        self.pool = self.made = _FakePool(self._fabric, self.rank)
        self.pool.reserve(256, 1, 2)
        self.transport = None  # a subclass that forgets the transport


class _FailingState(CeState):
    def _make_transport(self) -> None:
        self.pool = self.made = _FakePool(self._fabric, self.rank)
        self.pool.reserve(256, 1, 2)
        raise RuntimeError("injected: setup failed after the pool was reserved")


@pytest.fixture
def gloo_world():
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(ce_protocol, "logger", rec)
    return rec


def test_state_lifecycle_on_gloo(gloo_world, recorder, monkeypatch):
    monkeypatch.setattr(ce_protocol, "_is_capturing", lambda: False)  # no CUDA query on CPU
    group = gloo_world
    name = group.group_name
    with pytest.raises(ValueError, match="process group; got None"):
        _ToyState(None, name)
    with pytest.raises(ValueError, match="timeout_ms >= 0"):
        _ToyState(group, name, timeout_ms=-1)
    state = _ToyState(group, name)
    assert state.world_size == 1 and state.rank == 0
    assert state.timeout_ms == DEFAULT_TIMEOUT_MS == 600_000  # the pool's default
    assert _ToyState(group, name, timeout_ms=60_000).timeout_ms == 60_000  # a protocol's own
    assert not state.effective and state.reason == "not probed yet"
    assert state.pool is None and state.transport is None
    state.require_open()
    # Registry: setdefault semantics, one dict per subclass.
    assert _ToyState.register(state) is state
    assert _ToyState.register(_ToyState(group, name)) is state
    assert _ToyState.get(name) is state
    with pytest.raises(KeyError):
        _OtherState.get(name)
    with pytest.raises(KeyError):
        CeState.get(name)
    assert _ToyState._registry is not _OtherState._registry is not CeState._registry
    # The probe declines on gloo (not a CUDA NCCL group), with ONE warning naming the fallback.
    assert state.probe_once(min_channels=2, what="toy exchange") is False
    assert not state.effective and "needs >= 2 ranks" in state.reason
    assert len(recorder.warnings) == 1
    assert recorder.warnings[0].startswith("toy exchange unavailable (")
    assert "keeping the NCCL path" in recorder.warnings[0] and state.reason in recorder.warnings[0]
    assert state.probe_once(min_channels=2, what="toy exchange") is False
    state.disable("later")
    assert len(recorder.warnings) == 1 and state.reason == "later"  # warned once per state
    # release closes and unregisters; close is idempotent; a closed state refuses.
    _ToyState.release(name)
    with pytest.raises(KeyError):
        _ToyState.get(name)
    assert not state.effective and state.reason == "closed"
    state.close()
    _ToyState.release(name)  # no-op without a state
    with pytest.raises(RuntimeError, match="is closed"):
        state.require_open()
    with pytest.raises(RuntimeError, match="is closed"):
        state.probe_once(min_channels=2, what="toy exchange")
    # A fresh state for the same name probes again (and warns again, once).
    fresh = _ToyState.register(_ToyState(group, name))
    assert fresh is not state and _ToyState.get(name) is fresh
    assert fresh.probe_once(min_channels=2, what="toy exchange") is False
    assert len(recorder.warnings) == 2
    fresh.close()
    with pytest.raises(KeyError):
        _ToyState.get(name)
    # Closing a state that another one replaced in the registry leaves the replacement alone.
    a = _ToyState.register(_ToyState(group, name))
    b = _ToyState(group, name)
    a.close()
    assert _ToyState.register(b) is b
    a.close()
    assert _ToyState.get(name) is b
    b.close()


def test_probe_once_refuses_capture_and_builds_the_transport_on_success(
    gloo_world, recorder, monkeypatch
):
    group = gloo_world
    name = group.group_name
    fabric = _Fabric(2)
    fabric.install(monkeypatch)
    probes: list[int] = []

    def fake_probe(group, device, *, min_channels):
        probes.append(min_channels)
        return True, ""

    monkeypatch.setattr(ce_protocol, "probe_symmetric_memory", fake_probe)
    # Never under capture.
    fabric.capturing = True
    state = _ToyState(group, name)
    with pytest.raises(RuntimeError, match="under CUDA-graph capture; run an eager warm-up"):
        state.probe_once(min_channels=2, what="toy exchange")
    assert not state._probed and probes == []
    fabric.capturing = False
    # The base cannot build a transport; a subclass that leaves none is refused.
    with pytest.raises(NotImplementedError, match="_make_transport"):
        CeState(group, name).probe_once(min_channels=2, what="base")
    other = _OtherState(group, name)
    other._fabric = fabric
    with pytest.raises(RuntimeError, match="left no transport"):
        other.probe_once(min_channels=2, what="other")
    assert other.made.releases == 1 and other.pool is None  # the pool it reserved is released
    assert other._probed and not other.effective
    assert other.reason.startswith("other setup failed: RuntimeError: _OtherState._make_transport")
    # Success: pool and transport built once, the verdict stored, no warning.
    state._fabric = fabric
    assert state.probe_once(min_channels=4, what="toy exchange") is True
    assert state.effective and state.reason == "" and probes == [2, 2, 4]
    assert isinstance(state.transport, _Exchange) and state.transport.pool is state.pool
    assert state.probe_once(min_channels=4, what="toy exchange") is True and probes == [2, 2, 4]
    assert recorder.warnings == []
    # disable(): off the protocol with the one warning; the pool stays until close().
    state.disable("ranks disagree after a failed boundary")
    assert not state.effective and state.reason == "ranks disagree after a failed boundary"
    assert len(recorder.warnings) == 1 and "keeping the NCCL path" in recorder.warnings[0]
    assert "ranks disagree" in recorder.warnings[0] and state.pool.reserved
    state.disable("again")
    assert len(recorder.warnings) == 1
    assert state.probe_once(min_channels=4, what="toy exchange") is False  # the stored verdict
    # close() closes the transport (releasing the pool) and forgets both.
    pool = state.pool
    _ToyState.register(state)
    state.close()
    assert pool.releases == 1 and not pool.reserved
    assert state.pool is None and state.transport is None and state.reason == "closed"
    with pytest.raises(KeyError):
        _ToyState.get(name)
    state.close()
    assert pool.releases == 1


def test_probe_once_releases_the_pool_when_make_transport_raises(gloo_world, recorder, monkeypatch):
    """A subclass that reserves its pool and then raises in ``_make_transport``: the verdict is
    stored as failed (no second collective probe, no second allocation on a retry), the pool is
    released at once rather than staying allocated (and registered as a live pad) until process
    exit, and the exception propagates unmasked; ``close`` afterwards releases nothing twice."""
    group = gloo_world
    fabric = _Fabric(2)
    fabric.install(monkeypatch)
    probes: list[int] = []

    def fake_probe(group, device, *, min_channels):
        probes.append(min_channels)
        return True, ""

    monkeypatch.setattr(ce_protocol, "probe_symmetric_memory", fake_probe)
    state = _FailingState(group, group.group_name)
    state._fabric = fabric
    with pytest.raises(RuntimeError, match="injected: setup failed"):
        state.probe_once(min_channels=2, what="toy exchange")
    assert state._probed and not state.effective
    assert state.reason == (
        "toy exchange setup failed: RuntimeError: injected: setup failed after the pool was "
        "reserved"
    )
    assert state.pool is None and state.transport is None
    assert state.made.releases == 1 and not state.made.reserved
    assert recorder.warnings == []  # loud, not a fallback warning
    assert state.probe_once(min_channels=2, what="toy exchange") is False and probes == [2]
    state.close()
    assert state.made.releases == 1 and state.reason == "closed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
