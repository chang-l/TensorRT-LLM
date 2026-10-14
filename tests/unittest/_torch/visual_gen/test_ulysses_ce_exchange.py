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
"""CPU tests of the head-chunked Ulysses exchange (no GPU).

``TestHeadPlan``: plan offsets, specs and validation, and the wave-time rule (``auto``) at the
Wan2.2-A14B 720p/81f and 480p/81f geometries for 2 / 4 / 8 ranks, its tie-break, lower bound and
SM-count parameter. The rule is evaluated for B200's SM count whatever device runs the test. The
protocol, workspaces, state and custom op of the module are covered by the tests that arrive with
them.

Run with:
    pytest tests/unittest/_torch/visual_gen/test_ulysses_ce_exchange.py -v
"""

import math

import pytest

from tensorrt_llm._torch.visual_gen.attention.backends.ulysses_ce_exchange import (
    CUDNN_Q_TILE,
    SM_COUNT_B200,
    HeadPlan,
    auto_candidates,
    auto_choice,
    device_sm_count,
    rank_key,
    wave_times,
)

pytestmark = pytest.mark.cpu_only

S720, S480 = 75_600, 32_760  # Wan2.2 720p/81f and 480p/81f global sequence lengths (40 heads x 128)
SMS = 148  # the rule is evaluated for B200's SM count, whatever GPU runs this test


# =============================================================================
# Head plan (CPU)
# =============================================================================


def _table(choice, *names) -> tuple:
    t = dict(choice.table)
    return tuple(t.get(n) for n in names)


class TestHeadPlan:
    def test_default_is_per_head(self):
        p = HeadPlan(10)
        assert p.groups == (1,) * 10
        assert p.offsets == tuple(range(10))
        assert p.num_groups == 10 and p.uniform
        assert list(p.heads(3)) == [3]
        assert p.sms == SM_COUNT_B200 and p.note == ""

    def test_explicit_groups_and_offsets(self):
        p = HeadPlan(10, (1, 4, 5))
        assert p.groups == (1, 4, 5)
        assert p.offsets == (0, 1, 5)
        assert not p.uniform
        assert list(p.heads(1)) == [1, 2, 3, 4]
        assert list(p.heads(2)) == [5, 6, 7, 8, 9]

    @pytest.mark.parametrize(
        "spec,H_local,groups",
        [
            ("per_head", 5, (1, 1, 1, 1, 1)),
            ("whole", 7, (7,)),
            ("first_small", 10, (1, 9)),
            ("first_small", 1, (1,)),
            ("ends_small", 10, (1, 8, 1)),
            ("ends_small", 2, (1, 1)),
            ("ends_small", 1, (1,)),
            ("1,4,5", 10, (1, 4, 5)),
            (" 2, 2 ,1", 5, (2, 2, 1)),
            ("1,rest", 6, (1, 5)),
            ("rest,1", 3, (2, 1)),
            ("1,rest,1", 7, (1, 5, 1)),
        ],
    )
    def test_from_spec(self, spec, H_local, groups):
        p = HeadPlan.from_spec(spec, H_local)
        assert p.groups == groups
        assert sum(p.groups) == H_local
        assert p.offsets == tuple(sum(groups[:g]) for g in range(len(groups)))
        assert p.note == "" and p.sms == SM_COUNT_B200
        # The named and explicit plans do not depend on S: passing it changes only the SM count.
        assert HeadPlan.from_spec(spec, H_local, S=S480, sms=132).groups == groups
        assert HeadPlan.from_spec(spec, H_local, S=S480, sms=132).sms == 132

    @pytest.mark.parametrize("bad", [(1, 4), (0, 10), (11,), (-1, 11)])
    def test_invalid_groups(self, bad):
        with pytest.raises(ValueError):
            HeadPlan(10, bad)

    def test_invalid_specs(self):
        with pytest.raises(ValueError):
            HeadPlan.from_spec("1,2", 10)
        with pytest.raises(ValueError):
            HeadPlan.from_spec("rest,rest", 10)
        with pytest.raises(ValueError):
            HeadPlan.from_spec("", 10)
        with pytest.raises(ValueError):
            HeadPlan(0)
        with pytest.raises(ValueError):
            HeadPlan(10, (1, 9), sms=0)

    def test_wave_estimate_wan_u4(self):
        # Wan 720p/81f: S = 75,600, 10 local heads at U4; one head = 296 CTAs = 2 waves on 148 SMs.
        p = HeadPlan(10)
        assert p.ctas(S720) == (296,) * 10
        assert p.waves(S720) == (2.0,) * 10
        q = HeadPlan.from_spec("ends_small", 10)
        assert q.ctas(S720, B=2) == (592, 8 * 592, 592)
        assert math.isclose(q.waves(S720)[1], 8 * 296 / SM_COUNT_B200)
        assert math.ceil(S720 / CUDNN_Q_TILE) == 296
        text = q.describe(S720)
        assert "groups=[1, 8, 1]" in text and "waves@S=75600" in text
        assert (
            "W 20 wave-times on 148 SMs (candidates: whole 20, per_head 20, first_small 20, ends_small 20"
            in text
        )
        assert q.wave_times(S720) == 20 and q.wave_times(S720, B=2) == 40
        assert q.waves(S720, sms=74) == (4.0, 32.0, 4.0)

    def test_wave_table_720p(self):
        # 296 CTAs per head = exactly 2.0 waves: every plan ties -> ends_small at every U.
        for H_local, tie in ((20, 40), (10, 20), (5, 10)):
            c = auto_choice(H_local, S720, sms=SMS)
            assert all(W == tie for _, W in c.table), c.table
            assert c.name == "ends_small" and c.groups == (1, H_local - 2, 1) and c.W == tie
            assert "every plan ties" in c.reason
        c4 = auto_choice(10, S720, sms=SMS)
        assert (
            _table(c4, "whole", "per_head", "first_small", "ends_small", "pairs", "halves")
            == (20,) * 6
        )
        assert (
            _table(auto_choice(5, S720, sms=SMS), "whole", "per_head", "first_small", "ends_small")
            == (10,) * 4
        )
        p = HeadPlan.from_spec("auto", 10, S=S720, sms=SMS)
        assert (
            p.groups == (1, 8, 1)
            and p.num_groups == 3
            and p.wave_times(S720) == 20
            and p.sms == SMS
        )
        assert p.note.startswith(
            "auto -> ends_small: 3 groups [1, 8, 1] at W 20 = whole's 20 (every plan ties)"
        )
        assert HeadPlan.from_spec("auto", 5, S=S720, sms=SMS).groups == (1, 3, 1)  # U8
        assert HeadPlan.from_spec("auto", 20, S=S720, sms=SMS).groups == (1, 18, 1)  # U2

    def test_wave_table_480p(self):
        # 128 CTAs per head = 0.86 wave: one-head groups leave 14 % of the SMs idle, so per_head
        # pays an extra wave-time at U2 / U4 and is refused; the plans that tie the whole-plan W
        # keep the small first and last groups.
        c4 = auto_choice(10, S480, sms=SMS)
        assert _table(c4, "whole", "per_head", "first_small", "ends_small", "pairs") == (
            9,
            10,
            9,
            9,
            10,
        )
        assert c4.name == "ends_small" and c4.groups == (1, 8, 1) and c4.W == 9
        assert c4.reason.startswith(
            "ends_small: 3 groups [1, 8, 1] at W 9 = whole's 9 (refused: per_head 10, pairs 10"
        )
        c8 = auto_choice(5, S480, sms=SMS)
        assert _table(c8, "whole", "per_head", "first_small", "ends_small") == (5, 5, 5, 5)
        assert c8.name == "ends_small" and c8.groups == (1, 3, 1) and "every plan ties" in c8.reason
        c2 = auto_choice(20, S480, sms=SMS)
        assert _table(c2, "whole", "per_head", "first_small", "ends_small", "pairs", "halves") == (
            18,
            20,
            18,
            18,
            20,
            18,
        )
        assert c2.name == "ends_small" and c2.groups == (1, 18, 1)
        for H_local, groups in ((20, (1, 18, 1)), (10, (1, 8, 1)), (5, (1, 3, 1))):
            p = HeadPlan.from_spec("auto", H_local, S=S480, sms=SMS)
            assert p.groups == groups and p.note.startswith("auto -> ends_small")
        # The named specs stay selectable, S or not.
        q = HeadPlan.from_spec("per_head", 10, S=S480, sms=SMS)
        assert q.groups == (1,) * 10 and q.wave_times(S480) == 10 and q.note == ""
        assert HeadPlan.from_spec("whole", 10).wave_times(S480) == 9
        assert HeadPlan.from_spec("first_small", 10).wave_times(S480) == 9
        assert HeadPlan.from_spec("2,2,2,2,2", 10).wave_times(S480) == 10

    def test_tie_break_order(self):
        # Exact-wave heads (148 q-tiles x 1 head = 1.0 wave): every plan ties at W = H_local. The
        # tie goes to the smallest first group, then the smallest last group, then the fewest
        # groups: ends_small > per_head > first_small > whole.
        S = SMS * CUDNN_Q_TILE
        for H_local in (4, 5, 7, 10, 20):
            c = auto_choice(H_local, S, sms=SMS)
            assert (
                all(W == H_local for _, W in c.table)
                and c.name == "ends_small"
                and "every plan ties" in c.reason
            )
            named = dict(auto_candidates(H_local))
            order = sorted(
                ("ends_small", "per_head", "first_small", "whole"),
                key=lambda n: rank_key(H_local, named[n]),
            )
            assert order == ["ends_small", "per_head", "first_small", "whole"]
        assert auto_choice(3, S, sms=SMS).groups == (
            1,
            1,
            1,
        )  # ends_small == per_head (deduplicated)
        assert auto_choice(2, S, sms=SMS).groups == (1, 1)  # ends_small == first_small == per_head
        assert (
            auto_choice(1, S, sms=SMS).groups == (1,)
            and auto_choice(1, S, sms=SMS).reason == "whole: no chunking"
        )
        ends, per_head, first, whole = (1, 8, 1), (1,) * 10, (1, 9), (10,)
        assert rank_key(9, ends) < rank_key(9, per_head) < rank_key(9, first) < rank_key(9, whole)
        assert rank_key(9, first) < rank_key(
            9, (2, 2, 2, 2, 2)
        )  # the first group decides before the last

    def test_rule_lower_bound_and_candidates(self):
        # W is a sum of per-group ceilings: never below the whole-plan W; auto always reaches that
        # W (the stock single call's cost) and, among the plans that reach it, picks the smallest
        # rank_key.
        for H_local in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 40):
            cands = auto_candidates(H_local)
            assert cands[0][0] == "whole" and cands[0][1] == (H_local,)
            assert len({groups for _, groups in cands}) == len(cands)  # deduplicated
            assert all(sum(groups) == H_local for _, groups in cands)
            for S in (300, 4_095, 8_190, 9_450, 16_380, 18_900, S480, 37_800, S720, 151_200):
                for B in (1, 2):
                    c = auto_choice(H_local, S, B, sms=SMS)
                    w_whole = c.table[0][1]
                    assert all(W >= w_whole for _, W in c.table) and c.W == w_whole
                    tied = [
                        groups
                        for (_, groups), (_, W) in zip(cands, c.table, strict=True)
                        if W == w_whole
                    ]
                    assert rank_key(c.W, c.groups) == min(rank_key(w_whole, g) for g in tied)
                    assert c.W <= HeadPlan.from_spec("whole", H_local).wave_times(S, B)
                    p = HeadPlan.from_spec("auto", H_local, S=S, B=B, sms=SMS)
                    assert p.groups == c.groups and p.wave_times(S, B) == c.W
        names = [name for name, _ in auto_candidates(10)]
        assert names == [
            "whole",
            "per_head",
            "first_small",
            "ends_small",
            "pairs",
            "halves",
            "first_2",
            "first_3",
            "first_4",
            "ends_2",
            "ends_3",
            "ends_4",
        ]
        assert dict(auto_candidates(10))["ends_2"] == (2, 6, 2) and dict(auto_candidates(10))[
            "first_4"
        ] == (4, 6)
        assert [name for name, _ in auto_candidates(20)][:8] == [
            "whole",
            "per_head",
            "first_small",
            "ends_small",
            "pairs",
            "groups_of_4",
            "groups_of_5",
            "halves",
        ]
        assert [name for name, _ in auto_candidates(2)] == ["whole", "per_head"]
        assert [name for name, _ in auto_candidates(1)] == ["whole"]
        # Tiny sequences (300 rows x 1 head = 2 CTAs): every group costs a whole wave-time -> whole.
        c = auto_choice(10, 300, sms=SMS)
        assert c.groups == (10,) and c.reason.startswith(
            "whole: no chunking, every chunked plan costs more wave-times"
        )
        assert "per_head 10" in c.reason and "> 1" in c.reason
        # Half-wave heads (74 q-tiles x 1 head = 0.5 wave): [2, 6, 2] and pairs tie the whole plan
        # at 5; the three-group plan wins on fewer launches at equal first and last groups.
        c = auto_choice(10, 74 * CUDNN_Q_TILE, sms=SMS)
        assert _table(c, "whole", "per_head", "first_small", "ends_small", "pairs", "ends_2") == (
            5,
            10,
            6,
            6,
            5,
            5,
        )
        assert c.name == "ends_2" and c.groups == (2, 6, 2)
        assert (
            wave_times((2, 6, 2), 74 * CUDNN_Q_TILE, sms=SMS) == 5
            and wave_times((10,), 300, sms=SMS) == 1
        )

    def test_auto_needs_S_and_rejects_bad_geometry(self):
        with pytest.raises(ValueError, match="needs the global sequence length S"):
            HeadPlan.from_spec("auto", 10)
        with pytest.raises(ValueError):
            auto_choice(10, S480, sms=0)
        with pytest.raises(ValueError):
            auto_choice(0, S480, sms=SMS)
        with pytest.raises(ValueError):
            auto_choice(10, 0, sms=SMS)

    def test_sm_count_parameter(self):
        # 132 SMs (H100): one 720p head is 296 / 132 = 2.24 waves -> per_head 10 x 3 = 30, whole
        # ceil(2960 / 132) = 23, first_small 3 + 21 = 24, ends_small 3 + 18 + 3 = 24; [2, 8] reaches
        # 23 (5 + 18) and starts earliest -> first_2.
        c = auto_choice(10, S720, sms=132)
        assert _table(
            c, "whole", "per_head", "first_small", "ends_small", "pairs", "first_2", "ends_3"
        ) == (
            23,
            30,
            24,
            24,
            25,
            23,
            23,
        )
        assert (
            c.name == "first_2"
            and c.groups == (2, 8)
            and "refused: per_head 30, first_small 24" in c.reason
        )
        p = HeadPlan.from_spec("auto", 10, S=S720, sms=132)
        assert p.sms == 132 and p.groups == (2, 8) and p.wave_times(S720) == 23
        assert p.waves(S720) == pytest.approx((592 / 132, 2368 / 132))
        assert (
            "W 23 wave-times on 132 SMs" in p.describe(S720) and "auto -> first_2" in p.describe()
        )
        assert HeadPlan.from_spec("auto", 10, S=S720, sms=148).groups == (1, 8, 1)
        assert device_sm_count("cpu") == SM_COUNT_B200  # CPU transports evaluate the rule for B200

    def test_describe(self):
        p = HeadPlan.from_spec("auto", 10, S=S480, sms=SMS)
        text = p.describe(S480)
        head = "HeadPlan(H_local=10, groups=[1, 8, 1], offsets=[0, 1, 9])"
        assert text.startswith(
            head + " waves@S=32760,B=1: [0.86, 6.92, 0.86]; W 9 wave-times on 148 SMs (candidates: "
        )
        assert (
            "(candidates: whole 9, per_head 10, first_small 9, ends_small 9, pairs 10, halves 10, first_2 9"
            in text
        )
        assert (
            "; auto -> ends_small: 3 groups [1, 8, 1] at W 9 = whole's 9 (refused: per_head 10, pairs 10"
            in text
        )
        assert p.describe() == head + "; " + p.note
        q = HeadPlan.from_spec("1,rest", 6)
        assert q.describe() == "HeadPlan(H_local=6, groups=[1, 5], offsets=[0, 1])"
        # B = 2 doubles the CTAs (256 per head = 1.73 waves); (2, 2, 2) and (3, 3) are pairs / halves.
        assert q.describe(S480, B=2) == (
            "HeadPlan(H_local=6, groups=[1, 5], offsets=[0, 1]) waves@S=32760,B=2: [1.73, 8.65]; "
            "W 11 wave-times on 148 SMs (candidates: whole 11, per_head 12, first_small 11, "
            "ends_small 11, pairs 12, halves 12, first_2 11)"
        )
