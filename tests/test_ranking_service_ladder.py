from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.modules.ranking.service import RankingService


class TestRankingServiceLadder(unittest.TestCase):
    def test_league_snapshot_default_rating(self) -> None:
        league, division, lp = RankingService.build_league_snapshot(1200.0)
        self.assertEqual(league, "Protocol")
        self.assertEqual(division, "III")
        self.assertEqual(lp, 0)

    def test_league_snapshot_progression(self) -> None:
        league, division, lp = RankingService.build_league_snapshot(1342.7)
        self.assertEqual(league, "Protocol")
        self.assertEqual(division, "II")
        self.assertEqual(lp, 42)

        league2, division2, lp2 = RankingService.build_league_snapshot(1725.2)
        self.assertEqual(league2, "Kernel")
        self.assertEqual(division2, "I")
        self.assertEqual(lp2, 25)

    def test_league_snapshot_top_cap(self) -> None:
        league, division, lp = RankingService.build_league_snapshot(9999.0)
        self.assertEqual(league, "Root")
        self.assertEqual(division, "I")
        self.assertEqual(lp, 99)

    def test_delta_clamp(self) -> None:
        self.assertEqual(RankingService._clamp_delta(100.0), 30.0)
        self.assertEqual(RankingService._clamp_delta(-100.0), -30.0)
        self.assertEqual(RankingService._clamp_delta(12.5), 12.5)

    def test_next_major_promo_name(self) -> None:
        self.assertEqual(
            RankingService.next_major_promo_name("Protocol", "I"),
            "Kernel Glitch",
        )
        self.assertEqual(
            RankingService.next_major_promo_name("Kernel", "I"),
            "Root Access",
        )
        self.assertIsNone(RankingService.next_major_promo_name("Protocol", "II"))
        self.assertIsNone(RankingService.next_major_promo_name("Root", "I"))

    def test_ladder_transition_subdivision_promotion(self) -> None:
        transition = RankingService.build_ladder_transition(1299.0, 1305.0)
        self.assertTrue(transition.promoted)
        self.assertFalse(transition.demoted)
        self.assertEqual(transition.before.league, "Protocol")
        self.assertEqual(transition.before.division, "III")
        self.assertEqual(transition.after.league, "Protocol")
        self.assertEqual(transition.after.division, "II")
        self.assertIsNone(transition.major_promo)

    def test_ladder_transition_major_promo_protocol_to_kernel(self) -> None:
        transition = RankingService.build_ladder_transition(1499.0, 1503.0)
        self.assertTrue(transition.promoted)
        self.assertFalse(transition.demoted)
        self.assertEqual(transition.before.league, "Protocol")
        self.assertEqual(transition.before.division, "I")
        self.assertEqual(transition.after.league, "Kernel")
        self.assertEqual(transition.after.division, "III")
        self.assertEqual(transition.major_promo, "Kernel Glitch")

    def test_ladder_transition_major_promo_kernel_to_root(self) -> None:
        transition = RankingService.build_ladder_transition(1799.0, 1801.0)
        self.assertTrue(transition.promoted)
        self.assertFalse(transition.demoted)
        self.assertEqual(transition.before.league, "Kernel")
        self.assertEqual(transition.before.division, "I")
        self.assertEqual(transition.after.league, "Root")
        self.assertEqual(transition.after.division, "III")
        self.assertEqual(transition.major_promo, "Root Access")

    def test_ladder_transition_demotion(self) -> None:
        transition = RankingService.build_ladder_transition(1505.0, 1498.0)
        self.assertFalse(transition.promoted)
        self.assertTrue(transition.demoted)
        self.assertEqual(transition.before.league, "Kernel")
        self.assertEqual(transition.before.division, "III")
        self.assertEqual(transition.after.league, "Protocol")
        self.assertEqual(transition.after.division, "I")
        self.assertIsNone(transition.major_promo)


if __name__ == "__main__":
    unittest.main()
