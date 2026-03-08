from __future__ import annotations

import unittest

from training.warmup_runtime import allocate_warmup_games, resolve_warmup_levels


class TestTrainingWarmupRuntime(unittest.TestCase):
    def test_resolve_warmup_levels_dedupes_and_preserves_order(self) -> None:
        levels = resolve_warmup_levels(
            raw_levels_csv="hard, apex,hard,sentinel",
            fallback_level="normal",
        )
        self.assertListEqual(levels, ["hard", "apex", "sentinel"])

    def test_resolve_warmup_levels_falls_back_when_csv_is_empty(self) -> None:
        levels = resolve_warmup_levels(
            raw_levels_csv="  ",
            fallback_level="sentinel",
        )
        self.assertListEqual(levels, ["sentinel"])

    def test_allocate_warmup_games_distributes_evenly(self) -> None:
        allocation = allocate_warmup_games(
            total_games=10,
            levels=["hard", "apex", "sentinel"],
        )
        self.assertEqual(sum(games for _, games in allocation), 10)
        self.assertListEqual(
            [level for level, _ in allocation],
            ["hard", "apex", "sentinel"],
        )
        # 10 games over 3 levels -> 4/3/3
        self.assertListEqual(
            [games for _, games in allocation],
            [4, 3, 3],
        )


if __name__ == "__main__":
    unittest.main()
