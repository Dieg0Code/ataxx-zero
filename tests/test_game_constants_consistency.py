from __future__ import annotations

import unittest

from game.constants import PLAYER_1, PLAYER_2, WIN_P1, WIN_P2


class TestGameConstantsConsistency(unittest.TestCase):
    def test_winner_constants_match_player_encoding(self) -> None:
        self.assertEqual(WIN_P1, PLAYER_1)
        self.assertEqual(WIN_P2, PLAYER_2)


if __name__ == "__main__":
    unittest.main()
