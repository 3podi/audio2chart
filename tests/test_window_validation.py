import unittest

from modules.utils_train import usable_window_count


class UsableWindowCountTest(unittest.TestCase):
    GRID = 20
    WINDOW = 30.0

    def test_a_clean_chart_has_every_window_usable(self):
        times = [i * 0.5 for i in range(240)]          # 120 s at 500 ms spacing
        self.assertEqual(4, usable_window_count(times, self.GRID, self.WINDOW))

    def test_one_tight_run_only_costs_its_own_window(self):
        times = [i * 0.5 for i in range(240)]
        times += [65.000, 65.010]                       # 10 ms apart, third window
        times.sort()
        self.assertEqual(3, usable_window_count(times, self.GRID, self.WINDOW))

    def test_events_exactly_one_grid_step_apart_are_allowed(self):
        # discretize_time raises on `< grid`, not `<= grid`; quantized charts land
        # exactly on the grid constantly and must not be discarded.
        self.assertEqual(1, usable_window_count([1.0, 1.02], self.GRID, self.WINDOW))
        self.assertEqual(0, usable_window_count([1.0, 1.019], self.GRID, self.WINDOW))

    def test_a_window_with_fewer_than_two_events_is_usable(self):
        self.assertEqual(1, usable_window_count([5.0], self.GRID, self.WINDOW))

    def test_an_entirely_tight_chart_has_no_usable_window(self):
        times = [i * 0.005 for i in range(200)]        # 5 ms apart throughout
        self.assertEqual(0, usable_window_count(times, self.GRID, self.WINDOW))

    def test_empty_chart(self):
        self.assertEqual(0, usable_window_count([], self.GRID, self.WINDOW))


if __name__ == "__main__":
    unittest.main()
