import unittest
import numpy as np
from src.edge.tracker import MetricKalmanFilter, MetricTrack


class TestMetricTracker(unittest.TestCase):
    def test_kalman_filter_constant_velocity(self):
        initial_pos = np.array([10.0, 20.0])
        kf = MetricKalmanFilter(initial_pos)

        # Simulate object moving with vx=2.0, vy=1.0 for 15 steps (1.5 seconds, dt=0.1)
        for step in range(1, 16):
            meas_pos = initial_pos + np.array([2.0 * step * 0.1, 1.0 * step * 0.1])
            kf.predict(dt=0.1)
            kf.update(meas_pos)

        # After 15 steps, filter tracks position and positive velocity
        self.assertAlmostEqual(kf.state[0], 13.0, delta=0.20)
        self.assertAlmostEqual(kf.state[1], 21.5, delta=0.20)
        self.assertGreater(kf.state[2], 1.0)  # Estimated vx positive and increasing toward 2.0

    def test_track_confirmation_and_history(self):
        track = MetricTrack(track_id=101, class_id=0, initial_pos=np.array([0.0, 0.0]))
        self.assertFalse(track.confirmed)
        self.assertEqual(len(track.history), 1)

        # 2 more updates to trigger confirmation threshold (hits >= 3)
        track.step_predict(0.1)
        track.step_update(np.array([0.1, 0.0]))
        self.assertFalse(track.confirmed)

        track.step_predict(0.1)
        track.step_update(np.array([0.2, 0.0]))
        self.assertTrue(track.confirmed)
        self.assertEqual(len(track.history), 3)


if __name__ == "__main__":
    unittest.main()
