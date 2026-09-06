import numpy as np
from typing import List, Dict, Any, Optional


class MetricKalmanFilter:
    """
    8-State Extended Kalman Filter operating directly in metric ground coordinates.
    State vector: [x, y, vx, vy, ax, ay, theta, omega]
    """
    def __init__(self, initial_pos: np.ndarray, initial_theta: float = 0.0):
        self.state = np.zeros(8, dtype=np.float64)
        self.state[0:2] = initial_pos
        self.state[6] = initial_theta

        self.P = np.eye(8, dtype=np.float64) * 0.1
        self.P[2:4, 2:4] *= 1.0  # Velocity uncertainty
        self.P[4:6, 4:6] *= 2.0  # Acceleration uncertainty

    def predict(self, dt: float = 0.1):
        F = np.eye(8, dtype=np.float64)
        F[0, 2] = dt; F[0, 4] = 0.5 * dt * dt
        F[1, 3] = dt; F[1, 5] = 0.5 * dt * dt
        F[2, 4] = dt; F[3, 5] = dt
        F[6, 7] = dt

        q_pos = 0.02; q_vel = 0.08; q_acc = 0.15; q_rot = 0.04
        Q = np.eye(8, dtype=np.float64) * q_pos
        Q[2:4, 2:4] *= q_vel
        Q[4:6, 4:6] *= q_acc
        Q[6:8, 6:8] *= q_rot

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q

    def update(self, measured_pos: np.ndarray, measured_theta: Optional[float] = None):
        if measured_theta is not None:
            H = np.zeros((3, 8), dtype=np.float64)
            H[0, 0] = 1.0; H[1, 1] = 1.0; H[2, 6] = 1.0
            z = np.array([measured_pos[0], measured_pos[1], measured_theta], dtype=np.float64)
            R = np.eye(3, dtype=np.float64) * 0.04
            y = z - H @ self.state
            y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi
        else:
            H = np.zeros((2, 8), dtype=np.float64)
            H[0, 0] = 1.0; H[1, 1] = 1.0
            z = measured_pos.astype(np.float64)
            R = np.eye(2, dtype=np.float64) * 0.04
            y = z - H @ self.state

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(8, dtype=np.float64) - K @ H) @ self.P


class MetricTrack:
    def __init__(self, track_id: int, class_id: int, initial_pos: np.ndarray, length: float = 2.0, width: float = 1.0):
        self.track_id = track_id
        self.class_id = class_id
        self.length = length
        self.width = width
        self.kf = MetricKalmanFilter(initial_pos)
        self.history = [self.kf.state.copy()]
        self.max_history = 30  # 3.0 seconds at 10Hz
        self.time_since_update = 0
        self.confirmed = False
        self.hits = 1

    def step_predict(self, dt: float = 0.1):
        self.kf.predict(dt)
        self.time_since_update += 1

    def step_update(self, pos: np.ndarray, theta: Optional[float] = None):
        self.kf.update(pos, theta)
        self.history.append(self.kf.state.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)
        self.time_since_update = 0
        self.hits += 1
        if self.hits >= 3:
            self.confirmed = True
