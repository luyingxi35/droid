"""High-frequency joint position controller for the Franka robot.

Runs as a background threading.Thread on the NUC alongside the Polymetis
server, accessing polymetis.RobotInterface (gRPC localhost) at high frequency.

Used by FrankaRobot.start_trajectory_controller() to decouple remote
policy inference (10 Hz, GPU server) from smooth robot execution (200 Hz, NUC).
"""

import logging
import threading
import time

import numpy as np


class JointTrajectoryInterpolator:
    """Linear interpolator over (wall-clock time, joint_positions_7d) waypoints.

    Thread-safe when calls are serialized by the caller's lock.
    """

    def __init__(self):
        self._times = np.array([], dtype=np.float64)
        self._positions = np.empty((0, 7), dtype=np.float64)

    def set_waypoints(self, times: np.ndarray, positions: np.ndarray) -> None:
        """Replace the current trajectory. times must be sorted ascending."""
        self._times = np.asarray(times, dtype=np.float64)
        self._positions = np.asarray(positions, dtype=np.float64)

    def __call__(self, t: float):
        """Return interpolated 7-dof joint positions at wall-clock time t.

        Returns None if no waypoints are loaded.
        """
        if len(self._times) == 0:
            return None
        if t <= self._times[0]:
            return self._positions[0].copy()
        if t >= self._times[-1]:
            return self._positions[-1].copy()
        idx = int(np.searchsorted(self._times, t, side="right")) - 1
        t0, t1 = self._times[idx], self._times[idx + 1]
        alpha = (t - t0) / (t1 - t0)
        return (1.0 - alpha) * self._positions[idx] + alpha * self._positions[idx + 1]

    @property
    def is_empty(self) -> bool:
        return len(self._times) == 0


class HighFreqController(threading.Thread):
    """200 Hz joint position controller that runs on the NUC.

    Accesses polymetis.RobotInterface (gRPC localhost) directly — safe
    because this thread runs on the NUC alongside the Polymetis server.

    Prerequisites (must be done before calling start()):
        FrankaRobot.update_joints(current_joints, velocity=False, blocking=False)
    This triggers DROID's impedance controller startup so Polymetis is ready
    to accept continuous position targets via update_desired_joint_positions.

    The controller runs independently at high frequency.  The GPU-server policy
    loop calls add_waypoints() at ~10 Hz; the controller interpolates between
    those waypoints smoothly at 200 Hz.
    """

    def __init__(self, polymetis_robot, frequency: float = 200.0) -> None:
        super().__init__(daemon=True, name="HighFreqController")
        self._robot = polymetis_robot  # polymetis.RobotInterface (gRPC localhost)
        self._dt = 1.0 / frequency
        self._interp = JointTrajectoryInterpolator()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def add_waypoints(self, times: np.ndarray, positions: np.ndarray) -> None:
        """Replace the current trajectory with a new batch of waypoints.

        Non-blocking, thread-safe.

        Args:
            times: (N,) float64 wall-clock target times (time.time() seconds).
                   Already adjusted for robot_action_latency by the caller.
            positions: (N, 7) float64 absolute joint angles in radians.
        """
        with self._lock:
            self._interp.set_waypoints(times, positions)

    def stop(self) -> None:
        """Signal the controller loop to exit."""
        self._stop_event.set()

    def run(self) -> None:
        import grpc
        import torch

        t_start = time.time()
        iter_idx = 0

        while not self._stop_event.is_set():
            t_now = time.time()

            with self._lock:
                joint_target = self._interp(t_now)

            if joint_target is not None:
                try:
                    self._robot.update_desired_joint_positions(
                        torch.tensor(joint_target, dtype=torch.float32)
                    )
                except grpc.RpcError:
                    pass  # transient gRPC error — skip this tick
                except Exception:
                    logging.exception("HighFreqController: unexpected error in update_desired_joint_positions")

            iter_idx += 1
            sleep_s = t_start + iter_idx * self._dt - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)
