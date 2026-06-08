"""High-frequency joint position controller for the Franka robot.

Runs as a background threading.Thread on the NUC alongside the Polymetis
server, accessing polymetis.RobotInterface (gRPC localhost) at high frequency.

Used by FrankaRobot.start_trajectory_controller() to decouple remote
policy inference (10 Hz, GPU server) from smooth robot execution (200 Hz, NUC).
"""

from __future__ import annotations  # Python 3.8 compat: defer annotation evaluation

import logging
import threading
import time

import numpy as np


class JointTrajectoryInterpolator:
    """Linear interpolator over (monotonic time, joint_positions_7d) waypoints.

    Internally all times are in time.monotonic() so the 200 Hz loop is immune
    to NTP wall-clock adjustments.  Callers that hold wall-clock times must
    convert them before calling update_waypoints() (see HighFreqController).

    Thread-safe when calls are serialized by the caller's lock.
    """

    def __init__(self):
        self._times = np.array([], dtype=np.float64)
        self._positions = np.empty((0, 7), dtype=np.float64)

    def set_waypoints(self, times: np.ndarray, positions: np.ndarray) -> None:
        """Hard-replace the current trajectory. times must be sorted ascending.

        Prefer update_waypoints() for normal use — it guarantees C0 continuity
        and applies a joint-speed cap.  set_waypoints() is kept for callers
        that need unconditional replacement (e.g. tests, episode reset).
        """
        self._times = np.asarray(times, dtype=np.float64)
        self._positions = np.asarray(positions, dtype=np.float64)

    def update_waypoints(
        self,
        times: np.ndarray,
        positions: np.ndarray,
        curr_time: float,
        max_joint_speed_rad_s: float = 3.0,
    ) -> None:
        """Replace trajectory, preserving C0 continuity from the current execution point.

        Two guarantees:

        1. **Continuity** — if ``curr_time`` precedes the first new waypoint,
           the current interpolated pose is prepended as a leading waypoint so
           the 200 Hz loop transitions smoothly from wherever it currently is.
           This correctly handles the *overlap* case where a new action chunk
           arrives while the previous chunk is still being executed: without
           this, ``JointTrajectoryInterpolator.__call__`` would clamp to the
           first new waypoint immediately and the robot would jump.

        2. **Speed cap** — if any consecutive waypoint pair implies a joint
           velocity exceeding ``max_joint_speed_rad_s``, the later waypoint's
           time is extended (and all subsequent times shifted) to satisfy the
           limit.  This mirrors UMI's
           ``PoseTrajectoryInterpolator.schedule_waypoint`` max-speed constraint
           and prevents runaway velocities when action chunks have large gaps.

        Args:
            times: (N,) monotonic target times for each waypoint.
            positions: (N, 7) absolute joint angles in radians.
            curr_time: current time.monotonic() value at the call site.
            max_joint_speed_rad_s: per-joint speed limit (rad/s). Default 3.0
                rad/s is roughly 1.5× the DROID training speed at action_scale=1.
        """
        times     = np.asarray(times,     dtype=np.float64)
        positions = np.asarray(positions, dtype=np.float64)

        # ── 1. Continuity: prepend current interpolated position ────────────
        curr_pos = self.__call__(curr_time)
        if curr_pos is not None and len(times) > 0 and curr_time < times[0]:
            times     = np.concatenate([[curr_time], times])
            positions = np.vstack([curr_pos[None], positions])

        # ── 2. Speed cap: extend waypoint times where needed ────────────────
        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            if dt <= 0:
                continue
            max_delta   = float(np.max(np.abs(positions[i] - positions[i - 1])))
            required_dt = max_delta / max_joint_speed_rad_s
            if required_dt > dt:
                times[i:] = times[i:] + (required_dt - dt)  # shift tail forward

        self._times     = times
        self._positions = positions

    def __call__(self, t: float):
        """Return interpolated 7-dof joint positions at monotonic time t.

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

    Accesses polymetis.RobotInterface and GripperInterface (both gRPC localhost)
    directly — safe because this thread runs on the NUC alongside Polymetis.

    Responsibilities:
    1. Trajectory execution: interpolate waypoints at 200 Hz and send
       update_desired_joint_positions() to the arm.
    2. State history: record (timestamp, joint_positions_7d, gripper_pos_norm)
       at each tick into a ring buffer so the GPU-server can retrieve a
       high-frequency history for UMI-style proprioception interpolation.

    Prerequisites (must be done before calling start()):
        FrankaRobot.update_joints(current_joints, velocity=False, blocking=False)
    This triggers DROID's impedance controller startup so Polymetis accepts
    continuous position targets.
    """

    STATE_HISTORY_LEN = 400  # 400 entries @ 200 Hz = 2 s of history

    def __init__(
        self,
        polymetis_robot,    # polymetis.RobotInterface  (gRPC localhost, thread-safe)
        polymetis_gripper,  # polymetis.GripperInterface (gRPC localhost, thread-safe)
        frequency: float = 200.0,
    ) -> None:
        super().__init__(daemon=True, name="HighFreqController")
        self._robot   = polymetis_robot
        self._gripper = polymetis_gripper
        self._dt = 1.0 / frequency
        # ── Trajectory interpolator (arm waypoints) ───────────────────────────
        self._interp = JointTrajectoryInterpolator()
        self._lock = threading.Lock()
        # ── State ring buffer ─────────────────────────────────────────────────
        # Each entry: [timestamp_s, j0, j1, j2, j3, j4, j5, j6, gripper_norm]
        # gripper_norm = 1 - (width / max_width)  [0=open, 1=closed]
        # matches FrankaRobot.get_gripper_position() convention.
        self._state_buf: list[list[float]] = []
        self._state_lock = threading.Lock()
        self._max_gripper_width: float = 1.0  # overwritten in run() from metadata
        self._stop_event = threading.Event()

    # ── Waypoint scheduling ────────────────────────────────────────────────────

    def add_waypoints(self, times: np.ndarray, positions: np.ndarray) -> None:
        """Replace the current arm trajectory with a new batch of waypoints.

        Non-blocking, thread-safe.

        The incoming ``times`` are wall-clock (``time.time()``) values as
        supplied by the GPU server.  They are converted to ``time.monotonic()``
        at this boundary so the internal interpolator is unaffected by NTP
        adjustments.  A single offset sample is sufficient because NTP drift
        is orders of magnitude slower than the inter-call interval (~100 ms).

        Args:
            times: (N,) float64 wall-clock target times (time.time() seconds),
                   already adjusted for robot_action_latency by the caller.
            positions: (N, 7) float64 absolute joint angles in radians.
        """
        # Convert wall-clock → monotonic once at the entry boundary.
        _wall_to_mono = time.monotonic() - time.time()
        mono_times = np.asarray(times, dtype=np.float64) + _wall_to_mono
        with self._lock:
            self._interp.update_waypoints(
                mono_times,
                np.asarray(positions, dtype=np.float64),
                curr_time=time.monotonic(),
            )

    # ── State history access ───────────────────────────────────────────────────

    def get_state_history(self, n: int = 100) -> tuple[list, list, list]:
        """Return the last n state records as three parallel lists.

        Thread-safe.  Called by FrankaRobot.get_state_history() which is
        zerorpc-exposed, so all values must be plain Python types (no numpy).

        Returns:
            times:   list[float]       — wall-clock timestamps (seconds)
            joints:  list[list[float]] — 7-DOF joint positions (radians)
            gripper: list[float]       — normalized gripper position [0=open, 1=closed]
        """
        with self._state_lock:
            recent = self._state_buf[-n:] if len(self._state_buf) >= n else list(self._state_buf)
        times   = [s[0]   for s in recent]
        joints  = [s[1:8] for s in recent]
        gripper = [s[8]   for s in recent]
        return times, joints, gripper

    # ── Controller stop ────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Signal the controller loop to exit."""
        self._stop_event.set()

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        import grpc
        import torch

        # Read max gripper width once from Polymetis metadata (fast gRPC call).
        try:
            self._max_gripper_width = float(self._gripper.metadata.max_width)
        except Exception:
            logging.warning("HighFreqController: could not read gripper metadata; "
                            "gripper normalization will use default max_width=1.0")

        # Use time.monotonic() for the control loop so NTP wall-clock
        # adjustments cannot corrupt inter-tick sleep timing.
        # time.time() is still used for the state ring buffer because those
        # timestamps must correlate with the GPU server's observation timestamps
        # (which are anchored to time.time() via camera hardware timestamps).
        t_start = time.monotonic()
        iter_idx = 0

        while not self._stop_event.is_set():
            t_now_mono = time.monotonic()   # for loop scheduling + interpolation
            t_now_wall = time.time()        # for state ring buffer (wall-clock)

            # ── 1. Read current robot state into ring buffer ───────────────────
            joints_list       = [-1.0] * 7
            gripper_norm      = -1.0
            try:
                joints_list = self._robot.get_joint_positions().numpy().tolist()
            except Exception:
                pass
            try:
                width       = self._gripper.get_state().width
                gripper_norm = 1.0 - float(width) / self._max_gripper_width
            except Exception:
                pass

            with self._state_lock:
                self._state_buf.append([t_now_wall] + joints_list + [gripper_norm])
                if len(self._state_buf) > self.STATE_HISTORY_LEN:
                    self._state_buf.pop(0)

            # ── 2. Execute interpolated arm position target ────────────────────
            with self._lock:
                joint_target = self._interp(t_now_mono)

            if joint_target is not None:
                try:
                    self._robot.update_desired_joint_positions(
                        torch.tensor(joint_target, dtype=torch.float32)
                    )
                except grpc.RpcError:
                    pass  # transient gRPC error — skip this tick
                except Exception:
                    logging.exception("HighFreqController: unexpected error in "
                                      "update_desired_joint_positions")

            iter_idx += 1
            sleep_s = t_start + iter_idx * self._dt - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
