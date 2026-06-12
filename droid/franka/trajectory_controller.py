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


DEFAULT_FRANKA_GRIPPER_MAX_WIDTH = 0.085


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
        max_joint_speed_rad_s: float = 0.5,
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

        # ── 1. Continuity: bridge from current commanded position ───────────
        # arm_times are computed as new_t - robot_action_latency, so times[0]
        # is typically ~170ms in the PAST relative to curr_time.  The old
        # condition `curr_time < times[0]` was therefore never True, meaning
        # the bridge never fired and every chunk switch caused a hard jump.
        # Fix: always trim past waypoints and prepend (curr_time, curr_pos) so
        # the 200 Hz loop transitions smoothly from wherever it currently is.
        curr_pos = self.__call__(curr_time)
        if curr_pos is not None and len(times) > 0:
            future_mask = times > curr_time
            if np.any(future_mask):
                # Keep only future waypoints; bridge from current commanded pos.
                times     = np.concatenate([[curr_time], times[future_mask]])
                positions = np.vstack([curr_pos[None], positions[future_mask]])
            else:
                # Entire chunk is in the past (clock drift or very slow inference).
                # Smoothly go from curr_pos to last commanded pos over original
                # trajectory duration instead of clamping instantly.
                traj_duration = float(times[-1] - times[0]) if len(times) > 1 else 0.1
                times     = np.array([curr_time, curr_time + max(traj_duration, 0.05)])
                positions = np.vstack([curr_pos[None], positions[[-1]]])

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
    """Joint position controller that runs on the NUC.

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
        frequency: float = 100.0,
        on_fatal_error=None,
    ) -> None:
        super().__init__(daemon=True, name="HighFreqController")
        self._robot   = polymetis_robot
        self._gripper = polymetis_gripper
        self._dt = 1.0 / frequency
        self._frequency = float(frequency)
        # ── Trajectory interpolator (arm waypoints) ───────────────────────────
        self._interp = JointTrajectoryInterpolator()
        self._lock = threading.Lock()
        # ── State ring buffer ─────────────────────────────────────────────────
        # Each entry: [timestamp_s, j0, j1, j2, j3, j4, j5, j6, gripper_norm]
        # gripper_norm = 1 - (width / max_width)  [0=open, 1=closed]
        # matches FrankaRobot.get_gripper_position() convention.
        self._state_buf: list[list[float]] = []
        self._state_lock = threading.Lock()
        self._max_gripper_width: float = DEFAULT_FRANKA_GRIPPER_MAX_WIDTH
        self._stop_event = threading.Event()
        self._consecutive_joint_update_failures = 0
        self._fatal_error_reason = None
        self._on_fatal_error = on_fatal_error

    # ── Waypoint scheduling ────────────────────────────────────────────────────

    def add_waypoints(
        self,
        times: np.ndarray,
        positions: np.ndarray,
        max_joint_speed_rad_s: float = 0.5,
    ) -> None:
        """Replace the current arm trajectory with a new batch of waypoints.

        Non-blocking, thread-safe.

        The incoming ``times`` are **seconds relative to when the GPU server
        called add_waypoints** (i.e. ``arm_times - time.time()`` on the GPU).
        Using offsets instead of absolute wall-clock values makes the
        conversion immune to GPU-NUC clock skew: even a 100 ms NTP offset
        between the two machines does not shift waypoint times.  The only
        residual error is network latency (~5 ms), which is acceptable.

        Args:
            times: (N,) float64 time offsets (seconds from GPU call time).
                   Negative values are waypoints already in the past;
                   positive values are future targets.
            positions: (N, 7) float64 absolute joint angles in radians.
            max_joint_speed_rad_s: per-joint speed cap (rad/s). Default 0.5
                (conservative safety limit). Pass a higher value (e.g. 3.0) via
                CLI config to allow faster execution. Forwarded to
                update_waypoints() speed-cap logic.
        """
        # Convert relative offsets → NUC monotonic times.
        # Network latency (~5 ms) shifts all offsets slightly toward the past;
        # update_waypoints trims past waypoints and bridges automatically.
        mono_times = time.monotonic() + np.asarray(times, dtype=np.float64)
        with self._lock:
            self._interp.update_waypoints(
                mono_times,
                np.asarray(positions, dtype=np.float64),
                curr_time=time.monotonic(),
                max_joint_speed_rad_s=max_joint_speed_rad_s,
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

    def get_failure_reason(self):
        return self._fatal_error_reason

    def _record_fatal_error(self, reason: str) -> None:
        if self._fatal_error_reason is not None:
            return
        self._fatal_error_reason = str(reason)
        logging.error("HighFreqController: %s", self._fatal_error_reason)
        self._stop_event.set()
        if self._on_fatal_error is not None:
            try:
                self._on_fatal_error(self._fatal_error_reason)
            except Exception:
                logging.exception(
                    "HighFreqController: failed to report fatal error to owner"
                )

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        import grpc
        import torch

        # Read max gripper width once from Polymetis metadata (fast gRPC call).
        try:
            metadata = getattr(self._gripper, "metadata", None)
            max_width = getattr(metadata, "max_width", 0.0)
            if max_width <= 0.0:
                raise ValueError("gripper metadata max_width unavailable")
            self._max_gripper_width = float(max_width)
        except Exception:
            logging.warning(
                "HighFreqController: could not read gripper metadata; "
                "gripper normalization will use default max_width=%.3f m",
                DEFAULT_FRANKA_GRIPPER_MAX_WIDTH,
            )

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
                    self._consecutive_joint_update_failures = 0
                except grpc.RpcError:
                    self._consecutive_joint_update_failures += 1
                    if self._consecutive_joint_update_failures == 1:
                        logging.warning(
                            "HighFreqController: desired joint update rejected by Polymetis."
                        )
                    try:
                        policy_running = self._robot.is_running_policy()
                    except Exception:
                        policy_running = None
                    if policy_running is False:
                        self._record_fatal_error(
                            "Polymetis joint-impedance controller is no longer running. "
                            "This usually means a robot reflex or control interruption occurred; "
                            "call start_trajectory_controller() again after recovery."
                        )
                    elif self._consecutive_joint_update_failures >= 5:
                        self._record_fatal_error(
                            "Desired joint updates were rejected repeatedly by Polymetis "
                            f"at {self._frequency:.1f} Hz."
                        )
                except Exception:
                    self._consecutive_joint_update_failures += 1
                    logging.exception("HighFreqController: unexpected error in "
                                      "update_desired_joint_positions")
                    if self._consecutive_joint_update_failures >= 5:
                        self._record_fatal_error(
                            "Unexpected errors occurred repeatedly while sending desired joint positions."
                        )

            iter_idx += 1
            sleep_s = t_start + iter_idx * self._dt - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
