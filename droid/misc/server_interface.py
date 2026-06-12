import time

import numpy as np
import zerorpc


def attempt_n_times(function_list, max_attempts, sleep_time=0.1):
    if type(function_list) is not list:
        function_list = list(function_list)

    for i in range(max_attempts):
        try:
            [f() for f in function_list]
            return
        except zerorpc.exceptions.RemoteError as err:
            last_attempt = i == (max_attempts - 1)
            if last_attempt:
                raise err
            else:
                time.sleep(sleep_time)


def retry_on_exception(function, max_attempts, sleep_time=0.1, exception_types=(Exception,)):
    for i in range(max_attempts):
        try:
            return function()
        except exception_types:
            last_attempt = i == (max_attempts - 1)
            if last_attempt:
                raise
            time.sleep(sleep_time)


class ServerInterface:
    def __init__(self, ip_address="127.0.0.1", launch=True):
        self.ip_address = ip_address
        self.establish_connection()

        if launch:
            self.launch_controller()
            try:
                retry_on_exception(
                    self.launch_robot,
                    max_attempts=2,
                    sleep_time=0.5,
                    exception_types=(zerorpc.exceptions.RemoteError, RuntimeError),
                )
            except Exception:
                try:
                    self.kill_controller()
                except zerorpc.exceptions.RemoteError:
                    pass
                raise

    def establish_connection(self):
        self.server = zerorpc.Client(heartbeat=20)
        self.server.connect("tcp://" + self.ip_address + ":4242")

    def launch_controller(self):
        self.server.launch_controller()

    def launch_robot(self):
        self.server.launch_robot()

    def kill_controller(self):
        self.server.kill_controller()

    def get_controller_status(self):
        return self.server.get_controller_status()

    def preflight_check(self):
        return self.server.preflight_check()

    def update_command(self, command, action_space="cartesian_velocity", gripper_action_space="velocity", blocking=False):
        action_dict = self.server.update_command(command.tolist(), action_space, gripper_action_space, blocking)
        return action_dict

    def create_action_dict(self, command, action_space="cartesian_velocity"):
        action_dict = self.server.create_action_dict(command.tolist(), action_space)
        return action_dict

    def update_pose(self, command, velocity=True, blocking=False):
        self.server.update_pose(command.tolist(), velocity, blocking)

    def update_joints(self, command, velocity=True, blocking=False, cartesian_noise=None):
        if cartesian_noise is not None:
            cartesian_noise = cartesian_noise.tolist()
        self.server.update_joints(command.tolist(), velocity, blocking, cartesian_noise)

    def update_gripper(self, command, velocity=True, blocking=False):
        self.server.update_gripper(command, velocity, blocking)

    def get_ee_pose(self):
        return np.array(self.server.get_ee_pose())

    def get_joint_positions(self):
        return np.array(self.server.get_joint_positions())

    def get_joint_velocities(self):
        return np.array(self.server.get_joint_velocities())

    def get_gripper_state(self):
        return self.server.get_gripper_state()

    def get_robot_state(self):
        return self.server.get_robot_state()

    # ── High-frequency trajectory controller ──────────────────────────────────

    def get_state_history(self, n=100):
        """Return (timestamps, joints_list, gripper_list) from NUC's HighFreqController.

        Used for UMI-style proprioception interpolation: the GPU server calls this
        at each 10 Hz tick to get a high-rate history of (arm, gripper) states, then
        interpolates to the camera observation timestamp.
        """
        return self.server.get_state_history(int(n))

    def prepare_for_streaming(self, timeout_s=5.0):
        """Synchronously prepare the NUC-side robot for joint-target streaming."""
        self.server.prepare_for_streaming(float(timeout_s))

    def start_trajectory_controller(self, frequency=100.0):
        """Start the high-frequency joint position controller on the NUC server."""
        self.server.start_trajectory_controller(float(frequency))

    def stop_trajectory_controller(self):
        """Stop the high-frequency controller on the NUC server."""
        self.server.stop_trajectory_controller()

    def add_waypoints(self, times_list, positions_list, max_joint_speed_rad_s=0.5):
        """Send a waypoint batch to the NUC trajectory controller.

        Args:
            times_list: list[float] — time offsets from the caller's current time.
            positions_list: list[list[float]] — shape (N, 7), joint angles.
            max_joint_speed_rad_s: per-joint speed cap forwarded to the NUC.
        """
        self.server.add_waypoints(
            times_list,
            positions_list,
            float(max_joint_speed_rad_s),
        )
