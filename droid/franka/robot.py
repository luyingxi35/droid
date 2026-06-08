# ROBOT SPECIFIC IMPORTS
import os
import time
import logging

import grpc
import numpy as np
import torch
from polymetis import GripperInterface, RobotInterface

from droid.misc.parameters import sudo_password
from droid.misc.subprocess_utils import run_terminal_command, run_threaded_command

# UTILITY SPECIFIC IMPORTS
from droid.misc.transformations import add_poses, euler_to_quat, pose_diff, quat_to_euler
from droid.robot_ik.robot_ik_solver import RobotIKSolver

DEFAULT_FRANKA_GRIPPER_MAX_WIDTH = 0.085




class FrankaRobot:
    def launch_controller(self):
        try:
            self.kill_controller()
        except:
            pass

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._robot_process = run_terminal_command(
            "echo " + sudo_password + " | sudo -S " + "bash " + dir_path + "/launch_robot.sh"
        )
        self._gripper_process = run_terminal_command(
            "echo " + sudo_password + " | sudo -S " + "bash " + dir_path + "/launch_gripper.sh"
        )
        self._server_launched = True
        time.sleep(5)

    def get_gripper_max_width(gripper, default=DEFAULT_FRANKA_GRIPPER_MAX_WIDTH):
        metadata = getattr(gripper, "metadata", None)
        max_width = getattr(metadata, "max_width", None)
        if max_width is None:
            logging.warning(
                "Franka gripper metadata unavailable; using fallback max_width=%.3f m", DEFAULT_FRANKA_GRIPPER_MAX_WIDTH)
            max_width = DEFAULT_FRANKA_GRIPPER_MAX_WIDTH
        return max_width


    def launch_robot(self):
        self._robot = RobotInterface(ip_address="localhost")
        self._gripper = GripperInterface(ip_address="localhost")
        self._max_gripper_width = self.get_gripper_max_width(self._gripper)
        self._ik_solver = RobotIKSolver()
        self._controller_not_loaded = False
        self._joint_impedance_active = False  # True only when joint impedance is the active Polymetis controller
        self._traj_ctrl = None  # HighFreqController instance (set by start_trajectory_controller)

    def kill_controller(self):
        self._robot_process.kill()
        self._gripper_process.kill()

    def update_command(self, command, action_space="cartesian_velocity", gripper_action_space=None, blocking=False):
        action_dict = self.create_action_dict(command, action_space=action_space, gripper_action_space=gripper_action_space)

        self.update_joints(action_dict["joint_position"], velocity=False, blocking=blocking)
        self.update_gripper(action_dict["gripper_position"], velocity=False, blocking=blocking)

        return action_dict

    def update_pose(self, command, velocity=False, blocking=False):
        if blocking:
            if velocity:
                curr_pose = self.get_ee_pose()
                cartesian_delta = self._ik_solver.cartesian_velocity_to_delta(command)
                command = add_poses(cartesian_delta, curr_pose)

            pos = torch.Tensor(command[:3])
            quat = torch.Tensor(euler_to_quat(command[3:6]))
            curr_joints = self._robot.get_joint_positions()
            desired_joints = self._robot.solve_inverse_kinematics(pos, quat, curr_joints)
            self.update_joints(desired_joints, velocity=False, blocking=True)
        else:
            if not velocity:
                curr_pose = self.get_ee_pose()
                cartesian_delta = pose_diff(command, curr_pose)
                command = self._ik_solver.cartesian_delta_to_velocity(cartesian_delta)

            robot_state = self.get_robot_state()[0]
            joint_velocity = self._ik_solver.cartesian_velocity_to_joint_velocity(command, robot_state=robot_state)

            self.update_joints(joint_velocity, velocity=True, blocking=False)

    def update_joints(self, command, velocity=False, blocking=False, cartesian_noise=None):
        if cartesian_noise is not None:
            command = self.add_noise_to_joints(command, cartesian_noise)
        command = torch.Tensor(command)

        if velocity:
            joint_delta = self._ik_solver.joint_velocity_to_delta(command)
            command = joint_delta + self._robot.get_joint_positions()

        def helper_non_blocking():
            # Ensure joint impedance is active.  update_desired_joint_positions()
            # requires joint impedance; cartesian impedance (started after blocking
            # moves at line 128) will cause Polymetis to reject the call.
            if not self._robot.is_running_policy() or not self._joint_impedance_active:
                self._controller_not_loaded = True
                if self._robot.is_running_policy():
                    self._robot.terminate_current_policy()
                self._robot.start_joint_impedance()
                timeout = time.time() + 5
                while not self._robot.is_running_policy():
                    time.sleep(0.01)
                    if time.time() > timeout:
                        self._robot.start_joint_impedance()
                        timeout = time.time() + 5

                self._joint_impedance_active = True
                self._controller_not_loaded = False
            try:
                self._robot.update_desired_joint_positions(command)
            except grpc.RpcError:
                pass

        if blocking:
            if self._robot.is_running_policy():
                self._robot.terminate_current_policy()
            try:
                time_to_go = self.adaptive_time_to_go(command)
                self._robot.move_to_joint_positions(command, time_to_go=time_to_go)
            except grpc.RpcError:
                pass

            self._robot.start_cartesian_impedance()
            self._joint_impedance_active = False
        else:
            if not self._controller_not_loaded:
                run_threaded_command(helper_non_blocking)

    def update_gripper(self, command, velocity=True, blocking=False):
        if velocity:
            gripper_delta = self._ik_solver.gripper_velocity_to_delta(command)
            command = gripper_delta + self.get_gripper_position()

        command = float(np.clip(command, 0, 1))
        self._gripper.goto(width=self._max_gripper_width * (1 - command), speed=0.05, force=0.1, blocking=blocking)

    def add_noise_to_joints(self, original_joints, cartesian_noise):
        original_joints = torch.Tensor(original_joints)

        pos, quat = self._robot.robot_model.forward_kinematics(original_joints)
        curr_pose = pos.tolist() + quat_to_euler(quat).tolist()
        new_pose = add_poses(cartesian_noise, curr_pose)

        new_pos = torch.Tensor(new_pose[:3])
        new_quat = torch.Tensor(euler_to_quat(new_pose[3:]))

        noisy_joints, success = self._robot.solve_inverse_kinematics(new_pos, new_quat, original_joints)

        if success:
            desired_joints = noisy_joints
        else:
            desired_joints = original_joints

        return desired_joints.tolist()

    def get_joint_positions(self):
        return self._robot.get_joint_positions().tolist()

    def get_joint_velocities(self):
        return self._robot.get_joint_velocities().tolist()

    def get_gripper_position(self):
        return 1 - (self._gripper.get_state().width / self._max_gripper_width)

    def get_ee_pose(self):
        pos, quat = self._robot.get_ee_pose()
        angle = quat_to_euler(quat.numpy())
        return np.concatenate([pos, angle]).tolist()

    def get_robot_state(self):
        robot_state = self._robot.get_robot_state()
        gripper_position = self.get_gripper_position()
        pos, quat = self._robot.robot_model.forward_kinematics(torch.Tensor(robot_state.joint_positions))
        cartesian_position = pos.tolist() + quat_to_euler(quat.numpy()).tolist()

        state_dict = {
            "cartesian_position": cartesian_position,
            "gripper_position": gripper_position,
            "joint_positions": list(robot_state.joint_positions),
            "joint_velocities": list(robot_state.joint_velocities),
            "joint_torques_computed": list(robot_state.joint_torques_computed),
            "prev_joint_torques_computed": list(robot_state.prev_joint_torques_computed),
            "prev_joint_torques_computed_safened": list(robot_state.prev_joint_torques_computed_safened),
            "motor_torques_measured": list(robot_state.motor_torques_measured),
            "prev_controller_latency_ms": robot_state.prev_controller_latency_ms,
            "prev_command_successful": robot_state.prev_command_successful,
        }

        timestamp_dict = {
            "robot_timestamp_seconds": robot_state.timestamp.seconds,
            "robot_timestamp_nanos": robot_state.timestamp.nanos,
        }

        return state_dict, timestamp_dict

    def adaptive_time_to_go(self, desired_joint_position, t_min=0, t_max=4):
        curr_joint_position = self._robot.get_joint_positions()
        displacement = desired_joint_position - curr_joint_position
        time_to_go = self._robot._adaptive_time_to_go(displacement)
        clamped_time_to_go = min(t_max, max(time_to_go, t_min))
        return clamped_time_to_go

    # ── High-frequency trajectory controller ──────────────────────────────────

    def start_trajectory_controller(self, frequency=200.0):
        """Start the high-frequency joint position + state-logging controller.

        Explicitly switches Polymetis to joint impedance before spawning
        HighFreqController.  The warm-up update_joints(blocking=False) starts
        cartesian impedance, but update_desired_joint_positions() requires joint
        impedance — these are different controllers in Polymetis.

        zerorpc-exposed: called by GPU-server via ServerInterface.
        """
        from droid.franka.trajectory_controller import HighFreqController
        if self._traj_ctrl is not None and self._traj_ctrl.is_alive():
            return  # already running

        # Switch to joint impedance.  terminate first if another policy is active.
        if self._robot.is_running_policy():
            self._robot.terminate_current_policy()
        self._joint_impedance_active = False
        self._robot.start_joint_impedance()
        deadline = time.time() + 3.0
        while not self._robot.is_running_policy():
            time.sleep(0.01)
            if time.time() > deadline:
                raise RuntimeError(
                    "Timed out waiting for joint impedance controller to start"
                )
        self._joint_impedance_active = True

        self._traj_ctrl = HighFreqController(
            self._robot, self._gripper, frequency=float(frequency)
        )
        self._traj_ctrl.start()

    def stop_trajectory_controller(self):
        """Stop the high-frequency controller. Called at episode end.

        zerorpc-exposed: called by GPU-server via ServerInterface.
        """
        if self._traj_ctrl is not None:
            self._traj_ctrl.stop()
            self._traj_ctrl.join(timeout=2.0)
            self._traj_ctrl = None
        # Terminate any active Polymetis policy so the subsequent blocking reset
        # (update_joints blocking=True → move_to_joint_positions) starts clean.
        try:
            if self._robot.is_running_policy():
                self._robot.terminate_current_policy()
        except grpc.RpcError:
            pass
        self._joint_impedance_active = False

    def get_state_history(self, n=100):
        """Return (timestamps, joints_list, gripper_list) from the HighFreqController
        state ring buffer.

        Used by the GPU server to retrieve a high-frequency proprioception history
        for UMI-style interpolation to the camera observation timestamp.

        zerorpc-exposed: returns plain Python lists (no numpy) for msgpack compat.
        Returns three empty lists if the trajectory controller is not running.
        """
        if self._traj_ctrl is None or not self._traj_ctrl.is_alive():
            return [], [], []
        return self._traj_ctrl.get_state_history(int(n))

    def add_waypoints(self, times_list, positions_list):
        """Send a batch of arm waypoints to the trajectory controller.

        Args:
            times_list: list[float] — wall-clock target times (time.time()).
                        Already compensated for robot_action_latency by caller.
            positions_list: list[list[float]] — shape (N, 7), absolute joint angles.

        zerorpc-exposed: called ~10 Hz from GPU-server policy loop.
        Lists are used (not numpy) because msgpack serialises them natively.
        """
        if self._traj_ctrl is None or not self._traj_ctrl.is_alive():
            raise RuntimeError(
                "Trajectory controller not running. "
                "Call start_trajectory_controller() first."
            )
        self._traj_ctrl.add_waypoints(
            np.array(times_list, dtype=np.float64),
            np.array(positions_list, dtype=np.float64),
        )

    def create_action_dict(self, action, action_space, gripper_action_space=None, robot_state=None):
        assert action_space in ["cartesian_position", "joint_position", "cartesian_velocity", "joint_velocity"]
        if robot_state is None:
            robot_state = self.get_robot_state()[0]
        action_dict = {"robot_state": robot_state}
        velocity = "velocity" in action_space

        if gripper_action_space is None:
            gripper_action_space = "velocity" if velocity else "position"
        assert gripper_action_space in ["velocity", "position"]
            

        if gripper_action_space == "velocity":
            action_dict["gripper_velocity"] = action[-1]
            gripper_delta = self._ik_solver.gripper_velocity_to_delta(action[-1])
            gripper_position = robot_state["gripper_position"] + gripper_delta
            action_dict["gripper_position"] = float(np.clip(gripper_position, 0, 1))
        else:
            action_dict["gripper_position"] = float(np.clip(action[-1], 0, 1))
            gripper_delta = action_dict["gripper_position"] - robot_state["gripper_position"]
            gripper_velocity = self._ik_solver.gripper_delta_to_velocity(gripper_delta)
            action_dict["gripper_delta"] = gripper_velocity

        if "cartesian" in action_space:
            if velocity:
                action_dict["cartesian_velocity"] = action[:-1]
                cartesian_delta = self._ik_solver.cartesian_velocity_to_delta(action[:-1])
                action_dict["cartesian_position"] = add_poses(
                    cartesian_delta, robot_state["cartesian_position"]
                ).tolist()
            else:
                action_dict["cartesian_position"] = action[:-1]
                cartesian_delta = pose_diff(action[:-1], robot_state["cartesian_position"])
                cartesian_velocity = self._ik_solver.cartesian_delta_to_velocity(cartesian_delta)
                action_dict["cartesian_velocity"] = cartesian_velocity.tolist()

            action_dict["joint_velocity"] = self._ik_solver.cartesian_velocity_to_joint_velocity(
                action_dict["cartesian_velocity"], robot_state=robot_state
            ).tolist()
            joint_delta = self._ik_solver.joint_velocity_to_delta(action_dict["joint_velocity"])
            action_dict["joint_position"] = (joint_delta + np.array(robot_state["joint_positions"])).tolist()

        if "joint" in action_space:
            # NOTE: Joint to Cartesian has undefined dynamics due to IK
            if velocity:
                action_dict["joint_velocity"] = action[:-1]
                joint_delta = self._ik_solver.joint_velocity_to_delta(action[:-1])
                action_dict["joint_position"] = (joint_delta + np.array(robot_state["joint_positions"])).tolist()
            else:
                action_dict["joint_position"] = action[:-1]
                joint_delta = np.array(action[:-1]) - np.array(robot_state["joint_positions"])
                joint_velocity = self._ik_solver.joint_delta_to_velocity(joint_delta)
                action_dict["joint_velocity"] = joint_velocity.tolist()

        return action_dict
