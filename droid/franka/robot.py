# ROBOT SPECIFIC IMPORTS
import json
import logging
import os
import re
import signal
import socket
import subprocess
import threading
import time
from enum import Enum

import grpc
import numpy as np
import polymetis_pb2
import polymetis_pb2_grpc
import torch
from polymetis import GripperInterface, RobotInterface

from droid.misc.parameters import sudo_password
from droid.misc.subprocess_utils import run_terminal_command, run_threaded_command

# UTILITY SPECIFIC IMPORTS
from droid.misc.transformations import add_poses, euler_to_quat, pose_diff, quat_to_euler
from droid.robot_ik.robot_ik_solver import RobotIKSolver

DEFAULT_FRANKA_GRIPPER_MAX_WIDTH = 0.085
CONTROLLER_STARTUP_TIMEOUT_S = 20.0
PROCESS_SHUTDOWN_TIMEOUT_S = 5.0
ROBOT_SERVER_PORT = 50051
GRIPPER_SERVER_PORT = 50052
SERVER_RPC_PORT = 4242
CONTROLLER_STATE_FILE = "/tmp/droid_franka_controller_state.json"
DEFAULT_TRAJECTORY_CONTROLLER_FREQUENCY_HZ = 100.0
KNOWN_CONTROLLER_COMMAND_MARKERS = (
    "launch_gripper.py",
    "launch_robot.py",
    "franka_panda_client",
    "/polymetis/build/run_server",
)


class ControllerStatus(str, Enum):
    IDLE = "idle"
    PREFLIGHT_FAILED = "preflight_failed"
    STARTING = "starting"
    READY = "ready"
    FAULTED = "faulted"




class FrankaRobot:
    def __init__(self):
        self._gripper_process = None
        self._robot_process = None
        self._server_launched = False
        self._robot = None
        self._gripper = None
        self._max_gripper_width = DEFAULT_FRANKA_GRIPPER_MAX_WIDTH
        self._ik_solver = None
        self._controller_not_loaded = False
        self._joint_impedance_active = False
        self._traj_ctrl = None
        self._controller_lock = threading.RLock()
        self._status = ControllerStatus.IDLE
        self._status_reason = "Controller has not been started."
        self._last_preflight = {}
        self._launch_in_progress = False
        self._status_lock = threading.RLock()
        self._traj_ctrl_failure_reason = None

    def _set_status(self, status, reason):
        with self._status_lock:
            self._status = ControllerStatus(status)
            self._status_reason = str(reason)

    def _record_trajectory_controller_failure_locked(self, reason):
        self._traj_ctrl_failure_reason = str(reason)
        self._joint_impedance_active = False
        logging.error("Trajectory controller failed: %s", self._traj_ctrl_failure_reason)

    def _consume_trajectory_controller_failure_reason_locked(self):
        reason = self._traj_ctrl_failure_reason
        self._traj_ctrl_failure_reason = None
        return reason

    def _snapshot_process(self, process, name):
        return {
            "name": name,
            "running": self._is_process_running(process),
            "state": self._describe_process_state(process, name),
            "pid": None if process is None else process.pid,
        }

    def _run_subprocess(self, args, use_sudo=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False):
        command = list(args)
        input_text = None
        if use_sudo:
            command = ["sudo", "-S"] + command
            input_text = sudo_password + "\n"
        return subprocess.run(
            command,
            input=input_text,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=check,
        )

    def _check_port_open(self, port, timeout=0.2):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=timeout):
                return True
        except OSError:
            return False

    def _port_owner_details(self, port, use_sudo=False):
        result = self._run_subprocess(
            ["ss", "-ltnp"],
            use_sudo=use_sudo,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        details = []
        for line in result.stdout.splitlines():
            if not re.search(rf":{port}\b", line):
                continue
            details.append(
                {
                    "line": line.strip(),
                    "pids": sorted({int(pid) for pid in re.findall(r"pid=(\d+)", line)}),
                }
            )
        return details

    def _port_owner_summary(self, port, use_sudo=False):
        return [detail["line"] for detail in self._port_owner_details(port, use_sudo=use_sudo)]

    def _device_is_present(self, path):
        return os.path.exists(path)

    def _pid_exists(self, pid):
        if pid is None:
            return False
        try:
            os.kill(int(pid), 0)
            return True
        except PermissionError:
            return True
        except (ProcessLookupError, ValueError):
            return False

    def _get_process_group_id(self, pid):
        if pid is None:
            return None
        try:
            return os.getpgid(int(pid))
        except (ProcessLookupError, PermissionError, ValueError):
            return None

    def _get_process_command(self, pid):
        if pid is None:
            return ""
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        return result.stdout.strip()

    def _scan_known_controller_pids(self):
        result = subprocess.run(
            ["ps", "-eo", "pid=,command="],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        pids = set()
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                pid_text, command = line.split(None, 1)
            except ValueError:
                continue
            if any(marker in command for marker in KNOWN_CONTROLLER_COMMAND_MARKERS):
                pids.add(int(pid_text))
        return pids

    def _is_known_controller_pid(self, pid):
        command = self._get_process_command(pid)
        return any(marker in command for marker in KNOWN_CONTROLLER_COMMAND_MARKERS)

    def _load_controller_state(self):
        try:
            with open(CONTROLLER_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, OSError) as exc:
            logging.warning("Failed to read controller state file %s: %s", CONTROLLER_STATE_FILE, exc)
            return {}
        return data if isinstance(data, dict) else {}

    def _save_controller_state(self, state):
        temp_path = CONTROLLER_STATE_FILE + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        os.replace(temp_path, CONTROLLER_STATE_FILE)

    def _clear_controller_state(self):
        try:
            os.remove(CONTROLLER_STATE_FILE)
        except FileNotFoundError:
            pass

    def _capture_service_state(self, name, process, port):
        owner_details = self._port_owner_details(port, use_sudo=True)
        owner_pids = sorted(
            {
                pid
                for detail in owner_details
                for pid in detail["pids"]
                if self._pid_exists(pid)
            }
        )
        owner_pgids = sorted(
            {
                pgid
                for pgid in (self._get_process_group_id(pid) for pid in owner_pids)
                if pgid is not None
            }
        )
        launcher_pid = None
        if process is not None and self._is_process_running(process):
            launcher_pid = process.pid
        return {
            "name": name,
            "port": int(port),
            "launcher_pid": launcher_pid,
            "launcher_pgid": self._get_process_group_id(launcher_pid),
            "owner_pids": owner_pids,
            "owner_pgids": owner_pgids,
            "owner_lines": [detail["line"] for detail in owner_details],
            "updated_at": time.time(),
        }

    def _refresh_controller_state(self):
        services = {}
        if self._is_process_running(self._gripper_process) or self._check_port_open(GRIPPER_SERVER_PORT):
            services["gripper"] = self._capture_service_state(
                "Polymetis gripper service",
                self._gripper_process,
                GRIPPER_SERVER_PORT,
            )
        if self._is_process_running(self._robot_process) or self._check_port_open(ROBOT_SERVER_PORT):
            services["robot"] = self._capture_service_state(
                "Polymetis robot service",
                self._robot_process,
                ROBOT_SERVER_PORT,
            )

        if services:
            self._save_controller_state(
                {
                    "version": 1,
                    "server_pid": os.getpid(),
                    "updated_at": time.time(),
                    "services": services,
                }
            )
        else:
            self._clear_controller_state()

    def _collect_controller_targets(self):
        targets = {"pids": set(), "pgids": set()}

        def add_pid(pid):
            if pid is not None:
                targets["pids"].add(int(pid))
                pgid = self._get_process_group_id(pid)
                if pgid is not None:
                    targets["pgids"].add(int(pgid))

        def add_pgid(pgid):
            if pgid is not None:
                targets["pgids"].add(int(pgid))

        for process in (self._robot_process, self._gripper_process):
            if process is not None:
                add_pid(process.pid)

        recorded_services = self._load_controller_state().get("services", {})
        for service in recorded_services.values():
            add_pid(service.get("launcher_pid"))
            add_pgid(service.get("launcher_pgid"))
            for pid in service.get("owner_pids", []):
                add_pid(pid)
            for pgid in service.get("owner_pgids", []):
                add_pgid(pgid)

        for port in (ROBOT_SERVER_PORT, GRIPPER_SERVER_PORT):
            for detail in self._port_owner_details(port, use_sudo=True):
                for pid in detail["pids"]:
                    if recorded_services or self._is_known_controller_pid(pid):
                        add_pid(pid)

        for pid in self._scan_known_controller_pids():
            add_pid(pid)

        return targets

    def _terminate_pid(self, pid, timeout_s=PROCESS_SHUTDOWN_TIMEOUT_S):
        if not self._pid_exists(pid):
            return

        self._run_subprocess(
            ["kill", "-TERM", str(pid)],
            use_sudo=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if not self._pid_exists(pid):
                return
            time.sleep(0.1)

        self._run_subprocess(
            ["kill", "-KILL", str(pid)],
            use_sudo=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def _terminate_process_group(self, pgid):
        if pgid is None:
            return
        self._run_subprocess(
            ["kill", "-TERM", "--", f"-{pgid}"],
            use_sudo=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def _kill_process_group(self, pgid):
        if pgid is None:
            return
        self._run_subprocess(
            ["kill", "-KILL", "--", f"-{pgid}"],
            use_sudo=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def _cleanup_controller_processes(self, timeout_s=PROCESS_SHUTDOWN_TIMEOUT_S):
        targets = self._collect_controller_targets()
        if not targets["pids"] and not targets["pgids"]:
            self._clear_controller_state()
            return

        for pgid in sorted(targets["pgids"]):
            self._terminate_process_group(pgid)
        for pid in sorted(targets["pids"]):
            self._terminate_pid(pid, timeout_s=0.5)

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            remaining_pids = [pid for pid in targets["pids"] if self._pid_exists(pid)]
            if not remaining_pids:
                break
            time.sleep(0.1)

        remaining_pids = [pid for pid in targets["pids"] if self._pid_exists(pid)]
        if remaining_pids:
            remaining_pgids = {
                self._get_process_group_id(pid)
                for pid in remaining_pids
                if self._get_process_group_id(pid) is not None
            }
            for pgid in sorted(remaining_pgids):
                self._kill_process_group(pgid)
            for pid in remaining_pids:
                self._terminate_pid(pid, timeout_s=0.1)

        self._refresh_controller_state()

    def _build_preflight_report(self):
        report = {
            "status": str(self._status.value),
            "status_reason": self._status_reason,
            "ports": {
                SERVER_RPC_PORT: {
                    "open": self._check_port_open(SERVER_RPC_PORT),
                    "owner_lines": self._port_owner_summary(SERVER_RPC_PORT, use_sudo=True),
                },
                ROBOT_SERVER_PORT: {
                    "open": self._check_port_open(ROBOT_SERVER_PORT),
                    "owner_lines": self._port_owner_summary(ROBOT_SERVER_PORT, use_sudo=True),
                },
                GRIPPER_SERVER_PORT: {
                    "open": self._check_port_open(GRIPPER_SERVER_PORT),
                    "owner_lines": self._port_owner_summary(GRIPPER_SERVER_PORT, use_sudo=True),
                },
            },
            "device": {
                "ttyUSB0_present": self._device_is_present("/dev/ttyUSB0"),
            },
            "processes": {
                "robot": self._snapshot_process(self._robot_process, "Polymetis robot service"),
                "gripper": self._snapshot_process(self._gripper_process, "Polymetis gripper service"),
            },
        }
        self._last_preflight = report
        return report

    def get_controller_status(self):
        report = self._build_preflight_report()
        return report

    def preflight_check(self):
        report = self._build_preflight_report()
        errors = []
        if report["ports"][ROBOT_SERVER_PORT]["open"]:
            errors.append(f"Port {ROBOT_SERVER_PORT} is already in use.")
        if report["ports"][GRIPPER_SERVER_PORT]["open"]:
            errors.append(f"Port {GRIPPER_SERVER_PORT} is already in use.")
        if not report["device"]["ttyUSB0_present"]:
            errors.append("Gripper device /dev/ttyUSB0 is missing.")
        report["ready_to_start"] = len(errors) == 0
        report["errors"] = errors
        return report

    def _assert_ready(self, action):
        if self._status == ControllerStatus.READY:
            if not self._is_process_running(self._robot_process):
                self._set_status(
                    ControllerStatus.FAULTED,
                    self._describe_process_state(self._robot_process, "Polymetis robot service"),
                )
            elif not self._is_process_running(self._gripper_process):
                self._set_status(
                    ControllerStatus.FAULTED,
                    self._describe_process_state(self._gripper_process, "Polymetis gripper service"),
                )
            elif self._robot is None or self._gripper is None:
                self._set_status(
                    ControllerStatus.FAULTED,
                    "Controller marked ready without live robot and gripper interfaces.",
                )
        if self._status != ControllerStatus.READY:
            raise RuntimeError(
                f"Cannot {action} while controller status is '{self._status.value}': {self._status_reason}"
            )

    def _assert_ports_available_for_startup(self):
        conflicts = []
        for port, label in (
            (ROBOT_SERVER_PORT, "Polymetis robot service"),
            (GRIPPER_SERVER_PORT, "Polymetis gripper service"),
        ):
            if self._check_port_open(port):
                conflicts.append(
                    f"{label} port {port} is already occupied. Owners: {self._port_owner_summary(port, use_sudo=True)}"
                )

        if conflicts:
            raise RuntimeError(" ; ".join(conflicts))

    def _assert_gripper_device_ready(self):
        device_path = "/dev/ttyUSB0"
        if not self._device_is_present(device_path):
            raise RuntimeError(f"Required gripper device {device_path} is not present.")

    def _is_process_running(self, process):
        return process is not None and process.poll() is None

    def _describe_process_state(self, process, name):
        if process is None:
            return f"{name} has not been started."
        returncode = process.poll()
        if returncode is None:
            return f"{name} is still running."
        if returncode < 0:
            return f"{name} exited due to signal {-returncode}."
        return f"{name} exited with return code {returncode}."

    def _ensure_process_running(self, process, name):
        if not self._is_process_running(process):
            raise RuntimeError(self._describe_process_state(process, name))

    def _wait_for_port(self, port, process, name, timeout_s):
        deadline = time.time() + timeout_s
        last_error = None
        while time.time() < deadline:
            self._ensure_process_running(process, name)
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    return
            except OSError as exc:
                last_error = exc
                time.sleep(0.1)

        message = f"Timed out waiting for {name} on port {port}."
        if last_error is not None:
            message = f"{message} Last error: {last_error}"
        raise RuntimeError(message)

    def _wait_for_gripper_ready(self, timeout_s):
        deadline = time.time() + timeout_s
        last_error = None
        empty = polymetis_pb2.Empty()

        while time.time() < deadline:
            self._ensure_process_running(self._gripper_process, "Polymetis gripper service")
            try:
                channel = grpc.insecure_channel(f"localhost:{GRIPPER_SERVER_PORT}")
                stub = polymetis_pb2_grpc.GripperServerStub(channel)
                metadata = stub.GetRobotClientMetadata(empty)
                state = stub.GetState(empty)
                channel.close()
                if getattr(metadata, "max_width", 0.0) <= 0.0:
                    raise RuntimeError("Gripper metadata is not initialized yet.")
                if state.timestamp.seconds == 0 and state.timestamp.nanos == 0:
                    raise RuntimeError("Gripper state has not been published yet.")
                return
            except Exception as exc:
                last_error = exc
                time.sleep(0.2)

        message = "Timed out waiting for Polymetis gripper service to accept state queries."
        if last_error is not None:
            message = f"{message} Last error: {last_error}"
        raise RuntimeError(message)

    def _wait_for_robot_ready(self, timeout_s):
        deadline = time.time() + timeout_s
        last_error = None

        while time.time() < deadline:
            self._ensure_process_running(self._robot_process, "Polymetis robot service")
            try:
                robot = RobotInterface(ip_address="localhost", port=ROBOT_SERVER_PORT)
                robot.get_robot_state()
                return
            except Exception as exc:
                last_error = exc
                time.sleep(0.2)

        message = "Timed out waiting for Polymetis robot service to accept robot-state queries."
        if last_error is not None:
            message = f"{message} Last error: {last_error}"
        raise RuntimeError(message)

    def _terminate_process(self, process, name, timeout_s=PROCESS_SHUTDOWN_TIMEOUT_S):
        if process is None:
            return
        if not self._is_process_running(process):
            return

        pgid = None
        try:
            pgid = os.getpgid(process.pid)
        except ProcessLookupError:
            pass

        try:
            if pgid is not None:
                subprocess.run(
                    ["sudo", "-S", "kill", "-TERM", f"-{pgid}"],
                    input=sudo_password + "\n",
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                process.terminate()
            process.wait(timeout=timeout_s)
            return
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass

        try:
            if pgid is not None:
                subprocess.run(
                    ["sudo", "-S", "kill", "-KILL", f"-{pgid}"],
                    input=sudo_password + "\n",
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                process.kill()
            process.wait(timeout=1.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            logging.warning("Failed to terminate %s cleanly.", name)

    def launch_controller(self, startup_timeout_s=CONTROLLER_STARTUP_TIMEOUT_S):
        print("enter launch control")
        with self._controller_lock:
            if self._launch_in_progress:
                raise RuntimeError("Controller launch already in progress.")
            self._launch_in_progress = True
        self._set_status(ControllerStatus.STARTING, "Running startup preflight checks.")
        dir_path = os.path.dirname(os.path.realpath(__file__))
        failure_status = None
        failure_reason = None
        try:
            self.kill_controller()
            try:
                self._assert_gripper_device_ready()
                self._assert_ports_available_for_startup()
            except Exception as exc:
                failure_status = ControllerStatus.PREFLIGHT_FAILED
                failure_reason = f"Startup preflight failed: {exc}"
                raise
            self._set_status(ControllerStatus.STARTING, "Starting Polymetis gripper service.")
            self._gripper_process = run_terminal_command(
                "echo " + sudo_password + " | sudo -S " + "bash " + dir_path + "/launch_gripper.sh",
                start_new_session=True,
            )
            self._wait_for_gripper_ready(timeout_s=startup_timeout_s)
            self._refresh_controller_state()
            self._set_status(ControllerStatus.STARTING, "Starting Polymetis robot service.")
            self._robot_process = run_terminal_command(
                "echo " + sudo_password + " | sudo -S " + "bash " + dir_path + "/launch_robot.sh",
                start_new_session=True,
            )
            self._wait_for_robot_ready(timeout_s=startup_timeout_s)
            self._refresh_controller_state()
            self._ensure_process_running(self._gripper_process, "Polymetis gripper service")
            self._ensure_process_running(self._robot_process, "Polymetis robot service")
            self._server_launched = True
            self._set_status(
                ControllerStatus.STARTING,
                "Polymetis services started. Waiting for robot interface connection.",
            )
        except Exception:
            self.kill_controller()
            if failure_status is None:
                failure_status = ControllerStatus.FAULTED
                failure_reason = "Controller startup failed during service launch."
            self._set_status(failure_status, failure_reason)
            raise
        finally:
            with self._controller_lock:
                self._launch_in_progress = False

    def get_gripper_max_width(gripper, default=DEFAULT_FRANKA_GRIPPER_MAX_WIDTH):
        metadata = getattr(gripper, "metadata", None)
        max_width = getattr(metadata, "max_width", None)
        if max_width is None or max_width <= 0:
            logging.warning(
                "Franka gripper metadata unavailable; using fallback max_width=%.3f m", DEFAULT_FRANKA_GRIPPER_MAX_WIDTH)
            max_width = DEFAULT_FRANKA_GRIPPER_MAX_WIDTH
        return max_width


    def launch_robot(self, connection_timeout_s=5.0):
        if self._status == ControllerStatus.IDLE:
            raise RuntimeError("Controller has not been launched. Call launch_controller() first.")
        self._ensure_process_running(self._robot_process, "Polymetis robot service")
        self._ensure_process_running(self._gripper_process, "Polymetis gripper service")
        self._wait_for_gripper_ready(timeout_s=connection_timeout_s)
        self._wait_for_robot_ready(timeout_s=connection_timeout_s)

        try:
            self._robot = RobotInterface(ip_address="localhost")
            self._gripper = GripperInterface(ip_address="localhost")
        except grpc.RpcError as exc:
            robot_state = self._describe_process_state(self._robot_process, "Polymetis robot service")
            gripper_state = self._describe_process_state(self._gripper_process, "Polymetis gripper service")
            self._set_status(ControllerStatus.FAULTED, "Failed to connect to Polymetis interfaces.")
            raise RuntimeError(
                "Failed to connect to Polymetis interfaces. "
                + robot_state
                + " "
                + gripper_state
            ) from exc

        self._max_gripper_width = self.get_gripper_max_width(self._gripper)
        if self._ik_solver is None:
            self._ik_solver = RobotIKSolver()
        self._controller_not_loaded = False
        self._joint_impedance_active = False  # True only when joint impedance is the active Polymetis controller
        self._traj_ctrl = None  # HighFreqController instance (set by start_trajectory_controller)
        self._set_status(ControllerStatus.READY, "Robot and gripper interfaces connected and healthy.")

    def kill_controller(self):
        if self._traj_ctrl is not None and self._traj_ctrl.is_alive():
            self._traj_ctrl.stop()
            self._traj_ctrl.join(timeout=2.0)
        self._cleanup_controller_processes()
        self._robot_process = None
        self._gripper_process = None
        self._robot = None
        self._gripper = None
        self._server_launched = False
        self._joint_impedance_active = False
        self._controller_not_loaded = False
        self._traj_ctrl = None
        self._set_status(ControllerStatus.IDLE, "Controller stopped.")

    def update_command(self, command, action_space="cartesian_velocity", gripper_action_space=None, blocking=False):
        self._assert_ready("send robot commands")
        action_dict = self.create_action_dict(command, action_space=action_space, gripper_action_space=gripper_action_space)

        self.update_joints(action_dict["joint_position"], velocity=False, blocking=blocking)
        self.update_gripper(action_dict["gripper_position"], velocity=False, blocking=blocking)

        return action_dict

    def update_pose(self, command, velocity=False, blocking=False):
        self._assert_ready("update robot pose")
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

    def _assert_no_trajectory_controller_locked(self, action):
        if self._traj_ctrl is not None and self._traj_ctrl.is_alive():
            raise RuntimeError(
                f"Cannot {action} while the high-frequency trajectory controller is running."
            )

    def _wait_for_running_policy_locked(self, timeout_s=5.0, context="controller"):
        deadline = time.time() + timeout_s
        while not self._robot.is_running_policy():
            if time.time() > deadline:
                raise RuntimeError(f"Timed out waiting for {context} to start.")
            time.sleep(0.01)

    def _terminate_active_policy_locked(self, reason):
        if self._robot.is_running_policy():
            logging.info("Terminating active Polymetis policy (%s).", reason)
            self._robot.terminate_current_policy()
        self._joint_impedance_active = False

    def _probe_joint_impedance_locked(self, desired_joints=None):
        if desired_joints is None:
            desired_joints = self._robot.get_joint_positions()
        if not torch.is_tensor(desired_joints):
            desired_joints = torch.Tensor(desired_joints)
        self._robot.update_desired_joint_positions(desired_joints)

    def _ensure_joint_impedance_ready_locked(self, probe_joint_positions=None, timeout_s=5.0):
        policy_running = self._robot.is_running_policy()
        if (not policy_running) or (not self._joint_impedance_active):
            self._controller_not_loaded = True
            self._terminate_active_policy_locked("preparing for joint-target streaming")
            logging.info("Starting joint impedance controller.")
            self._robot.start_joint_impedance()
            self._wait_for_running_policy_locked(
                timeout_s=timeout_s,
                context="joint impedance controller",
            )

        try:
            self._probe_joint_impedance_locked(desired_joints=probe_joint_positions)
        except grpc.RpcError as exc:
            self._joint_impedance_active = False
            raise RuntimeError(
                "Joint impedance probe failed. Polymetis rejected desired joint positions."
            ) from exc

        self._joint_impedance_active = True
        self._controller_not_loaded = False

    def _stop_trajectory_controller_locked(self, join_timeout_s=2.0):
        if self._traj_ctrl is not None:
            if self._traj_ctrl.is_alive():
                self._traj_ctrl.stop()
                self._traj_ctrl.join(timeout=join_timeout_s)
                if self._traj_ctrl.is_alive():
                    raise RuntimeError(
                        "Timed out waiting for the high-frequency trajectory controller to stop."
                    )
            failure_reason = self._traj_ctrl.get_failure_reason()
            if failure_reason:
                self._record_trajectory_controller_failure_locked(failure_reason)
            self._traj_ctrl = None

        self._terminate_active_policy_locked("stopping trajectory controller")
        self._controller_not_loaded = False

    def prepare_for_streaming(self, timeout_s=5.0):
        """Synchronously prepare Polymetis for joint-target streaming.

        Stops any stale trajectory controller, clears old policies, starts joint
        impedance, and probes that update_desired_joint_positions() is accepted.

        zerorpc-exposed: called by GPU-server via ServerInterface.
        """
        self._assert_ready("prepare for streaming")
        with self._controller_lock:
            if self._traj_ctrl is not None:
                self._stop_trajectory_controller_locked()
            current_joints = self._robot.get_joint_positions()
            self._ensure_joint_impedance_ready_locked(
                probe_joint_positions=current_joints,
                timeout_s=timeout_s,
            )

    def update_joints(self, command, velocity=False, blocking=False, cartesian_noise=None):
        self._assert_ready("update joints")
        if cartesian_noise is not None:
            command = self.add_noise_to_joints(command, cartesian_noise)
        command = torch.Tensor(command)

        if velocity:
            joint_delta = self._ik_solver.joint_velocity_to_delta(command)
            command = joint_delta + self._robot.get_joint_positions()

        def helper_non_blocking():
            try:
                with self._controller_lock:
                    self._assert_no_trajectory_controller_locked("send direct joint targets")
                    policy_running = self._robot.is_running_policy()
                    if (not policy_running) or (not self._joint_impedance_active):
                        self._ensure_joint_impedance_ready_locked(
                            probe_joint_positions=command,
                        )
                        return
                    try:
                        self._robot.update_desired_joint_positions(command)
                    except grpc.RpcError:
                        logging.warning(
                            "Desired joint update rejected; restarting joint impedance controller."
                        )
                        self._joint_impedance_active = False
                        self._ensure_joint_impedance_ready_locked(
                            probe_joint_positions=command,
                        )
            except Exception:
                logging.exception(
                    "FrankaRobot.update_joints: failed to send non-blocking joint target"
                )
            finally:
                self._controller_not_loaded = False

        if blocking:
            with self._controller_lock:
                self._assert_no_trajectory_controller_locked("run a blocking joint move")
                self._terminate_active_policy_locked("starting blocking joint move")
                try:
                    time_to_go = self.adaptive_time_to_go(command)
                    self._robot.move_to_joint_positions(command, time_to_go=time_to_go)
                except grpc.RpcError as exc:
                    raise RuntimeError(
                        "Blocking joint move failed while moving to the requested target."
                    ) from exc
                finally:
                    self._terminate_active_policy_locked("finishing blocking joint move")
                    self._controller_not_loaded = False
        else:
            if not self._controller_not_loaded:
                run_threaded_command(helper_non_blocking)

    def update_gripper(self, command, velocity=True, blocking=False):
        self._assert_ready("update gripper")
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
        self._assert_ready("read joint positions")
        return self._robot.get_joint_positions().tolist()

    def get_joint_velocities(self):
        self._assert_ready("read joint velocities")
        return self._robot.get_joint_velocities().tolist()

    def get_gripper_position(self):
        self._assert_ready("read gripper position")
        return 1 - (self._gripper.get_state().width / self._max_gripper_width)

    def get_ee_pose(self):
        self._assert_ready("read end-effector pose")
        pos, quat = self._robot.get_ee_pose()
        angle = quat_to_euler(quat.numpy())
        return np.concatenate([pos, angle]).tolist()

    def get_robot_state(self):
        self._assert_ready("read robot state")
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
        self._assert_ready("estimate motion timing")
        curr_joint_position = self._robot.get_joint_positions()
        displacement = desired_joint_position - curr_joint_position
        time_to_go = self._robot._adaptive_time_to_go(displacement)
        clamped_time_to_go = min(t_max, max(time_to_go, t_min))
        return clamped_time_to_go

    # ── High-frequency trajectory controller ──────────────────────────────────

    def start_trajectory_controller(self, frequency=DEFAULT_TRAJECTORY_CONTROLLER_FREQUENCY_HZ):
        """Start the high-frequency joint position + state-logging controller.

        This method is self-sufficient: it synchronously prepares Polymetis for
        joint-target streaming before spawning HighFreqController.

        zerorpc-exposed: called by GPU-server via ServerInterface.
        """
        self._assert_ready("start the trajectory controller")
        from droid.franka.trajectory_controller import HighFreqController
        with self._controller_lock:
            if self._traj_ctrl is not None and self._traj_ctrl.is_alive():
                return  # already running
            if self._traj_ctrl is not None and not self._traj_ctrl.is_alive():
                failure_reason = self._traj_ctrl.get_failure_reason()
                if failure_reason:
                    self._record_trajectory_controller_failure_locked(failure_reason)
                self._traj_ctrl = None

            self._traj_ctrl_failure_reason = None
            self.prepare_for_streaming()
            self._traj_ctrl = HighFreqController(
                self._robot,
                self._gripper,
                frequency=float(frequency),
                on_fatal_error=self._record_trajectory_controller_failure_locked,
            )
            self._traj_ctrl.start()

    def stop_trajectory_controller(self):
        """Stop the high-frequency controller. Called at episode end.

        zerorpc-exposed: called by GPU-server via ServerInterface.
        """
        with self._controller_lock:
            if self._status != ControllerStatus.READY:
                if self._traj_ctrl is not None and self._traj_ctrl.is_alive():
                    self._traj_ctrl.stop()
                    self._traj_ctrl.join(timeout=2.0)
                if self._traj_ctrl is not None:
                    failure_reason = self._traj_ctrl.get_failure_reason()
                    if failure_reason:
                        self._record_trajectory_controller_failure_locked(failure_reason)
                self._traj_ctrl = None
                self._joint_impedance_active = False
                self._controller_not_loaded = False
                logging.info(
                    "Ignoring stop_trajectory_controller while controller status is '%s'.",
                    self._status.value,
                )
                return
            self._stop_trajectory_controller_locked()

    def get_state_history(self, n=100):
        """Return (timestamps, joints_list, gripper_list) from the HighFreqController
        state ring buffer.

        Used by the GPU server to retrieve a high-frequency proprioception history
        for UMI-style interpolation to the camera observation timestamp.

        zerorpc-exposed: returns plain Python lists (no numpy) for msgpack compat.
        Returns three empty lists if the trajectory controller is not running.
        """
        self._assert_ready("read state history")
        with self._controller_lock:
            if self._traj_ctrl is not None and not self._traj_ctrl.is_alive():
                failure_reason = self._traj_ctrl.get_failure_reason()
                if failure_reason:
                    self._record_trajectory_controller_failure_locked(failure_reason)
                self._traj_ctrl = None
        if self._traj_ctrl is None or not self._traj_ctrl.is_alive():
            return [], [], []
        return self._traj_ctrl.get_state_history(int(n))

    def add_waypoints(self, times_list, positions_list, max_joint_speed_rad_s=0.5):
        """Send a batch of arm waypoints to the trajectory controller.

        Args:
            times_list: list[float] — time offsets from caller's time.time(),
                        already compensated for robot_action_latency by caller.
            positions_list: list[list[float]] — shape (N, 7), absolute joint angles.
            max_joint_speed_rad_s: per-joint speed cap (rad/s). Default 0.5
                (conservative). Pass a higher value (e.g. 3.0) via CLI config
                for faster execution. Forwarded to HighFreqController.

        zerorpc-exposed: called ~10 Hz from GPU-server policy loop.
        Lists and float are used (not numpy) because msgpack serialises them natively.
        """
        self._assert_ready("add waypoints")
        with self._controller_lock:
            if self._traj_ctrl is not None and not self._traj_ctrl.is_alive():
                failure_reason = self._traj_ctrl.get_failure_reason()
                if failure_reason:
                    self._record_trajectory_controller_failure_locked(failure_reason)
                self._traj_ctrl = None
            if self._traj_ctrl is None or not self._traj_ctrl.is_alive():
                message = "Trajectory controller not running. Call start_trajectory_controller() first."
                failure_reason = self._consume_trajectory_controller_failure_reason_locked()
                if failure_reason:
                    message = f"{message} Last failure: {failure_reason}"
                raise RuntimeError(message)
            self._traj_ctrl.add_waypoints(
                np.array(times_list, dtype=np.float64),
                np.array(positions_list, dtype=np.float64),
                max_joint_speed_rad_s=float(max_joint_speed_rad_s),
            )

    def create_action_dict(self, action, action_space, gripper_action_space=None, robot_state=None):
        self._assert_ready("create an action dict")
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
