import multiprocessing
import subprocess
import threading


def run_terminal_command(command, stdout=None, stderr=None, start_new_session=False):
    process = subprocess.Popen(
        command,
        stdout=stdout,
        stderr=stderr,
        stdin=subprocess.DEVNULL,
        shell=True,
        executable="/bin/bash",
        encoding="utf8",
        start_new_session=start_new_session,
    )

    return process


def run_threaded_command(command, args=(), daemon=True):
    thread = threading.Thread(target=command, args=args, daemon=daemon)
    thread.start()

    return thread


def run_multiprocessed_command(command, args=()):
    process = multiprocessing.Process(target=command, args=args)
    process.start()

    return process
