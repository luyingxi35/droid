import zerorpc

from droid.franka.robot import FrankaRobot

if __name__ == "__main__":
    robot_client = FrankaRobot()
    s = zerorpc.Server(robot_client)
    try:
        s.bind("tcp://0.0.0.0:4242")
        print("Start listening on tcp://0.0.0.0:4242...")
        s.run()
    finally:
        try:
            robot_client.kill_controller()
        except Exception:
            pass
