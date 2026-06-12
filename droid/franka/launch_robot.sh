source /home/robot/anaconda3/etc/profile.d/conda.sh
conda activate polymetis-local
launch_robot.py ip=0.0.0.0 port=50051 robot_client=franka_hardware robot_client.executable_cfg.control_ip=127.0.0.1 robot_client.executable_cfg.control_port=50051
