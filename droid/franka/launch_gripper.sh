source /home/robot/anaconda3/etc/profile.d/conda.sh
conda activate polymetis-local
pkill -9 gripper
chmod a+rw /dev/ttyUSB0
launch_gripper.py ip=0.0.0.0 port=50052 gripper.server_ip=127.0.0.1 gripper.server_port=50052 gripper=robotiq_2f gripper.comport=/dev/ttyUSB0
