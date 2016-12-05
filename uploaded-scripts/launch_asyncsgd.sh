#!/bin/bash


source ~/gitRepository/TensorFlow/uploaded-scripts/tfdefs.sh
terminate_cluster
pdsh -R ssh -w vm-14-[1-5] "killall python"

type=async

start_cluster startserver.py

pdsh -R ssh -w vm-14-[1-5] "mkdir -p ~/gitRepository/TensorFlow/uploaded-scripts/results/async;"
pdsh -R ssh -w vm-14-[1-5] "rm ~/gitRepository/TensorFlow/uploaded-scripts/results/async/dstat_res;"
sleep 5

pdsh -R ssh -w vm-14-[1-5] "ps -ef | grep dstat | awk '{print \$2}' | xargs kill"
sleep 10

pdsh -R ssh -w vm-14-[1-5] 'sudo -S sh -c "sync; echo 3 > /proc/sys/vm/drop_caches";'
sleep 2
echo "Dropped caches.."

echo "Starting dstat"
pdsh -R ssh -w vm-14-[1-5] "dstat -tcmsdn 30 > ~/gitRepository/TensorFlow/uploaded-scripts/results/async/dstat_res;" &
sleep 2

echo "copying over asyncsgd.py"
pdsh -R ssh -w vm-14-[2-5] "scp vm-14-1:~/gitRepository/TensorFlow/uploaded-scripts/asyncsgd.py ~/gitRepository/TensorFlow/uploaded-scripts/"

echo "Executing the distributed tensorflow job"

# start multiple clients
echo "start client 0"
python asyncsgd.py --task_index=0 &
echo "start other clients"
sleep 5 # wait for variable to be initialized
pdsh -R ssh -w vm-14-2 'cd ~/gitRepository/TensorFlow/uploaded-scripts; python asyncsgd.py --task_index=1' &
pdsh -R ssh -w vm-14-3 'cd ~/gitRepository/TensorFlow/uploaded-scripts; python asyncsgd.py --task_index=2' &
pdsh -R ssh -w vm-14-4 'cd ~/gitRepository/TensorFlow/uploaded-scripts; python asyncsgd.py --task_index=3' &
pdsh -R ssh -w vm-14-5 'cd ~/gitRepository/TensorFlow/uploaded-scripts; python asyncsgd.py --task_index=4'
#nohup python asyncsgd.py --task_index=2 > asynclog-2.out 2>&1&
#nohup python asyncsgd.py --task_index=3 > asynclog-3.out 2>&1&
#python asyncsgd.py --task_index=4

echo "terminate cluster"
terminate_cluster

echo "Ending dstats"
pdsh -R ssh -w vm-14-[1-5] "ps -ef | grep dstat | awk '{print \$2}' | xargs kill"
sleep 10

scp vm-14-2:~/gitRepository/TensorFlow/uploaded-scripts/results/async/dstat_res ~/gitRepository/TensorFlow/uploaded-scripts/results/async/dstat_res2
scp vm-14-3:~/gitRepository/TensorFlow/uploaded-scripts/results/async/dstat_res ~/gitRepository/TensorFlow/uploaded-scripts/results/async/dstat_res3
scp vm-14-4:~/gitRepository/TensorFlow/uploaded-scripts/results/async/dstat_res ~/gitRepository/TensorFlow/uploaded-scripts/results/async/dstat_res4
scp vm-14-5:~/gitRepository/TensorFlow/uploaded-scripts/results/async/dstat_res ~/gitRepository/TensorFlow/uploaded-scripts/results/async/dstat_res5
