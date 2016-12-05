#!/bin/bash

# tfdefs.sh has helper function to start process on all VMs
# it contains definition for start_cluster and terminate_cluster
source ~/gitRepository/TensorFlow/uploaded-scripts/tfdefs.sh
terminate_cluster
pdsh -R ssh -w vm-14-[1-5] "killall python"

type=sync

rm out_runSYN_sync

export TF_LOG_DIR="/home/ubuntu/tf/logs"

start_cluster startserver.py
pdsh -R ssh -w vm-14-[1-5] "mkdir -p ~/gitRepository/TensorFlow/uploaded-scripts/results/${type};"
pdsh -R ssh -w vm-14-[1-5] "rm ~/gitRepository/TensorFlow/uploaded-scripts/results/${type}/dstat_res;"
sleep 5

pdsh -R ssh -w vm-14-[1-5] "ps -ef | grep dstat | awk '{print \$2}' | xargs kill"
sleep 10

pdsh -R ssh -w vm-14-[1-5] 'sudo -S sh -c "sync; echo 3 > /proc/sys/vm/drop_caches";'
sleep 2
echo "Dropped caches.."

echo "Starting dstat"
pdsh -R ssh -w vm-14-[1-5] "dstat -tcmsdn 30 > ~/gitRepository/TensorFlow/uploaded-scripts/results/${type}/dstat_res;" &
sleep 2

echo "Executing the distributed tensorflow job"
# testdistributed.py is a client that can run jobs on the cluster.
# please read testdistributed.py to understand the steps defining a Graph and
# launch a session to run the Graph
python synchronoussgd.py

# defined in tfdefs.sh to terminate the cluster
terminate_cluster

pdsh -R ssh -w vm-14-[1-5] "ps -ef | grep dstat | awk '{print \$2}' | xargs kill"
sleep 10

scp vm-14-2:~/gitRepository/TensorFlow/uploaded-scripts/results/${type}/dstat_res ~/gitRepository/TensorFlow/uploaded-scripts/results/${type}/dstat_res2
scp vm-14-3:~/gitRepository/TensorFlow/uploaded-scripts/results/${type}/dstat_res ~/gitRepository/TensorFlow/uploaded-scripts/results/${type}/dstat_res3
scp vm-14-4:~/gitRepository/TensorFlow/uploaded-scripts/results/${type}/dstat_res ~/gitRepository/TensorFlow/uploaded-scripts/results/${type}/dstat_res4
scp vm-14-5:~/gitRepository/TensorFlow/uploaded-scripts/results/${type}/dstat_res ~/gitRepository/TensorFlow/uploaded-scripts/results/${type}/dstat_res5


# TODO: Uncomment this for printing
#tensorboard --logdir=$TF_LOG_DIR