cd ~/gitRepository/TensorFlow/uploaded-scripts/results/

echo "VM1"
python dstat_parser.py dstat_res

echo "VM2"
python dstat_parser.py dstat_res2

echo "VM3"
python dstat_parser.py dstat_res3

echo "VM4"
python dstat_parser.py dstat_res4

echo "VM5"
python dstat_parser.py dstat_res5
