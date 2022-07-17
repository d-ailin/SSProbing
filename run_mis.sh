
python -u mis_detect.py -c snapshots/cifar10_resnet18/cifar10_resnet18_dp_baseline_epoch_299.pt -m mcp -t ./task_configs/rot4_trans5.yaml -sf cifar10_mis_res.txt -se 5
python -u mis_detect.py -c snapshots/cifar10_vgg16/config_1.yaml -m mcp -t ./task_configs/rot4_trans5.yaml -sf cifar10_mis_res.txt  -se 5

python -u mis_detect.py -c snapshots/cinic10_resnet18/cinic10_resnet18_dp_baseline_epoch_299.pt -m mcp -t ./task_configs/rot4_trans5.yaml -sf cinic10_mis_res.txt -se 5
python -u mis_detect.py -c snapshots/cinic10_vgg16/config_1.yaml -e 300 -m mcp -t ./task_configs/rot4_trans5.yaml -sf cinic10_mis_res.txt -se 5

python -u mis_detect.py -c snapshots/stl10_resnet18/stl10_resnet18_dp_baseline_epoch_99.pt -m mcp -t ./task_configs/rot2_trans3_stl.yaml -sf stl10_mis_res.txt -se 5
