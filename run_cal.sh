
python -u cal.py -c snapshots/cifar10_vgg16/config_1.yaml -sf cifar10_cal_res.txt
python -u cal.py -c snapshots/cifar10_resnet18/cifar10_resnet18_dp_baseline_epoch_299.pt -sf cifar10_cal_res.txt
python -u cal.py -c snapshots/stl10_resnet18/stl10_resnet18_dp_baseline_epoch_99.pt -sf stl10_cal_res.txt
python -u cal.py -c snapshots/cinic10_vgg16/config_1.yaml -sf cinic10_cal_res.txt
python -u cal.py -c snapshots/cinic10_resnet18/cinic10_resnet18_dp_baseline_epoch_299.pt -sf cinic10_cal_res.txt
