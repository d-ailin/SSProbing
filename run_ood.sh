python -u ood_detect.py -c snapshots/cifar10_vgg16/config_1.yaml -m mcp -se 5 -sf cifar10_ood_res.txt
python -u ood_detect.py -c snapshots/cifar10_resnet18/cifar10_resnet18_dp_baseline_epoch_299.pt -m mcp -se 5 -sf cifar10_ood_res.txt
python -u ood_detect.py -c snapshots/cinic10_vgg16/config_1.yaml -m mcp -se 5 -sf cinic10_ood_res.txt
python -u ood_detect.py -c snapshots/cinic10_resnet18/cinic10_resnet18_dp_baseline_epoch_299.pt -m mcp -se 5 cinic10_ood_res.txt