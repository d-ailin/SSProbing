# SSProbing

Code for the paper: [Trust, but Verify: Using Self-Supervised Probing to Improve Trustworthiness (ECCV'22)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730362.pdf)

## Demo Visualization
We have provided a notebook `vis_demo.ipynb`[(link)](https://github.com/d-ailin/SSProbing/blob/main/vis_demo.ipynb) for a demo visualization of self-supervised probing confidence scores.

## Environment Setup
Setup environment and install with the packages:

* Python==3.6.13
* torch==1.10.1
* torchvision==0.11.2


Our environment is partly based on the [ConfidNet](https://github.com/valeoai/ConfidNet).
```
$ git clone https://github.com/valeoai/ConfidNet
$ pip install -e ConfidNet
```
After installing above packages, install packages in the requirements.txt
```
    pip install -r requirements.txt
```
The Juypter Lab is not installed in the requirements. If you would like to use it, you could install it manually.

## Data
All the datasets should be placed under data/ directory.
```
    mkdir data/
```
* CIFAR-10: can be downloaded automatically from torchvision.datasets
* CINIC-10: [https://github.com/BayesWatch/cinic-10](https://github.com/BayesWatch/cinic-10)
* STL-10: [https://cs.stanford.edu/~acoates/stl10](https://cs.stanford.edu/~acoates/stl10)

OOD related datasets:
* SVHN: can be downloaded automatically from torchvision.datasets
* LSUN_resize, ImageNet_resize, LSUN_fix, ImageNet_fix: refer to [CSI](https://github.com/alinlab/CSI).
    * download links:  [LSUN_resize](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz), [ImageNet_resize](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz), [LSUN_fix](https://drive.google.com/file/d/1KVWj9xpHfVwGcErH5huVujk9snhEGOxE/view?usp=sharing), [ImageNet_fix](https://drive.google.com/file/d/1sO_-noq10mmziB1ECDyNhD5T4u5otyKA/view?usp=sharing)

## Pre-trained models
The pre-trained model weights we used in the paper can be accessed via this [link](https://drive.google.com/file/d/114fDcXJqBo3t4nUOv0-yK7OoDhhJp_Lr/view?usp=sharing). 

## Training the base model
If you perfer to train a new base model and test it out, you could follow the following guidance.

As we use MCDropout as one of our baselines, the base models require Dropout components. In detail, the model implementation can refer to [VGG16](https://github.com/valeoai/ConfidNet/blob/master/confidnet/models/vgg16.py) and our model file under the `train_base/models/`. 

For the VGG16 model training, please refer to [ConfidNet](https://github.com/valeoai/ConfidNet).

For the resnet18 model training, you could follow the below example:
```
    $ cd train_base
    $ python train_cifar10.py -e 300
```
After the training, the models should be saved in `train_base/snapshots/xxx`



## Applying SSProbing on the pre-trained models
The following commands or examples are applicable if you have downloaded the pre-trained models and unzipped it in the repo directory. Otherwise, you could slightly modify the commands according your actual demand, e.g. the config path argument to your actual saved or trained model path. 

```
    # create the res output directory
    mkdir res_dir/
```

### Misclassification Detection
Example:

```
    python -u mis_detect.py -c snapshots/cifar10_resnet18/cifar10_resnet18_dp_baseline_epoch_299.pt -m mcp -t ./task_configs/rot4_trans5.yaml -sf cifar10_mis_res.txt -se 5
```
* `-c`: a trained model path
* `-m`: the baseline method, options:['mcp', 'mcdropout', 'trustscore', 'tcp']
* `-t`: task config path, set the probing task and details
* `-sf`: the result output filename, the file will be under `res_dir` after the successful execution
* `-se`: self-supervised probing task training epoch number


### OOD Detection
Example:
```
    python -u ood_detect.py -c snapshots/cifar10_resnet18/cifar10_resnet18_dp_baseline_epoch_299.pt -m mcp -se 5 -sf cifar10_ood_res.text
```
* `-c`: a trained model path
* `-m`: the baseline method, options:['mcp', 'entropy']
* `-sf`: the result output filename, the file will be under `res_dir` after the successful execution
* `-se`: self-supervised probing task training epoch number

### Calibration
Example:
```
    python -u cal.py -c snapshots/cifar10_resnet18/cifar10_resnet18_dp_baseline_epoch_299.pt -sf cifar10_cal_res.txt
```
* `-c`: a trained model path
* `-sf`: the result output filename, the file will be under `res_dir` after the successful execution

For more details, you could also refer to `run_mis.sh / run_ood.sh / run_cal.sh`.


## Citation
If you find this repo or our work useful for your research, please consider citing the paper
```
@inproceedings{deng2022trust,
  title={Trust, but Verify: Using Self-supervised Probing to Improve Trustworthiness},
  author={Deng, Ailin and Li, Shen and Xiong, Miao and Chen, Zhirui and Hooi, Bryan},
  booktitle={European Conference on Computer Vision},
  pages={361--377},
  year={2022},
  organization={Springer}
}
```


## Acknowledgement
* Part of the base models are from or modified based on [ConfidNet](https://github.com/valeoai/ConfidNet), [ResNet](https://github.com/weiaicunzai/pytorch-cifar100), [OE](https://github.com/hendrycks/outlier-exposure).
* Part of baseline methods or evaluation code are from or modified based on [TrustScore](https://github.com/google/TrustScore), [NN_calibration](https://github.com/markus93/NN_calibration).
* Part of Implementaion is modified based on [SS-OOD](https://github.com/hendrycks/ss-ood).
