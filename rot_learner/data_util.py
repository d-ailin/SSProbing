import copy
import os
import torch
from pathlib import Path


from train_base.models.resnet_dp import resnet18_dp
import torch.nn as nn
import torchvision.transforms as trn
import numpy as np
import torchvision.datasets as dset
from torch.utils.data.sampler import SubsetRandomSampler
from types import SimpleNamespace
import torchvision
from tqdm import tqdm

def entropy(scores):
    return -(scores * np.log(scores + 1e-6)).sum(-1)

def retrieve_data(dataset_name, dataset, indices=[]):
    if dataset_name == 'cifar10':
        return dataset.data, dataset.targets
    if dataset_name == 'stl10':
        return dataset.data, dataset.labels
    if dataset_name in ['cinic10']:
        if len(indices) <= 0:
            imgs = np.array([ np.asarray(dataset.loader(sample[0])) for sample in dataset.samples])
            targets = dataset.targets
        else:
            imgs = np.array([ np.asarray(dataset.loader(sample[0])) for i, sample in enumerate(tqdm(dataset.samples)) if i in indices])
            targets = np.array(dataset.targets)[indices].tolist()

        return imgs, targets

import yaml
def process_task_config(config_path):
    with open(config_path, "r") as f:
        config_args = yaml.load(f, Loader=yaml.SafeLoader)

    return config_args


def preprocess_config(config_path, args, device):

    # process config from confidnet
    if os.path.exists(config_path) and '.yaml' in config_path:
        from confidnet.models import get_model
        from confidnet.utils.misc import load_yaml
        from confidnet.loaders import get_loader

        config_args = load_yaml(config_path)
        # print('config_args', config_args)


        config_args["training"]["metrics"] = [
                "accuracy",
                "auc",
                "ap_success",
                "ap_errors",
                "fpr_at_95tpr",
                "aurc"
            ]
        dataset = config_args['data']['dataset']

        model = get_model(config_args, device).to(device)
        # ckpt_path = config_args["training"]["output_folder"] / f"model.ckpt"
        ckpt_path = Path(config_path).resolve().parent / 'model.ckpt'
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        config_args["model"]["name"] = config_args["model"]["name"] + "_extractor"
        features_extractor = get_model(config_args, device).to(device)
        features_extractor.load_state_dict(checkpoint["model_state_dict"], strict=False)

        class_num = config_args['data']['num_classes']
        dloader = get_loader(config_args)
        dloader.make_loaders()

        all_train_data, all_train_targets = retrieve_data(dataset, dloader.train_dataset)
        all_test_data, all_test_targets = retrieve_data(dataset, dloader.test_dataset)
        if dloader.val_dataset is not None:
            print('using default val data split')
            valset_data, valset_targets = retrieve_data(dataset, dloader.val_dataset)
            trainset_data, trainset_targets = all_train_data, all_train_targets
        else:
            valset_data = all_train_data[dloader.val_idx]
            valset_targets = torch.tensor(all_train_targets)[dloader.val_idx]

            trainset_data = all_train_data[dloader.train_idx]
            trainset_targets = torch.tensor(all_train_targets)[dloader.train_idx]

        setattr(dloader, 'all_test_data', all_test_data)
        setattr(dloader, 'all_test_targets', all_test_targets)
        setattr(dloader, 'mean', config_args['training']['augmentations']['normalize'][0])
        setattr(dloader, 'std', config_args['training']['augmentations']['normalize'][1])

        return model, features_extractor, dloader,\
                trainset_data, trainset_targets,\
                valset_data, valset_targets,\
                dataset, class_num
 
    else:
        # process config from custom trained models
        file_name = Path(config_path).name
        prefix = file_name.split('baseline')[0].split('_')
        dataset, model_name = prefix[0], '_'.join([ _ for _ in prefix[1:] if _ != ''])
        # print(prefix, dataset, model_name)

        class_num_map = {
            'cifar10': 10,
            'cinic10': 10,
            'stl10': 10
        }

        class_num = class_num_map[dataset]

        if 'resnet18_dp' in model_name:
            if 'droprate' in model_name:
                droprate = float(model_name.split('resnet18_dp_droprate_')[1])
                model = resnet18_dp(class_num, droprate=droprate).to(device)
            else:
                model = resnet18_dp(class_num).to(device)


        print('using model', model_name)

        assert '.pt' in config_path
        model.load_state_dict(torch.load(config_path))
        print('model weight loaded!')
        features_extractor = copy.deepcopy(model)
        if hasattr(features_extractor, 'fc2'):
            features_extractor.fc2 = nn.Identity()
        elif hasattr(features_extractor, 'classifier'):
            features_extractor.classifier = nn.Identity()
        else:
            features_extractor.fc = nn.Identity()

        idx_dir_path = Path(config_path).resolve().parent
        train_idx = np.load(idx_dir_path / 'train_idx.npy')
        val_idx = np.load(idx_dir_path / 'val_idx.npy')

        data_root = './data'

        if dataset == 'cifar10':
            mean = (0.4914, 0.4822, 0.4465) 
            std = (0.2023, 0.1994, 0.2010)

            train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                                        trn.ToTensor(), trn.Normalize(mean, std)])
            test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

            data_path = f'{data_root}/cifar10-data'
            train_data = dset.CIFAR10(data_path, train=True, transform=train_transform, download=True)
            test_data = dset.CIFAR10(data_path, train=False, transform=test_transform, download=True)

        elif dataset == 'cinic10':
            mean = [0.47889522, 0.47227842, 0.43047404]
            std = [0.24205776, 0.23828046, 0.25874835]

            train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                                        trn.ToTensor(), trn.Normalize(mean, std)])
            test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

            # data_dir = '/data/ailin/data/CINIC-10/'
            data_path = f'{data_root}/CINIC-10/'
            train_data = dset.ImageFolder(data_path + 'train', transform=train_transform)
            test_data = dset.ImageFolder(data_path + 'test', transform=test_transform)
            val_data = dset.ImageFolder(data_path + 'valid', transform=train_transform)


        elif dataset == 'stl10':
            mean = (.5, .5, .5) 
            std = (.5, .5, .5) 

            train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(96, padding=12),
                                        trn.ToTensor(), trn.Normalize(mean, std)])
            test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

            data_path = f'{data_root}/stl10/'
            train_data = dset.STL10(root=data_path, split='train', transform=train_transform, download=True)
            test_data = dset.STL10(root=data_path, split='test', transform=test_transform, download=True)
            unlabeled_data = dset.STL10(root=data_path, split='unlabeled', transform=train_transform, download=True)


        all_train_data, all_train_targets = retrieve_data(dataset, train_data)
        all_test_data, all_test_targets = retrieve_data(dataset, test_data)

        if dataset == 'cinic10':
            all_val_data, all_val_targets = retrieve_data(dataset, val_data)
            valset_data = all_val_data
            valset_targets = all_val_targets


            val_loader = torch.utils.data.DataLoader(
                val_data, batch_size=256, shuffle=True,
                num_workers=2, pin_memory=True
            )
        else:
            valset_data = all_train_data[val_idx]
            valset_targets = torch.tensor(all_train_targets)[val_idx]
            val_sampler = SubsetRandomSampler(val_idx)

            val_loader = torch.utils.data.DataLoader(
                    dataset=train_data,
                    batch_size=256,
                    sampler=val_sampler,
                    num_workers=2,
                )

        unlabeled_loader = None
        if dataset == 'stl10':
            unlabeled_loader = torch.utils.data.DataLoader(
                unlabeled_data, batch_size=256, shuffle=True,
                num_workers=2, pin_memory=True
            )
            all_unlabeled_data, all_unlabeled_targets = retrieve_data(dataset, unlabeled_data)

        trainset_data = all_train_data[train_idx]
        trainset_targets = torch.tensor(all_train_targets)[train_idx]


        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
    
        # Make loaders
        if dataset == 'stl10':
            # smaller batch size for gpu size
            train_loader = torch.utils.data.DataLoader(
                dataset=train_data,
                batch_size=128,
                sampler=train_sampler,
                num_workers=2,
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                dataset=train_data,
                batch_size=256,
                sampler=train_sampler,
                num_workers=2,
            )
        
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=200, shuffle=False,
            num_workers=2, pin_memory=True)

        dloader = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'augmentations_test': test_transform,
            'augmentations_train': train_transform,
            'test_dataset': test_data,
            'all_test_data': all_test_data,
            'all_test_targets': all_test_targets,
            'mean': mean,
            'std': std,
        }
        if unlabeled_loader is not None:
           dloader['unlabeled_loader'] = unlabeled_loader
           dloader['all_unlabeled_data'] = all_unlabeled_data
           dloader['all_unlabeled_targets'] = all_unlabeled_targets
        dloader = SimpleNamespace(**dloader)

        return model, features_extractor, dloader,\
                trainset_data, trainset_targets,\
                valset_data, valset_targets,\
                dataset, class_num

def get_data_from_ImageFolder(folder):

    samples = []
    targets = []
    print('folder loader', len(folder))
    for i in range(len(folder)):
        s, t = folder[i]
        samples.append(s)

    # print('samples', len(samples))
    return samples


def get_ood_data(dataset_name):
    data_root = './data'
    if dataset_name in ['svhn']:
        testset = torchvision.datasets.SVHN(root=data_root, split='test', download=True)

        test_ood_dataset = torchvision.datasets.SVHN(root=data_root, split='test', download=True)
        test_ood_dataset.labels = np.ones(len(test_ood_dataset))

    elif dataset_name in ['imagenet_resize', 'lsun_resize', 'imagenet_pil', 'lsun_pil']:
        
        dir_map = {
            'imagenet_resize': f"{data_root}/Imagenet_resize",
            'lsun_resize': f"{data_root}/LSUN_resize",
            'imagenet_pil': f"{data_root}/Imagenet_pil",
            'lsun_pil': f"{data_root}/LSUN_pil",
        }
        test_ood_dataset = torchvision.datasets.ImageFolder(dir_map[dataset_name])

        if not hasattr(test_ood_dataset, 'labels'):
            setattr(test_ood_dataset, 'labels', [])
            setattr(test_ood_dataset, 'data', [])

        data = get_data_from_ImageFolder(test_ood_dataset)
        test_ood_dataset.labels = np.ones(len(data))
        test_ood_dataset.data = data

    return test_ood_dataset.data, test_ood_dataset.labels

class PartialFolder(torch.utils.data.Dataset):
    def __init__(self, parent_ds, perm, length):
        self.parent_ds = parent_ds
        self.perm = perm
        self.length = length
        super(PartialFolder, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.parent_ds[self.perm[i]]


def validation_split_folder(dataset, val_share=0.1, seed=42):
    """
       Split a (training and vaidation combined) dataset into training and validation.
       Note that to be statistically sound, the items in the dataset should be statistically
       independent (e.g. not sorted by class, not several instances of the same dataset that
       could end up in either set).

       inputs:
          dataset:   ("training") dataset to split into training and validation
          val_share: fraction of validation data (should be 0<val_share<1, default: 0.1)
       returns: input dataset split into test_ds, val_ds

    """
    num_train = int(len(dataset) * (1 - val_share))
    num_val = len(dataset) - num_train

    perm = np.asarray(range(len(dataset)))
    # np.random.seed(0)
    np.random.seed(seed)
    np.random.shuffle(perm)

    train_perm, val_perm = perm[:num_train], perm[num_train:]

    return PartialFolder(dataset, train_perm, num_train), PartialFolder(dataset, val_perm, num_val)
