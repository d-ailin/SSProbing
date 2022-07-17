from statistics import mode
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset

import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torch
import torchvision.transforms.functional as trnF
import torchvision.transforms as trn

from scipy import ndimage, misc
import numpy as np


def rotate_any_degree(x, degree):
    return ndimage.rotate(x.copy(), degree, reshape=False, mode='reflect')

# Rot dataset
class RotDataset(Dataset):
    def __init__(self, samples, PRED_ROTS, class_labels, dataset='cifar10', train_mode=True):
        self.rotations = PRED_ROTS
        self.samples = samples
        self.label_total = len(self.rotations)
        self.class_labels = class_labels

        self.train_mode = train_mode
        self.dataset = dataset

        self.init_transforms()

    def init_transforms(self):
        if self.dataset in ['cifar10']:
            self.transforms = {}
            self.transforms['normalize'] = trn.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            self.transforms['randomly_crop'] = trn.RandomCrop(32, padding=4)
        
        if self.dataset == 'cinic10':
            self.transforms = {}
            mean = [0.47889522, 0.47227842, 0.43047404]
            std = [0.24205776, 0.23828046, 0.25874835]

            self.transforms['normalize'] = trn.Normalize(mean, std) # to change
            self.transforms['randomly_crop'] = trn.RandomCrop(32, padding=4)

        if self.dataset == 'stl10':
            self.transforms = {}
            mean=[.5, .5, .5]
            std=[.5, .5, .5]

            self.transforms['normalize'] = trn.Normalize(mean, std) # to change
            self.transforms['randomly_crop'] = trn.RandomCrop(96, padding=12)
    
    def __len__(self):
        return len(self.samples)

    def set_train_mode(self, train_mode):
        self.train_mode = train_mode

    def __getitem__(self, idx):
        x_orig, classifier_target = self.samples[idx], self.class_labels[idx]

        # horizon flip
        if self.dataset in ['cifar10', 'cinic10', 'stl10'] and self.train_mode == True and np.random.uniform() < 0.5:
            x_orig = np.copy(x_orig)[:, ::-1]
        else:
            x_orig =  np.copy(x_orig)
        
        # solve inconsistent dimension shape, e.g. svhn
        if len(x_orig.shape) >= 3 and x_orig.shape[0] < x_orig.shape[1]:
            x_orig = np.transpose(x_orig, (1, 2, 0))

        if self.dataset in ['cifar10', 'cinic10', 'stl10'] and self.train_mode == True:
            x_orig = Image.fromarray(x_orig)
            x_orig = self.transforms['randomly_crop'](x_orig)
            x_orig = np.asarray(x_orig)


        x_tf_0 = np.copy(x_orig)
        x_tf_90 = np.rot90(x_orig.copy(), k=1).copy()
        x_tf_180 = np.rot90(x_orig.copy(), k=2).copy()
        x_tf_270 = np.rot90(x_orig.copy(), k=3).copy()

            
        x_tfs = {
            '0': x_tf_0,
            '90': x_tf_90,
            '180': x_tf_180,
            '270': x_tf_270,
        }

        cls_input = self.transforms['normalize'](trnF.to_tensor(x_tf_0))
        
 
        aug_inputs = []
        for rot_degree in self.rotations:
            if str(rot_degree) in x_tfs: 
                aug_inputs.append(self.transforms['normalize'](trnF.to_tensor(x_tfs[str(rot_degree)])))
            else:
                aug_inputs.append(self.transforms['normalize'](trnF.to_tensor( rotate_any_degree(x_orig, rot_degree) )))


        aug_inputs = torch.cat([a_input.unsqueeze(0) for a_input in aug_inputs], 0)

        aug_labels = torch.arange(len(self.rotations)).long().clone().detach()
        if  torch.is_tensor(classifier_target):
            class_label = classifier_target.clone().long().detach()
        else:
            class_label = torch.tensor(classifier_target).long()


        return cls_input, aug_inputs, class_label, aug_labels

