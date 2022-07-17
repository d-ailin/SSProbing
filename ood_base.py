from distutils.command.config import config
import enum
from tabnanny import verbose
from tkinter.messagebox import NO
import torch
import argparse

from torch._C import device
from confidnet.models import get_model
from confidnet.utils.misc import load_yaml
from confidnet.loaders import get_loader

import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from rot_learner.dataset import RotDataset
from rot_learner.dataset_trans import TransDataset
from rot_learner.util import eval_res, retrieve_data
from rot_learner.data_util import preprocess_config
from tqdm import tqdm


import copy
def get_weighted_res(all_scores, args, device, metrics=['auroc'], choose_smaller=False, cached_models=None, return_scores=False):
    

    # train_scores, val_scores, test_scores = all_scores
    val_scores = all_scores['val']
    test_scores = all_scores['test']

    model, features_extractor, dloader,\
    trainset_data, trainset_targets,\
    valset_data, valset_targets,\
    dataset, class_num = preprocess_config(args.config_path, args, device)

    
    if dataset == 'stl10':
        # TRANS = [(0, 0), (0, -32), (0, 32), (-32, 0), (32, 0)]
        TRANS = [(0, 0), (0, -32), (0, 32)]
        # TRANS = [(0, 0), (-32, 0), (32, 0)]
        # TRANS = [(0, 0), (-32, 0), (32, 0)]
        PRED_ROTS = [0, 90]
    else:
        TRANS = [(0, 0), (0, -8), (0, 8), (-8, 0), (8, 0)]
        PRED_ROTS = [0, 90, 180, 270]

    TRANS_TASK_NUM = len(TRANS)
    PRED_TASK_NUM = len(PRED_ROTS)

    train_ds = RotDataset(trainset_data, PRED_ROTS, trainset_targets, dataset=dataset, train_mode=True)
    val_ds = RotDataset(valset_data, PRED_ROTS, valset_targets, dataset=dataset, train_mode=False)
    test_ds = RotDataset(dloader.all_test_data, PRED_ROTS, dloader.all_test_targets, dataset=dataset, train_mode=False)
    
    train_trans_ds = TransDataset(trainset_data, TRANS, trainset_targets, dataset=dataset, train_mode=True)
    val_trans_ds = TransDataset(valset_data, TRANS, valset_targets, dataset=dataset, train_mode=False)
    test_trans_ds = TransDataset(dloader.all_test_data, TRANS, dloader.all_test_targets, dataset=dataset, train_mode=False)

    # print('common idx', list(set(dloader.train_idx).intersection(dloader.val_idx)))

    print('train_ds', len(train_ds))
    print('val_ds', len(val_ds))

    trainloader = DataLoader(train_ds, batch_size=300, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=100, shuffle=False)
    # train_valloader = DataLoader(train_val_ds, batch_size=100, shuffle=True)
    testloader = DataLoader(test_ds, batch_size=128, shuffle=False, pin_memory=False, num_workers=3)
    # test_trainloader = DataLoader(test_train_ds, batch_size=128, shuffle=False, pin_memory=False, num_workers=3)


    trans_trainloader = DataLoader(train_trans_ds, batch_size=128, shuffle=True)
    trans_valloader = DataLoader(val_trans_ds, batch_size=128, shuffle=False)
    trans_testloader = DataLoader(test_trans_ds, batch_size=128, shuffle=False)



    # rot predict model
    from rot_learner import main, main_cinic10, main_stl10

    if dataset == 'cinic10':
        main = main_cinic10
    if dataset == 'stl10':
        main = main_stl10

    if cached_models is None:
        net = main.train(features_extractor, len(PRED_ROTS), 512, args.ssl_epoch, trainloader, device)
        trans_net = main.train(copy.deepcopy(features_extractor), TRANS_TASK_NUM, 512, args.ssl_epoch, trans_trainloader, device)

    else:
        net = cached_models['rot_net']
        trans_net = cached_models['trans_net']

    # class_num = int(config_args['data']['num_classes'])
    # eval test rot
    cls_res, aug_res = main.test(net, model, testloader, class_num, len(PRED_ROTS), device)
    val_cls_res, val_aug_res = main.test(net, model, valloader, class_num, len(PRED_ROTS), device)
    # train_val_cls_res, train_val_aug_res = main.test(net, model, train_valloader, class_num, len(PRED_ROTS), device)
    # train_cls_res, train_aug_res = main.test(net, model, test_trainloader, class_num, len(PRED_ROTS), device)


    pred_scores, pred_corrects, gt_labels, pred_labels, pred_scores_all, pred_scores_for_gts, pred_features, pred_logits, pred_loss_all = cls_res
    rot_scores, rot_accs, rot_scores_full, rot_scores_all, aug_features, aug_logits, all_rot_corrects, aug_loss_all = aug_res

    val_pred_scores, val_pred_corrects, val_gt_labels, val_pred_labels, val_pred_scores_all, val_pred_scores_for_gts, val_pred_features, val_pred_logits, val_pred_loss_all = val_cls_res
    val_rot_scores, val_rot_accs, val_rot_scores_full, val_rot_scores_all, val_aug_features, val_aug_logits, val_all_rot_corrects, val_aug_loss_all = val_aug_res

    trans_cls_res, trans_aug_res = main.test(trans_net, model, trans_testloader, class_num, TRANS_TASK_NUM, device)
    trans_val_cls_res, trans_val_aug_res = main.test(trans_net, model, trans_valloader, class_num, TRANS_TASK_NUM, device)

    # trans_pred_scores, trans_pred_corrects, trans_gt_labels, trans_pred_labels, trans_pred_scores_all, trans_pred_scores_for_gts, trans_pred_features, trans_pred_logits, trans_pred_loss_all = trans_cls_res
    trans_scores, trans_accs, trans_scores_full, trans_scores_all, trans_aug_features, trans_aug_logits, all_trans_corrects, trans_aug_loss_all = trans_aug_res

    # val_pred_scores, val_pred_corrects, val_gt_labels, val_pred_labels, val_pred_scores_all, val_pred_scores_for_gts, val_pred_features, val_pred_logits, val_pred_loss_all = trans_val_cls_res
    val_trans_scores, val_trans_accs, val_trans_scores_full, val_trans_scores_all, val_trans_aug_features, val_trans_aug_logits, val_all_trans_corrects, val_trans_aug_loss_all = trans_val_aug_res


    def cal_entropy(pred_vector):
        return - np.sum( pred_vector * np.log(pred_vector), -1)

    def find_best_avg(val_info, test_info , trials=50, metric='auroc'):
        pred_scores, rot_scores, trans_scores, pred_corrects = val_info
        test_scores, test_rot_scores, test_trans_scores = test_info
        best_auroc = 0
        best_config = np.zeros(2)

        if metric == 'erm':
            pass
        else:
            for i in tqdm(range(trials)):
                for j in range(trials):
                    # scores = i /trials * pred_scores + (1 - i/trials) * rot_scores

                    if 1 - ( (i + j) / trials) < 0:
                        continue

                    scores = i / trials * rot_scores + j / trials * trans_scores + (1 - (i+j)/trials) * pred_scores

                    # set false later
                    res = eval_res(scores, pred_corrects, verbose=False)

                    auroc = res[metric]
                    criteria = auroc < best_auroc if choose_smaller else auroc > best_auroc
                    if criteria:
                        best_auroc = auroc
                        best_config = (i / trials, j / trials)

        new_scores = best_config[0] * test_rot_scores + best_config[1] * test_trans_scores\
                    + (1-best_config[0]-best_config[1]) * test_scores
        

        return new_scores, best_config


    all_new_res = {}
    for metric in metrics:
        new_scores, score_config = find_best_avg(
            (np.array(val_scores), np.array(val_rot_scores_full)[:, 0], np.array(val_trans_scores_full)[:, 0], np.array(val_pred_corrects)),
            (np.array(test_scores), np.array(rot_scores_full)[:, 0], np.array(trans_scores_full)[:, 0]),
            metric=metric
        )

        all_new_res[metric] = {**eval_res(new_scores, pred_corrects), 'config': copy.deepcopy(score_config)}


    print('baseline (without weighted)')
    eval_res(test_scores, pred_corrects)

    if return_scores:
        return all_new_res, {'rot_net': net, 'trans_net': trans_net},\
        {'pred': np.array(pred_scores), 'rot': np.array(rot_scores_full)[:, 0], 'trans': np.array(trans_scores_full)[:, 0]}
    else:
        return all_new_res, {'rot_net': net, 'trans_net': trans_net}



