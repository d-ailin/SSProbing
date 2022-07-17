from distutils.command.config import config
from tkinter.messagebox import NO
import torch
import argparse

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
from pathlib import Path


from rot_learner.util import show_ssl_cls
from rot_learner import main_mis
def main_run_one_task(task_name, task_config, model_pkg, data_pkg, dataset, args, device, cached_model=None):
    if task_name == 'rotation':
        PretextDataset = RotDataset
    elif task_name == 'translation':
        PretextDataset = TransDataset
    sub_tasks = task_config
    sub_task_num = len(sub_tasks)

    class_num = 10

    dloader,\
    trainset_data, trainset_targets,\
    valset_data, valset_targets = data_pkg

    features_extractor, model = model_pkg

    train_ds = PretextDataset(trainset_data, sub_tasks, trainset_targets, dataset=dataset, train_mode=True)
    val_ds = PretextDataset(valset_data, sub_tasks, valset_targets, dataset=dataset, train_mode=False)
    # train_val_ds = PretextDataset(valset_data, sub_tasks, valset_targets, dataset=dataset, train_mode=True)
    test_ds = PretextDataset(dloader.all_test_data, sub_tasks, dloader.all_test_targets, dataset=dataset, train_mode=False)
    test_train_ds = PretextDataset(trainset_data, sub_tasks, trainset_targets, dataset=dataset, train_mode=False)

    trainloader = DataLoader(train_ds, batch_size=300, shuffle=True)
    if dataset == 'stl10' and sub_task_num >= 6:
        # smaller batch size for gpu
        trainloader = DataLoader(train_ds, batch_size=100, shuffle=True)

    valloader = DataLoader(val_ds, batch_size=100, shuffle=False)
    # train_valloader = DataLoader(train_val_ds, batch_size=100, shuffle=True)
    testloader = DataLoader(test_ds, batch_size=128, shuffle=False)
    test_trainloader = DataLoader(test_train_ds, batch_size=128, shuffle=False, pin_memory=False, num_workers=3)

    from rot_learner import main, main_cinic10, main_stl10

    if dataset == 'cinic10':
        main = main_cinic10
    if dataset == 'stl10':
        main = main_stl10
    

    if cached_model is None:
        # train
        net = main_mis.train(features_extractor, sub_task_num, 512, args.ssl_epoch, trainloader, device, dataset, valloader, model)
    else:
        net = cached_model

    cls_res, aug_res = main.test(net, model, testloader, class_num, sub_task_num, device)
    val_cls_res, val_aug_res = main.test(net, model, valloader, class_num, sub_task_num, device)

    pred_scores, pred_corrects, gt_labels, pred_labels, pred_scores_all, pred_scores_for_gts, pred_features, pred_logits, pred_loss_all = cls_res
    rot_scores, rot_accs, rot_scores_full, rot_scores_all, aug_features, aug_logits, all_rot_corrects, aug_loss_all = aug_res

    val_pred_scores, val_pred_corrects, val_gt_labels, val_pred_labels, val_pred_scores_all, val_pred_scores_for_gts, val_pred_features, val_pred_logits, val_pred_loss_all = val_cls_res
    val_rot_scores, val_rot_accs, val_rot_scores_full, val_rot_scores_all, val_aug_features, val_aug_logits, val_all_rot_corrects, val_aug_loss_all = val_aug_res


    return {
        'task':  task_name,
        'val_aug_scores': np.array(val_rot_scores_full)[:, 0],
        'val_pred_corrects': np.array(val_pred_corrects),
        'val_pred_scores': np.array(val_pred_scores),
        'aug_scores': np.array(rot_scores_full)[:, 0],
        'pred_corrects': np.array(pred_corrects),
        'pred_scores': np.array(pred_scores),
        'net': net
    }


def get_weighted_res(all_scores, args, device, metrics='auroc', choose_smaller=False, cached_models=None, task_config=None, debug=None):

    # train_scores, val_scores, test_scores = all_scores
    val_scores = np.array(all_scores['val'])
    test_scores = np.array(all_scores['test'])
    test_corrects = np.array(all_scores['test_corrects'])
    val_corrects = np.array(all_scores['val_corrects'])

    model, features_extractor, dloader,\
    trainset_data, trainset_targets,\
    valset_data, valset_targets,\
    dataset, class_num = preprocess_config(args.config_path, args, device)

    
    if cached_models is None:
        cached_models = {}
    else:
        cached_models = cached_models
    all_output = {}
    for task_name, task_arr in task_config.items():
        task_model = None
        if task_name in cached_models:
            task_model = cached_models[task_name]

        task_output = main_run_one_task(task_name, task_arr,
            (features_extractor, model),
            ( dloader,\
                trainset_data, trainset_targets,\
                valset_data, valset_targets),
            dataset,
            args,
            device,
            cached_model=task_model
        )

        all_output[task_name] = task_output

    M = len(task_config.keys())
    cur_config = np.zeros(M)
    best_config = np.zeros(M)
    best_res = 0

    def get_depth_scores(cur=0, trials=100, M=3, pretext_scores=[], pred_scores=[], pred_corrects=[], metric='auroc', best_res = 0, best_config=[]):
        if 1 - cur_config.sum() < 0:
            return 0, None
        if cur == M:
            scores = np.dot(pretext_scores, cur_config) + (1-cur_config.sum()) * pred_scores
            res = eval_res(scores, pred_corrects, verbose=False)

            if res[metric] > best_res:
                best_res = res[metric]
                best_config = np.copy(cur_config)
            # print('cur_config', cur_config)
            return best_res, best_config

        if cur == 0:
            loop = tqdm(range(trials))
        else:
            loop = range(trials)

        # res = 0
        # temp_config = None
        for i in loop:
            cur_config[cur] = i / trials
            _res, _config = get_depth_scores(cur=cur+1, trials=trials, M=M, pretext_scores=pretext_scores, pred_scores=pred_scores, pred_corrects=pred_corrects, metric=metric, best_res=best_res, best_config=best_config)

            if _res > best_res:
                best_config = copy.deepcopy(_config)
                best_res = _res
            cur_config[cur+1:] = 0

        return best_res, best_config

    val_pretext_scores = []
    test_pretext_scores = []
    val_pred_scores = None
    val_pred_corrects = None
    test_pred_scores = None
    test_pred_corrects = None
    for task_name, task_out in all_output.items():
        val_pretext_scores.append(task_out['val_aug_scores'].reshape(-1, 1))
        test_pretext_scores.append(task_out['aug_scores'].reshape(-1, 1))

        val_pred_scores = task_out['val_pred_scores']
        val_pred_corrects = task_out['val_pred_corrects']
        test_pred_scores = task_out['pred_scores']
        test_pred_corrects = task_out['pred_corrects']

    val_pretext_scores = np.concatenate(val_pretext_scores, 1)
    test_pretext_scores = np.concatenate(test_pretext_scores, 1)

    all_new_res = {}
    for metric in metrics:
        # clear 
        cur_config = np.zeros(M)
        best_config = np.zeros(M)
        best_res = 0

        trials = 50 if M >= 3 else 100

        best_res, best_config = get_depth_scores(cur=0, trials=trials, M=M,
            pretext_scores=val_pretext_scores, pred_scores=val_scores, pred_corrects=val_corrects,
            metric=metric,
            best_res=best_res,
            best_config=best_config
        )

        new_scores = np.dot(test_pretext_scores, best_config) + (1-best_config.sum()) * test_scores
        all_new_res[metric] = {**eval_res(new_scores, test_corrects), 'config': copy.deepcopy(best_config.tolist()), 'task_config': copy.deepcopy(task_config)}
        # print('config:', best_config)

    print('baseline:')
    eval_res(test_scores, test_corrects)


    if len(cached_models.keys()) <= 0:
        cached_models = {}
        for task_name, task_out in all_output.items():
            cached_models[task_name] = copy.deepcopy(task_out['net'])

    return all_new_res, cached_models



