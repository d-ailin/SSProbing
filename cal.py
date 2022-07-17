from cv2 import normalize
import torch
import argparse
from torch import normal, optim

from torch._C import device
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from confidnet.models import get_model
from confidnet.utils.misc import load_yaml
from confidnet.loaders import get_loader
from confidnet.utils.schedulers import ConstantLR
from torch.optim.sgd import SGD
from torch.optim import lr_scheduler


import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from rot_learner.dataset_trans import TransDataset
from rot_learner.util import retrieve_data
from rot_learner import main_cal, main_cinic10, main_stl10
from rot_learner.dataset import RotDataset
import copy
from rot_learner.data_util import preprocess_config

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", "-c", type=str, default=None, help="Path for config yaml")
parser.add_argument("--save_file", "-sf", type=str, default='cal_res.txt', help="save file name")


args = parser.parse_args()
print('args', args)

device='cuda:0'


model, features_extractor, dloader,\
trainset_data, trainset_targets,\
valset_data, valset_targets,\
dataset, class_num = preprocess_config(args.config_path, args, device)

if dataset == 'stl10':
    TRANS = [(0, 0), (0, -32), (0, 32)]
    PRED_ROTS = [0, 90]
else:    
    TRANS = [(0, 0), (0, -8), (0, 8), (-8, 0), (8, 0)]
    PRED_ROTS = [0, 90, 180, 270]
PRED_TASK_NUM = len(PRED_ROTS)
TRANS_TASK_NUM = len(TRANS)



train_ds = RotDataset(trainset_data, PRED_ROTS, trainset_targets, dataset=dataset, train_mode=True)
val_ds = RotDataset(valset_data, PRED_ROTS, valset_targets, dataset=dataset, train_mode=False)
test_ds = RotDataset(dloader.all_test_data, PRED_ROTS, dloader.all_test_targets, dataset=dataset, train_mode=False)

train_trans_ds = TransDataset(trainset_data, TRANS, trainset_targets, dataset=dataset, train_mode=True)
val_trans_ds = TransDataset(valset_data, TRANS, valset_targets, dataset=dataset, train_mode=False)
test_trans_ds = TransDataset(dloader.all_test_data, TRANS, dloader.all_test_targets, dataset=dataset, train_mode=False)


# print('train_ds', len(train_ds))
# print('val_ds', len(val_ds))

trainloader = DataLoader(train_ds, batch_size=300, shuffle=True)
valloader = DataLoader(val_ds, batch_size=300, shuffle=False)
trans_valloader = DataLoader(val_trans_ds, batch_size=300, shuffle=False)
trans_trainloader = DataLoader(train_trans_ds, batch_size=300, shuffle=True)
testloader = DataLoader(test_ds, batch_size=300, shuffle=False, pin_memory=False, num_workers=3)
trans_testloader = DataLoader(test_trans_ds, batch_size=300, shuffle=False, pin_memory=False, num_workers=3)

if dataset == 'cinic10':
    trans_net = main_cinic10.train(copy.deepcopy(features_extractor), TRANS_TASK_NUM, 512, 2, trans_trainloader, device)
    rot_net = main_cinic10.train(features_extractor, len(PRED_ROTS), 512, 2, trainloader, device)
    cls_res, rot_res = main_cinic10.test(rot_net, model, testloader, class_num, PRED_TASK_NUM, device)
    cls_res, trans_res = main_cinic10.test(trans_net, model, trans_testloader, class_num, TRANS_TASK_NUM, device)
elif dataset == 'stl10':
    trans_net = main_stl10.train(copy.deepcopy(features_extractor), TRANS_TASK_NUM, 512, 5, trans_trainloader, device)
    rot_net = main_stl10.train(features_extractor, len(PRED_ROTS), 512, 5, trainloader, device)
    cls_res, rot_res = main_stl10.test(rot_net, model, testloader, class_num, PRED_TASK_NUM, device)
    cls_res, trans_res = main_stl10.test(trans_net, model, trans_testloader, class_num, TRANS_TASK_NUM, device)
else:
    trans_net = main_cal.train(copy.deepcopy(features_extractor), TRANS_TASK_NUM, 512, 2, trans_trainloader, device)
    rot_net = main_cal.train(features_extractor, len(PRED_ROTS), 512, 2, trainloader, device)
    cls_res, rot_res = main_cal.test(rot_net, model, testloader, class_num, PRED_TASK_NUM, device)
    cls_res, trans_res = main_cal.test(trans_net, model, trans_testloader, class_num, TRANS_TASK_NUM, device)


pred_scores, pred_corrects, gt_labels, pred_labels, pred_scores_all, pred_scores_for_gts, pred_features, pred_logits, pred_loss_all = cls_res
rot_scores, rot_accs, rot_scores_full, rot_scores_all, rot_features, rot_logits, all_rot_corrects, rot_loss_all = rot_res
trans_scores, trans_accs, trans_scores_full, trans_scores_all, trans_features, trans_logits, all_trans_corrects, trans_loss_all = trans_res

val_cls_res, val_rot_res = main_cal.test(rot_net, model, valloader, class_num, PRED_TASK_NUM, device)
val_cls_res, val_trans_res = main_cal.test(trans_net, model, trans_valloader, class_num, TRANS_TASK_NUM, device)

val_pred_scores, val_pred_corrects, val_gt_labels, val_pred_labels, val_pred_scores_all, val_pred_scores_for_gts, val_pred_features, val_pred_logits, val_pred_loss_all = val_cls_res
val_rot_scores, val_rot_accs, val_rot_scores_full, val_rot_scores_all, val_aug_features, val_aug_logits, val_all_rot_corrects, val_aug_loss_all = val_rot_res
val_trans_scores, val_trans_accs, val_trans_scores_full, val_trans_scores_all, val_trans_features, val_trans_logits, val_all_trans_corrects, val_trans_loss_all = val_trans_res

from rot_learner.cal_util import TemperatureScalingWithSSL, cal_results, HistogramBinning, TemperatureScaling, evaluate

base_res = evaluate(np.array(pred_scores_all), np.array(gt_labels), verbose=True, normalize=True)

# print(np.array(val_pred_logits).shape)
# print(np.array(val_gt_labels).shape)
# print( sum(val_pred_corrects) )
hb_input = (np.array(val_pred_logits), np.array(val_gt_labels) ), (np.array(pred_logits), np.array(gt_labels) )

print('histogram binning:')
histbin_res = cal_results(HistogramBinning, hb_input, {'M':15}, approach='single')

print('tempature scaling:')
tempscale_res = cal_results(TemperatureScaling, hb_input, {'maxiter': 100}, approach='all')


try:
    tscaler_ssl = TemperatureScalingWithSSL(maxiter=100, solver='SLSQP')
    fit_res = tscaler_ssl.fit(np.array(val_pred_logits), np.array(val_gt_labels),  np.concatenate([
        np.array(val_rot_scores_full)[:, 0].reshape(-1, 1),
        np.array(val_trans_scores_full)[:, 0].reshape(-1, 1),
    ], 1))

    val_ssl_scaled_softmax_outs = tscaler_ssl.predict(np.array(val_pred_logits), np.concatenate([
        np.array(val_rot_scores_full)[:, 0].reshape(-1, 1),
        np.array(val_trans_scores_full)[:, 0].reshape(-1, 1),
    ], 1))


    ssl_scaled_softmax_outs = tscaler_ssl.predict(np.array(pred_logits), np.concatenate([
        np.array(rot_scores_full)[:, 0].reshape(-1, 1),
        np.array(trans_scores_full)[:, 0].reshape(-1, 1),
    ], 1))

    print('temperature using ssl val:')
    evaluate(val_ssl_scaled_softmax_outs, np.array(val_gt_labels), verbose=True, normalize=True)

    print('temperature using ssl test:')
    res_with_ssl = evaluate(ssl_scaled_softmax_outs, np.array(gt_labels), verbose=True, normalize=True)

except Exception as err:
    print('failed', err)


all_res = {
    'base': base_res,
    'hist_bin': histbin_res,
    'temp_scaling': tempscale_res,
    'ts_with_ssl': res_with_ssl
}

import json
out = all_res
with open(f'res_dir/{args.save_file}', 'a') as f:
    out['config'] = args.config_path
    f.write(json.dumps(out) + '\n')


