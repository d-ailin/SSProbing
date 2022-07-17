import torch
import argparse
from torch import optim
import torch.nn as nn

import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from rot_learner.util import eval_res
import copy
from rot_learner.data_util import preprocess_config, entropy
import ood_base
import json
from ood_util import get_ood_scores


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", "-c", type=str, default=None, help="Path for config yaml")
parser.add_argument("--ssl_epoch", "-se", type=int, default=None, help="SSL training Epoch")
parser.add_argument("--method", "-m", type=str, default='mcp', help="baseline method", choices=['mcp', 'entropy', 'all'])
parser.add_argument("--choose_metric", "-cm", nargs='+', default=['err_aupr', 'auroc'], help="select criteria metric", choices=['err_aupr', 'auroc'])
parser.add_argument("--save_file", "-sf", type=str, default='ood_res.txt', help="save file name")


args = parser.parse_args()
print('args', args)

device='cuda:0'

model, features_extractor, dloader,\
trainset_data, trainset_targets,\
valset_data, valset_targets,\
dataset, class_num = preprocess_config(args.config_path, args, device)

import global_var

def test(model, dataloder, mc_dropout=False):
    model.eval()
    pred_scores = []
    pred_labels = []
    pred_groundtruths = []
    pred_corrects = []
    pred_outputs = []
    pred_all_scores = []

    if mc_dropout:
        global_var.set_mcdropout(True)
        mc_model = copy.deepcopy(model)
        mc_model.mc_dropout = True
        if hasattr(mc_model, 'keep_dropout_in_test'):
            mc_model.keep_dropout_in_test()

        test_model = mc_model
    else:
        test_model = model

    with torch.no_grad():
        for data, target in dataloder:
            data, target = data.to(device), target.to(device)
            output = test_model(data)
            scores = F.softmax(output, 1).max(1)[0]
            labels = output.data.max(1)[1]
            corrects = labels.eq(target)
            outputs = output
            all_scores = F.softmax(output, 1)

            pred_scores += scores.detach().cpu().tolist()
            pred_labels += labels.detach().cpu().tolist()
            pred_groundtruths += target.cpu().tolist()
            pred_corrects += corrects.cpu().tolist()
            pred_outputs += outputs.cpu().tolist()
            pred_all_scores += all_scores.cpu().tolist()

    if mc_dropout:
        global_var.set_mcdropout(False)
        test_model.mc_dropout = False
        test_model.eval()

    return pred_scores, pred_labels, pred_groundtruths, pred_corrects, pred_outputs, pred_all_scores



def get_features(model, dataloader):
    model.eval()
    features = []
    targets = []
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            features += output.detach().cpu().tolist()
            targets += target.cpu().tolist()

    return features, targets



test_methods = []
if args.method == 'all':
    test_methods = ['mcp', 'entropy']
else:
    test_methods = [args.method]



if valset_data[0].shape[0] == 3:
    valset_data = np.transpose(valset_data, (0, 2, 3, 1))

valset_data_tensor = torch.Tensor([dloader.augmentations_test(vd).tolist() for vd in valset_data])
# test_val_ds = TensorDataset(valset_data_tensor, torch.tensor(valset_targets))
test_val_ds = TensorDataset(valset_data_tensor, valset_targets.clone().detach())
test_val_loader = DataLoader(test_val_ds, batch_size=200, shuffle=False)



base_pred_scores, base_pred_labels, base_pred_groundtruths, base_pred_corrects, base_pred_logits, base_pred_all_scores = test(model, dloader.test_loader)
base_val_scores, base_val_labels, base_val_groundtruths, base_val_corrects, base_val_logits, base_val_all_scores = test(model, test_val_loader)

all_res = []
cached_models = None

for method in test_methods:
    if method == 'mcp':
        res = eval_res(base_pred_scores, base_pred_corrects)
        # print(np.array(base_val_scores).shape, np.array(base_pred_scores).shape)
        method_scores = {
            'val': base_val_scores,
            'test': base_pred_scores
        }
    elif method == 'entropy':
 
        scores = -1 * entropy(np.array(base_pred_all_scores))
        val_scores = -1 * entropy(np.array(base_val_all_scores))

        res = eval_res(scores, base_pred_corrects)

        method_scores = {
            'val': val_scores,
            'test': scores
        }

    if cached_models is None:
        weighted_res, trained_models, test_scores_pkg = ood_base.get_weighted_res(method_scores, args, device, metrics=args.choose_metric, return_scores=True)
        cached_models = trained_models
    else:
        weighted_res, _, test_scores_pkg = ood_base.get_weighted_res(method_scores, args, device, metrics=args.choose_metric, cached_models=cached_models, return_scores=True)

    out = {
        'method': method,
        'baseline': res,
    }
    for metric, w_res in weighted_res.items():
        out[f'weighted_{metric}'] = w_res

    # evaluate ood
    # only for mcp
    from sklearn.metrics import roc_auc_score

    print('config:', weighted_res['err_aupr']['config'])
    for ood_name in ['svhn', 'lsun_resize', 'lsun_pil', 'imagenet_resize', 'imagenet_pil']:
        ood_scores_pkg = get_ood_scores(ood_name, args, device, cached_models)

        pred_mcp_scores, rot_scores, trans_scores = test_scores_pkg['pred'], test_scores_pkg['rot'], test_scores_pkg['trans']
        ood_pred_mcp_scores, ood_rot_scores, ood_trans_scores = ood_scores_pkg['pred'], ood_scores_pkg['rot'], ood_scores_pkg['trans']

        if method == 'entropy':
            pred_scores = -1 * entropy(np.array(base_pred_all_scores))
            ood_pred_scores = -1 * entropy(np.array(ood_scores_pkg['pred_all']))

        else:
            pred_scores = pred_mcp_scores
            ood_pred_scores = ood_pred_mcp_scores


        print('ood name:', ood_name)
        print('baseline:')

        ood_test_scores = pred_scores.tolist() + ood_pred_scores.tolist()
        ood_labels = np.concatenate((np.ones_like(rot_scores), np.zeros_like(ood_rot_scores))).astype(int).tolist()
        print('ood task baseline scores:',roc_auc_score(ood_labels, ood_test_scores))
        base_auroc = roc_auc_score(ood_labels, ood_test_scores)

        # weighted score ood
        config = weighted_res['err_aupr']['config']
        cat_scores = np.array(pred_scores.tolist() + ood_pred_scores.tolist())
        cat_rot_scores = np.array(rot_scores.tolist() + ood_rot_scores.tolist())
        cat_trans_scores = np.array(trans_scores.tolist() + ood_trans_scores.tolist())
        weighted_ood_scores = (1-config[0]-config[1]) * cat_scores + config[0] * cat_rot_scores + config[1] * cat_trans_scores
        print('ood task weighted scores:',roc_auc_score(ood_labels, weighted_ood_scores))
        weighted_auroc = roc_auc_score(ood_labels, weighted_ood_scores)


        with open(f'res_dir/{args.save_file}', 'a') as f:
            out['config'] = args.config_path
            out['choose_metric'] = 'aupr_error'
            out['ood_name'] = ood_name
            out['base_auroc'] = base_auroc
            out['weighted_auroc'] = {'rot_trans': weighted_auroc}
            f.write(json.dumps(out) + '\n')



for res in all_res:
    print('method:', res['method'])
    print('baseline:', res['baseline'])

    for metric in args.choose_metric:
        out[f'weighted_{metric}'] = w_res
        print(f'weighted_{metric}:', res[f'weighted_{metric}'])
    