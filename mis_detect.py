import torch
import argparse
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

from rot_learner.util import eval_res
from rot_learner.data_util import preprocess_config, process_task_config
from utils.trust_score import TrustScore
import mis_base
import json
import copy


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", "-c", type=str, default=None, help="Path for config yaml")
parser.add_argument("--ssl_epoch", "-se", type=int, default=None, help="SSL training Epoch")
parser.add_argument("--method", "-m", type=str, default='mcp', help="baseline method", choices=['mcp', 'mcdropout', 'trustscore', 'tcp'])
parser.add_argument("--choose_metric", "-cm", nargs='+', default=['err_aupr'], help="select criteria metric", choices=['err_aupr', 'auroc'])
parser.add_argument("--save_file", "-sf", type=str, default='mis_detect_res.txt', help="save file name")
parser.add_argument("--task_config", "-t", type=str, default=None, help="task config path")


args = parser.parse_args()
print('args', args)

device='cuda:0'

model, features_extractor, dloader,\
trainset_data, trainset_targets,\
valset_data, valset_targets,\
dataset, class_num = preprocess_config(args.config_path, args, device)

task_config = process_task_config(args.task_config)

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
        T = 10
        if hasattr(mc_model, 'keep_dropout_in_test'):
            mc_model.keep_dropout_in_test()
            # print('bn training:', mc_model.conv1_bn.training)

        test_model = mc_model
    else:
        test_model = model
    
    test_model.eval()
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
        # if hasattr(mc_model, 'keep_dropout_in_test'):
        #     mc_model.keep_dropout_in_test()

    return pred_scores, pred_labels, pred_groundtruths, pred_corrects, pred_outputs, pred_all_scores


def mc_test(model, dataloder, mc_dropout=True, samples=10):
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
        samples = samples
        if hasattr(mc_model, 'keep_dropout_in_test'):
            mc_model.keep_dropout_in_test()
            # print('bn training:', mc_model.conv1_bn.training)

        test_model = mc_model
    else:
        test_model = model

    test_model.eval()
    with torch.no_grad():
        for data, target in tqdm(dataloder):
            data, target = data.to(device), target.to(device)
            outputs = []
            for i in range(samples):
                output = test_model(data)
                outputs.append(output.unsqueeze(0))

            output = torch.cat(outputs, 0).mean(0)
            probs = F.softmax(output, dim=1)
            confidence = (probs * torch.log(probs + 1e-9)).sum(dim=1)  # entropy
            pred = probs.max(dim=1)[1]

            corrects = pred.eq(target)

            pred_scores += confidence.detach().cpu().tolist()
            pred_labels += pred.detach().cpu().tolist()
            pred_groundtruths += target.cpu().tolist()
            pred_corrects += corrects.cpu().tolist()
            # pred_outputs += outputs.cpu().tolist()
            # pred_all_scores += all_scores.cpu().tolist()

    if mc_dropout:
        global_var.set_mcdropout(False)
        test_model.mc_dropout = False
        test_model.eval()
        # if hasattr(mc_model, 'keep_dropout_in_test'):
        #     mc_model.keep_dropout_in_test()

    return pred_scores, pred_labels, pred_groundtruths, pred_corrects


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

def tcp_training(feature_extractor, model, dataloader, in_size, test_loaders = []):
    tcp_model = nn.Sequential(
        nn.Linear(in_size, 500),
        nn.ReLU(),
        nn.Linear(500, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 1)
    )
    tcp_model.to(device)
    epochs = 100

    optimizer = optim.Adam(tcp_model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ConstantLR(optimizer)
    tcp_model.train()
    model.eval()
    feature_extractor.eval()
    loop = tqdm(range(epochs))
    for i in loop:
        loss_sum = 0
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            N = data.shape[0]
            optimizer.zero_grad()

            logits = model(data)
            feat = feature_extractor(data)

            pred_tcp = tcp_model(feat)

            row_index = torch.arange(N)
            gt_score = F.softmax(logits, 1)
            tcp_target = gt_score[row_index, target]

            loss = F.mse_loss(torch.sigmoid(pred_tcp).reshape(-1), tcp_target)

            loss.backward()
            optimizer.step()

            loss_sum = 0.9 * loss_sum + 0.1 * loss.item()
        # print(f'loss for epoch {i}', loss_sum)
        loop.set_description(f"Epoch {i+1}/{epochs}")
        loop.set_postfix({'loss': loss_sum})
        loop.update()
        scheduler.step()

    tcp_model.eval()
    model.eval()
    feature_extractor.eval()

    all_tcp_scores = []
    for test_loader in test_loaders:
        test_tcp_scores = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                feat = feature_extractor(data)
                tcp_scores = tcp_model(feat)
                
                test_tcp_scores += tcp_scores.reshape(-1).cpu().tolist()
        # print('test_tcp_scores', np.array(test_tcp_scores).shape)
        all_tcp_scores.append(test_tcp_scores)

    return all_tcp_scores



test_methods = [args.method]


if valset_data[0].shape[0] == 3:
    valset_data = np.transpose(valset_data, (0, 2, 3, 1))

valset_data_tensor = torch.Tensor([dloader.augmentations_test(vd).tolist() for vd in valset_data])
test_val_ds = TensorDataset(valset_data_tensor, valset_targets.clone().detach())
test_val_loader = DataLoader(test_val_ds, batch_size=200, shuffle=False)


base_pred_scores, base_pred_labels, base_pred_groundtruths, base_pred_corrects, _, base_pred_all_scores = test(model, dloader.test_loader)
base_val_scores, base_val_labels, base_val_groundtruths, base_val_corrects, _, base_val_all_scores = test(model, test_val_loader)

all_res = []
cached_models = None

for method in test_methods:
    if method == 'mcp':
        res = eval_res(base_pred_scores, base_pred_corrects)
        print(np.array(base_val_scores).shape, np.array(base_pred_scores).shape)
        method_scores = {
            'val': base_val_scores,
            'test': base_pred_scores
        }

    elif method == 'mcdropout':
        mc_samples = 30
        scores, _, gt, correct = mc_test(model, dloader.test_loader, mc_dropout=True, samples = mc_samples)
        val_scores, _, _, _ = mc_test(model, test_val_loader, mc_dropout=True, samples = mc_samples)


        print('pred acc:', sum(correct) / len(correct))
        res = eval_res(scores, correct)

        method_scores = {
            'val': val_scores,
            'test': scores
        }        
    elif method == 'trustscore':
        trust_model = TrustScore()


        # logit feature
        model.eval()
        loader = dloader.train_loader
        _, _, train_targets, _, train_features, _ = test(model, loader)
        train_features = np.array(train_features)
        train_targets = np.array(train_targets)
        # print('train_features', train_features.shape)
        trust_model.fit(train_features, train_targets)
        _, _, _, _, test_features, _ = test(model, dloader.test_loader)
        _, _, _, _, val_features, _ = test(model, test_val_loader)
        test_features = np.array(test_features)
        val_features = np.array(val_features)

        
        scores = trust_model.get_score(test_features, base_pred_labels)
        val_scores = trust_model.get_score(val_features, base_val_labels)

        res = eval_res(scores, base_pred_corrects)
        method_scores = {
            'val': val_scores,
            'test': scores
        }
    elif method == 'tcp':
        all_tcp_scores = tcp_training(features_extractor, model, dloader.train_loader, 512, test_loaders=[dloader.test_loader, test_val_loader])

        tcp_scores, val_tcp_scores = all_tcp_scores

        res = eval_res(tcp_scores, base_pred_corrects)
        method_scores = {
            'val': val_tcp_scores,
            'test': tcp_scores
        }

    method_scores['test_corrects'] = base_pred_corrects
    method_scores['val_corrects'] = base_val_corrects

    if cached_models is None:
        weighted_res, trained_models = mis_base.get_weighted_res(method_scores, args, device, metrics=args.choose_metric, task_config=task_config, debug=base_pred_scores)
        cached_models = trained_models
    else:
        weighted_res, _ = mis_base.get_weighted_res(method_scores, args, device, metrics=args.choose_metric, cached_models=cached_models, task_config=task_config)

    out = {
        'method': method,
        'baseline': res,
    }
    for metric, w_res in weighted_res.items():
        out[f'weighted_{metric}'] = w_res

    all_res.append(out)
    # write to txt
    with open(f'res_dir/{args.save_file}', 'a') as f:
        out['config'] = args.config_path
        out['choose_metric'] = args.choose_metric
        f.write(json.dumps(out) + '\n')


for res in all_res:
    print('method:', res['method'])
    print('baseline:', res['baseline'])

    for metric in args.choose_metric:
        out[f'weighted_{metric}'] = res[f'weighted_{metric}']
        print(f'weighted_{metric}:', res[f'weighted_{metric}'])
    