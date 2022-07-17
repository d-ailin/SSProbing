import torch.nn as nn
from tqdm.std import tqdm
from torch.optim.sgd import SGD
import torch.nn.functional as F
from collections import OrderedDict
import torch
import copy



def train(extractor, rot_num, in_size, epochs, dataloader, device, dataset, val_loader=None, base_net=None):
    select_Comb = None
    if dataset == 'stl10':
        from .main_stl10 import Comb
        select_Comb = Comb
        lr = 0.1
        # lr = 0.01
    elif dataset == 'cinic10':
        from .main_cinic10 import Comb
        select_Comb = Comb
        lr = 1e-2

    else:
        from .main import Comb
        select_Comb = Comb
        lr = 0.1

    
    net = select_Comb(extractor, in_size, rot_num).to(device)


    optimizer = SGD(net.fc.parameters(), lr=lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    val_perform = 0
    best_net = net
    torlerence=1
    tor_i = 0
    for epoch_i in range(epochs):
        net.train()
        net.disable_bn()
        
        train_loss = 0
        train_cls_loss = 0
        train_aug_loss = 0
        correct = 0
        aug_correct = 0
        total = 0
        aug_total = 0
        loop = tqdm(dataloader)
        for batch_idx, (inputs, aug_inputs, targets, aug_targets) in enumerate(loop):
            inputs, targets = inputs.to(device), targets.to(device)
            aug_inputs, aug_targets = aug_inputs.to(device), aug_targets.to(device)

            bs = inputs.shape[0]
            img_shape = aug_inputs.shape[2:]
            aug_inputs = aug_inputs.reshape((-1, *img_shape))
            aug_targets = aug_targets.reshape(-1)

            optimizer.zero_grad()
            aug_out = net(aug_inputs)

            aug_loss = F.cross_entropy(aug_out, aug_targets)

            
            loss = aug_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_aug_loss += aug_loss.item()
            

            total += targets.size(0)

            _, aug_predicted = aug_out.max(1)
            aug_total += aug_targets.size(0)
            aug_correct += aug_predicted.eq(aug_targets).sum().item()

            loop.set_description(f"Epoch {epoch_i+1}/{epochs}")
            loop.set_postfix(
                OrderedDict(
                    {
                        "aug_acc": f"{(aug_correct / aug_total):05.2%}",
                    }
                )
            )
            loop.update()

        if val_loader is not None:
            _, aug_res = test(net, base_net, val_loader, 10, rot_num, device)
            sample_scores, sample_corrects, sample_full_scores, sample_all_scores, aug_features, aug_logits, all_sample_corrects, aug_loss_all = aug_res

            val_all_aug_acc =  (all_sample_corrects.sum() / len(all_sample_corrects)) * 100
            val_0_aug_acc = (all_sample_corrects.reshape(-1, rot_num)[:, 0].sum() / len(all_sample_corrects) * rot_num) * 100
            # print('val all aug acc:', val_all_aug_acc )
            # print('val 0 aug acc:', val_0_aug_acc)

            if val_all_aug_acc > val_perform:
                val_perform = val_all_aug_acc
                best_net = copy.deepcopy(net)
                tor_i = 0
            else:
                if tor_i >= torlerence:
                    # print('stop at epoch', epoch_i)
                    # print('best aug acc', val_perform)
                    break
                else:
                    tor_i += 1
        scheduler.step()

    return best_net

def test(net, base_net, dataloader, class_num, rot_num, device):
    net.eval()
    base_net.eval()

    test_loss = 0
    test_cls_loss = 0
    test_aug_loss = 0
    correct = 0
    total = 0
    pred_scores = []
    pred_corrects = []
    gt_labels = []
    pred_labels = []
    pred_scores_all = []
    pred_scores_for_gts = []
    pred_features = []
    pred_logits = []
    pred_loss_all = []

    aug_total = 0
    aug_correct = 0
    sample_corrects = []
    all_sample_corrects = []
    sample_scores = []
    sample_full_scores = []
    sample_all_scores = []
    corrects_each_pred = torch.zeros(rot_num).to(device)
    aug_features = []
    aug_logits = []
    aug_loss_all = []

    loop = tqdm(dataloader)

    with torch.no_grad():
        for batch_idx, (inputs, aug_inputs, targets, aug_targets) in enumerate(loop):
            inputs, targets = inputs.to(device), targets.to(device)
            aug_inputs, aug_targets = aug_inputs.to(device), aug_targets.to(device)

            bs = inputs.shape[0]
            img_shape = aug_inputs.shape[2:]
            aug_inputs = aug_inputs.reshape((-1, *img_shape))
            aug_targets = aug_targets.reshape(-1)

            cls_out = base_net(inputs)
            aug_out = net(aug_inputs)

            cls_loss = F.cross_entropy(cls_out, targets)
            aug_loss = F.cross_entropy(aug_out, aug_targets)

            loss = aug_loss

            test_loss += loss.item()
            test_cls_loss += cls_loss.item()
            test_aug_loss += aug_loss.item()

            _, predicted = cls_out.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            gt_labels += targets.detach().cpu().tolist()
            pred_scores += F.softmax(cls_out, dim=-1).max(1)[0].detach().cpu().tolist()
            pred_scores_all += F.softmax(cls_out, dim=-1).reshape(-1, class_num).detach().cpu().tolist()
            pred_corrects += predicted.eq(targets).detach().cpu().tolist()
            pred_labels += predicted.detach().cpu().tolist()
            pred_scores_for_gts += F.softmax(cls_out, dim=-1)\
                .reshape(-1, class_num)[torch.arange(cls_out.shape[0]), targets]\
                .detach().cpu().tolist()
            pred_loss_all += F.cross_entropy(cls_out, targets, reduction='none').detach().cpu().tolist()

            _, aug_predicted = aug_out.max(1)
            aug_total += aug_targets.size(0)
            aug_correct += aug_predicted.eq(aug_targets).sum().item()

            pred_features.append(net.backbone(inputs).detach().cpu())
            pred_logits.append(cls_out)

            aug_features.append(net.backbone(aug_inputs).detach().cpu())
            aug_logits.append(aug_out)


            
            gt_mask = torch.eye(rot_num).to(device).bool()
            corrects_each_pred += aug_predicted.eq(aug_targets).reshape(-1, rot_num).sum(0)

            sample_scores += torch.masked_select(F.softmax(aug_out, -1).reshape(-1, rot_num, rot_num), gt_mask).reshape(-1, rot_num).mean(-1).cpu().tolist()
            sample_full_scores += torch.masked_select(F.softmax(aug_out, -1).reshape(-1, rot_num, rot_num), gt_mask).reshape(-1, rot_num).cpu().tolist()
            sample_all_scores += F.softmax(aug_out, -1).reshape(-1, rot_num * rot_num).cpu().tolist()
            # sample_scores += F.softmax(outputs, -1).max(1)[0].reshape(-1, rot_num).mean(-1).cpu().tolist()
            sample_corrects += (aug_predicted.eq(aug_targets).reshape(-1, rot_num).sum(-1) / rot_num).cpu().tolist()
            all_sample_corrects += aug_predicted.eq(aug_targets).cpu().tolist()

            aug_loss_all += F.cross_entropy(aug_out, aug_targets, reduction='none').detach().cpu().tolist()


            loop.set_postfix(
                OrderedDict(
                    {
                        "acc": f"{(correct / total):05.2%}",
                        "aug_acc": f"{(aug_correct / aug_total):05.2%}",
                    }
                )
            )
            loop.update()


    pred_features = torch.cat(pred_features, 0).cpu().numpy()
    aug_features = torch.cat(aug_features, 0).cpu().numpy()
    pred_logits = torch.cat(pred_logits, 0).cpu().numpy()
    aug_logits = torch.cat(aug_logits, 0).cpu().numpy()
    all_sample_corrects = torch.tensor(all_sample_corrects).cpu().numpy()

    cls_res = (pred_scores, pred_corrects, gt_labels, pred_labels, pred_scores_all, pred_scores_for_gts, pred_features, pred_logits, pred_loss_all)
    aug_res = (sample_scores, sample_corrects, sample_full_scores, sample_all_scores, aug_features, aug_logits, all_sample_corrects, aug_loss_all)

    return cls_res, aug_res