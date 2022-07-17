import torch.nn as nn
from tqdm.std import tqdm
from torch.optim.sgd import SGD
import torch.nn.functional as F
from collections import OrderedDict
import torch

class Comb(nn.Module):
    def __init__(self, extractor, in_size, class_num) -> None:
        super().__init__()
        self.backbone = extractor
        self.fc = nn.Linear(in_size, class_num)

        self.freeze_backbone()

    def freeze_backbone(self):
        # disable backbone bn and gradients
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
    def disable_bn_dp(self):
        self.backbone.eval()

    def forward(self, x):
        ft = self.backbone(x)
        return self.fc(ft)



def train(extractor, rot_num, in_size, epochs, dataloader, device):
    net = Comb(extractor, in_size, rot_num).to(device)
    optimizer = SGD(net.fc.parameters(), lr=0.1,
                        momentum=0.9, weight_decay=5e-4)
    net.train()
    # net.freeze_backbone()
    net.disable_bn_dp()
    for epoch_i in range(epochs):
        train_loss = 0
        train_cls_loss = 0
        train_aug_loss = 0
        correct = 0
        aug_correct = 0
        total = 0
        aug_total = 0
        aug_0_correct = 0
        aug_90_correct = 0
        aug_180_correct = 0
        aug_270_correct = 0

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

            aug_0_correct += aug_predicted.eq(aug_targets)[::rot_num].sum().item()
            aug_90_correct += aug_predicted.eq(aug_targets)[1::rot_num].sum().item()
            aug_180_correct += aug_predicted.eq(aug_targets)[2::rot_num].sum().item()
            aug_270_correct += aug_predicted.eq(aug_targets)[3::rot_num].sum().item()

            loop.set_description(f"Epoch {epoch_i+1}/{epochs}")
            loop.set_postfix(
                OrderedDict(
                    {
                        # "acc": f"{(correct / total):05.2%}",
                        "aug_acc": f"{(aug_correct / aug_total):05.2%}",
                        "aug_0_acc": f"{(aug_0_correct / aug_total * rot_num):05.2%}",
                        "aug_90_acc": f"{(aug_90_correct / aug_total * rot_num):05.2%}",
                        "aug_180_acc": f"{(aug_180_correct / aug_total * rot_num):05.2%}",
                        "aug_270_acc": f"{(aug_270_correct / aug_total * rot_num):05.2%}",

                    }
                )
            )
            loop.update()

    return net

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
    aug_0_correct = 0
    aug_90_correct = 0
    aug_180_correct = 0
    aug_270_correct = 0


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
            aug_0_correct += aug_predicted.eq(aug_targets)[::rot_num].sum().item()
            aug_90_correct += aug_predicted.eq(aug_targets)[1::rot_num].sum().item()
            aug_180_correct += aug_predicted.eq(aug_targets)[2::rot_num].sum().item()
            aug_270_correct += aug_predicted.eq(aug_targets)[3::rot_num].sum().item()

            pred_features.append(net.backbone(inputs).detach().cpu())
            pred_logits.append(cls_out)

            aug_features.append(net.backbone(aug_inputs).detach().cpu())
            aug_logits.append(aug_out)

            
            gt_mask = torch.eye(rot_num).to(device).bool()
            corrects_each_pred += aug_predicted.eq(aug_targets).reshape(-1, rot_num).sum(0)

            sample_scores += torch.masked_select(F.softmax(aug_out, -1).reshape(-1, rot_num, rot_num), gt_mask).reshape(-1, rot_num).mean(-1).cpu().tolist()
            sample_full_scores += torch.masked_select(F.softmax(aug_out, -1).reshape(-1, rot_num, rot_num), gt_mask).reshape(-1, rot_num).cpu().tolist()
            sample_all_scores += F.softmax(aug_out, -1).reshape(-1, rot_num * rot_num).cpu().tolist()
            sample_corrects += (aug_predicted.eq(aug_targets).reshape(-1, rot_num).sum(-1) / rot_num).cpu().tolist()
            all_sample_corrects += aug_predicted.eq(aug_targets).cpu().tolist()

            aug_loss_all += F.cross_entropy(aug_out, aug_targets, reduction='none').detach().cpu().tolist()


            loop.set_postfix(
                OrderedDict(
                    {
                        "acc": f"{(correct / total):05.2%}",
                        "aug_acc": f"{(aug_correct / aug_total):05.2%}",
                        "aug_0_acc": f"{(aug_0_correct / aug_total * rot_num):05.2%}",
                        "aug_90_acc": f"{(aug_90_correct / aug_total * rot_num):05.2%}",
                        "aug_180_acc": f"{(aug_180_correct / aug_total * rot_num):05.2%}",
                        "aug_270_acc": f"{(aug_270_correct / aug_total * rot_num):05.2%}",

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