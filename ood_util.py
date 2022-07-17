from torch.utils.data import DataLoader
import numpy as np

from rot_learner.dataset import RotDataset
from rot_learner.dataset_trans import TransDataset
# from rot_learner.util import eval_res, retrieve_data
from rot_learner.data_util import preprocess_config
from rot_learner.data_util import get_ood_data, entropy


import copy
def get_ood_scores(ood_name, args, device, cached_models=None):
    ood_data, ood_targets = get_ood_data(ood_name)

    model, features_extractor, dloader,\
    trainset_data, trainset_targets,\
    valset_data, valset_targets,\
    dataset, class_num = preprocess_config(args.config_path, args, device)

    PRED_ROTS = [0, 90, 180, 270]
    PRED_TASK_NUM = len(PRED_ROTS)
    
    if dataset == 'stl10':
        TRANS = [(0, 0), (0, -32), (0, 32), (-32, 0), (32, 0)]
    else:
        TRANS = [(0, 0), (0, -8), (0, 8), (-8, 0), (8, 0)]

    TRANS_TASK_NUM = len(TRANS)

    ood_ds = RotDataset(ood_data, PRED_ROTS, ood_targets, dataset=dataset, train_mode=False)
    oodloader = DataLoader(ood_ds, batch_size=128, shuffle=False, pin_memory=False, num_workers=3)

    trans_ood_ds = TransDataset(ood_data, TRANS, ood_targets, dataset=dataset, train_mode=False)
    trans_oodloader = DataLoader(trans_ood_ds, batch_size=128, shuffle=False, pin_memory=False, num_workers=3)


    # rot predict model
    from rot_learner import main_ood, main_cinic10, main_stl10

    if dataset == 'cinic10':
        main_ood = main_cinic10
    if dataset == 'stl10':
        main_ood = main_stl10

    net = cached_models['rot_net']
    trans_net = cached_models['trans_net']

    ood_cls_res, ood_aug_res = main_ood.test(net, model, oodloader, class_num, PRED_TASK_NUM, device)
    ood_pred_scores, ood_pred_corrects, ood_gt_labels, ood_pred_labels, ood_pred_scores_all, ood_pred_scores_for_gts, ood_pred_features, ood_pred_logits, ood_pred_loss_all = ood_cls_res
    ood_rot_scores, ood_rot_accs, ood_rot_scores_full, ood_rot_scores_all, ood_aug_features, ood_aug_logits, ood_all_rot_corrects, ood_aug_loss_all = ood_aug_res


    trans_cls_res, trans_aug_res = main_ood.test(trans_net, model, trans_oodloader, class_num, TRANS_TASK_NUM, device)
    trans_scores, trans_accs, trans_scores_full, trans_scores_all, trans_aug_features, trans_aug_logits, all_trans_corrects, trans_aug_loss_all = trans_aug_res

 
    return {
        'pred': np.array(ood_pred_scores),
        'pred_all': np.array(ood_pred_scores_all),
        'rot': np.array(ood_rot_scores_full)[:, 0],
        'trans': np.array(trans_scores_full)[:, 0],
        'logit': np.array(ood_pred_logits)
    }



