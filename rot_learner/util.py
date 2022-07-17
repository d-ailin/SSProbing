from scipy.stats import wasserstein_distance
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import seaborn as sns
from scipy.stats import pearsonr
from torch.optim.adam import Adam

from rot_learner.main_cal import Comb
# import cv2

def retrieve_data(dataset_name, dataset):
    if dataset_name == 'cifar10':
        return dataset.data, dataset.targets
    if dataset_name == 'svhn':
        return dataset.data, dataset.labels

def show_res(scores, corrects, save_img_path, tag, config={}, other_config={}):
    xlabel = other_config['xlabel']

    scores = np.array(scores).reshape(-1)
    corrects = np.array(corrects).reshape(-1)

    correct_mask = corrects == 1
    wrong_mask = corrects == 0    

    dist = wasserstein_distance(scores[wrong_mask], scores[correct_mask])
    # print('{} dist:'.format(tag), dist)

    font = {'family' : 'normal',
        # 'weight' : 'bold',
            'size'   : 12}

    mpl.rc('font', **font)

    # sns.color_palette("colorblind")
    sns.color_palette("Set2")
    plt.figure(figsize=(4.8, 3))
    sns.distplot(scores[correct_mask], label='Successes', **config)
    sns.distplot(scores[wrong_mask], label='Errors', **config)
    plt.xlim(0, 1)

    # plt.title('cifar10 {} test scores distribution: scores for correct vs scores for misclassified'.format(tag))
    plt.xlabel(xlabel)
    plt.ylabel('Relative Density')
    plt.legend()
    plt.title(other_config['title'])

    plt.tight_layout()
    if save_img_path is None:
        plt.show()
    else:
        plt.savefig('{}/dist_{}.pdf'.format(save_img_path, tag))
        plt.savefig('{}/dist_{}.png'.format(save_img_path, tag))


def show_ssl_cls(pred_scores, pred_scores_for_gts, pred_corrects, rot_scores, rot_accs, gt_labels, epoch, save_img_path, tag, config={}):
    plt.style.use('seaborn-whitegrid')

    font = {
        # 'family' : 'normal',
        # 'weight' : 'bold',
            'size'   : 12}

    mpl.rc('font', **font)

    xlabel = config['xlabel']

    sns.color_palette("colorblind")
    plt.figure(figsize=(3.5, 3.5))

    rot_scores_left_bins = np.arange(0, 1, 0.1)
    rot_scores_right_bins = np.arange(0.1, 1.1, 0.1)
    X = np.arange(len(rot_scores_left_bins))

    xtick_arr = []
    res_arr = []
    count_arr = []
    for rot_bin_left, rot_bin_right in zip(rot_scores_left_bins, rot_scores_right_bins):
        rot_mask = (rot_scores >= rot_bin_left) & (rot_scores < rot_bin_right)
        pred_total = len(rot_scores[rot_mask])
        pred_correct_num = sum(pred_corrects[rot_mask])
        if pred_total > 0:
            pred_acc = pred_correct_num / pred_total
        else:
            pred_acc = 0
        res_arr.append(pred_acc)
        xtick_arr.append( round(rot_bin_left, 2))
        count_arr.append(pred_total)

    plt.bar(X+0.5, res_arr, width=1, alpha=config['alpha'])
    # print(config['title'], xtick_arr, res_arr, count_arr)

    x_index = np.array(X.tolist() + [10])
    x_ticks = np.array(xtick_arr + [1.0])
    plt.xticks(x_index[::2], x_ticks[::2])

    ax = plt.gca()
    temp = ax.xaxis.get_ticklabels()
    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % 1 != 0:
            label.set_visible(False)
    # for index,data in enumerate(res_arr):
    #     if data > 0:
    #         plt.text(x=X[index]-0.05 , y =data+0.01 , s=f"{round(data,2)}" , fontdict=dict(fontsize=10))
    plt.xlim(0, 10)
    plt.ylim(bottom=max(0, min(res_arr) - 0.1), top=1)
    # plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel('Classfication Accuracy')
    plt.tight_layout()
    plt.title(config['title'])

    # plt.title('cifar10 {} trained epoch({}):  rot pred conf vs classification pred acc'.format(tag, epoch))
    if save_img_path is None:
        plt.show()
    else:
        plt.savefig('{}/rot_score_vs_pred_acc_epoch_{}_{}.png'.format(save_img_path, epoch, tag))
        plt.savefig('{}/rot_score_vs_pred_acc_epoch_{}_{}.pdf'.format(save_img_path, epoch, tag))


def get_is_pos(ind_scores, ood_scores, order):
    assert order in ["largest2smallest", "smallest2largest"]
    scores = np.concatenate((ind_scores, ood_scores))
    is_pos = np.concatenate((np.ones(len(ind_scores), dtype="bool"), np.zeros(len(ood_scores), dtype="bool")))
    
    # shuffle before sort
    random_idx = np.random.permutation(list(range(len(scores))))
    scores = scores[random_idx]
    is_pos = is_pos[random_idx]

    idxs = scores.argsort()
    if order == "largest2smallest":
        idxs = np.flip(idxs)
    is_pos = is_pos[idxs]
    return is_pos

def fpr_at_tpr(ind_scores, ood_scores, order, tpr = 0.95):
    assert order in ["largest2smallest", "smallest2largest"]
    is_pos = get_is_pos(ind_scores, ood_scores, order)
    P = len(ind_scores)
    N = len(ood_scores)
    TP = 0
    FP = 0
    for _is_pos in is_pos:
        if _is_pos:
            TP += 1
        else:
            FP += 1
        TPR = TP / P
        if TPR >= tpr:
            FPR = FP / N
            return FPR

def tnr_at_tpr(ind_scores, ood_scores, order, tpr = 0.95):
    assert order in ["largest2smallest", "smallest2largest"]
    is_pos = get_is_pos(ind_scores, ood_scores, order)
    P = len(ind_scores)
    N = len(ood_scores)
    TP = 0
    TN = N
    for _is_pos in is_pos:
        if _is_pos:
            TP += 1
        else:
            TN -= 1
        TPR = TP / P
        if TPR >= tpr:
            TNR = TN / N
            return TNR

from sklearn.metrics import average_precision_score, roc_auc_score
def eval_res(scores, labels, verbose=True):
    scores = np.array(scores).reshape(-1)
    labels = np.array(labels).reshape(-1)
    # AUPR Err
    err_aupr = average_precision_score((labels + 1)%2, -scores)
    # AUPR Succ
    suc_aupr = average_precision_score(labels, scores)
    # auroc
    auroc = roc_auc_score(labels, scores)

    correct_scores = scores[labels == 1]
    wrong_scores = scores[labels == 0]
    fpr = fpr_at_tpr(correct_scores, wrong_scores, 'largest2smallest', 0.95)
    tnr = tnr_at_tpr(correct_scores, wrong_scores, 'largest2smallest', 0.95)

    # print('fpr_at_tpr 95 ↓:', fpr)
    # print('tnr_at_tpr 95 ↑:', tnr)
    # print('err aupr ↑:', err_aupr)
    # print('succ aupr ↑:', suc_aupr)
    # print('auroc ↑:', auroc)

    res = {
        'fpr': fpr,
        'tnr': tnr,
        'err_aupr': err_aupr,
        'suc_aupr': suc_aupr,
        'auroc': auroc,
    }

    if verbose:
        print({
            'fpr ↓': fpr,
            'tnr ↑': tnr,
            'err_aupr ↑': err_aupr,
            'suc_aupr ↑': suc_aupr,
            'auroc ↑': auroc,
        })

    return res

def get_err_aupr(scores, labels):
    scores = np.array(scores).reshape(-1)
    labels = np.array(labels).reshape(-1)
    # AUPR Err
    err_aupr = average_precision_score((labels + 1)%2, -scores)

    return err_aupr

def get_auroc(scores, labels):
    scores = np.array(scores).reshape(-1)
    labels = np.array(labels).reshape(-1)
    # AUPR Err
    auroc = roc_auc_score(labels, scores)

    return auroc

def get_succ_aupr(scores, labels):
    scores = np.array(scores).reshape(-1)
    labels = np.array(labels).reshape(-1)
    # AUPR Err
    suc_aupr = average_precision_score(labels, scores)

    return suc_aupr
