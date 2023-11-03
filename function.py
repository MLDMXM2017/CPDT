
import copy
import csv
import logging
import os
import sys

import numpy as np
import scipy.signal as sg

from scipy.stats import gaussian_kde
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score


def get_dirs(path):
    dir_dict = {
        'res_dir': path,
        'log_dir': "%s/Log" % path,
        'pic_dir': "%s/Picture" % path,
        'mod_dir': '%s/Model' % path,
        'fi_dir': "%s/Feature_Importance" % path,
        'ri_dir': "%s/Rule_Info" % path,
        'pi_dir': "%s/Pred_Info" % path,
        'roc_dir': "%s/ROC" % path
    }
    for d in dir_dict.values():
        if not os.path.exists(d):
            os.mkdir(d)
    return dir_dict


def get_logger(name, filepath, write=True, output=True):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s -  %(message)s')

    if write:
        file_handler = logging.FileHandler(filepath)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if output:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_metric_dict(true, pred, proba):
    report = classification_report(true, pred, output_dict=True)
    performance = {
        'ACC': np.round(report['accuracy'], 4),
        'F1': np.round(report['macro avg']['f1-score'], 4),
        'REC0': np.round(report["0"]['recall'], 4),
        'REC1': np.round(report["1"]['recall'], 4),
        'PRE0': np.round(report["0"]['precision'], 4),
        'PRE1': np.round(report["1"]['precision'], 4),
        'AUC': np.round(roc_auc_score(true, proba[:, 1]), 4),
        'AUPR': np.round(average_precision_score(true, proba[:, 1]), 4),
    }
    return performance


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def process_result(logger, file_name, perf_s, met_name, res_path):

    # output result
    logger.info("Final result on dataset {}: ".format(file_name))
    perf_mean = np.round(np.mean(np.array(perf_s), axis=0), 4)
    perf_std = np.round(np.std(np.array(perf_s), axis=0), 4)
    perf_ms = np.array(['{:.4f}\u00B1{:.4f}'.format(perf_mean[i], perf_std[i]) for i in range(len(perf_mean))])
    for i in range(len(met_name)):
        logger.info("{}\tmean:{:.4f}\tstd:{:.4f}".format(met_name[i], perf_mean[i], perf_std[i]))

    # save result
    row_name = np.array([[''] + [str(task_id + 1) for task_id in range(len(perf_s))] +
                         ['mean', 'std', 'mean\u00B1std']]).T
    col_name = np.array([met_name])
    save_data = np.hstack((row_name, np.vstack((col_name, np.vstack((perf_s, perf_mean, perf_std, perf_ms))))))
    with open('{}/{}.csv'.format(res_path, file_name), mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(save_data)


def calc_entropy_gain(node_y, left_y, right_y, class_labels=None):

    ent = entropy(node_y, class_labels)
    left_ent = entropy(left_y, class_labels)
    right_ent = entropy(right_y, class_labels)

    gain = ent - len(left_y) / len(node_y) * left_ent - len(right_y) / len(node_y) * right_ent
    return gain


def calc_gini_index(node_y, left_y, right_y, class_labels = None):

    left_gini = gini_impurity(left_y, class_labels)
    right_gini = gini_impurity(right_y, class_labels)

    gini = len(left_y) / len(node_y) * left_gini + len(right_y) / len(node_y) * right_gini
    return gini


def calc_impurity(y, criterion):
    if criterion == 'gini':
        impurity = gini_impurity(y)
    elif criterion == 'entropy':
        impurity = entropy(y)
    else:
        raise TypeError("Unknown criterion!")
    return impurity


def gini_impurity(y, labels=None):
    if labels is None:
        labels, counts = np.unique(y, return_counts=True)
    else:
        counts = np.array([len(np.where(y == label)[0]) for label in labels])
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)


def entropy(y, labels=None):
    if labels is None:
        labels, counts = np.unique(y, return_counts=True)
    else:
        counts = np.array([len(np.where(y == label)[0]) for label in labels])
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs))


def adapt_split(x, bw_method='scott'):
    # bw_method = x.std() * (4.0 / (3.0 * len(x))) ** (1 / 5)
    sort_x = np.sort(x)
    unique_x = np.unique(sort_x)
    kde = gaussian_kde(sort_x, bw_method=bw_method)
    tmp_x = (unique_x[1:] + unique_x[:-1]) / 2. if len(unique_x) > 1 else copy.deepcopy(unique_x)
    refer_x = np.sort(np.hstack((unique_x, tmp_x)))
    density = kde(refer_x)
    minima_indices = sg.argrelmin(density)[0]
    maxima_indices = sg.argrelmax(density)[0]
    divs = refer_x[minima_indices]
    centers = refer_x[maxima_indices]
    return x, sort_x, unique_x, tmp_x, density, divs, centers


def grid_search(x, y, grid_array, grid_label, cluster_info):

    directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    # directions = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
    for d in directions:
        i, j = x + d[0], y + d[1]

        if i < 0 or i >= grid_array.shape[0] or j < 0 or j >= grid_array.shape[1]:
            continue

        if grid_label[i, j] != 'O' or grid_array[i, j].n_sample == 0:
            continue

        if grid_array[i, j].n_sample < grid_array[x, y].n_sample:
            grid_label[i, j] = grid_label[x, y]
            cluster_info.append([i, j])
            grid_search(i, j, grid_array, grid_label, cluster_info)
    return


def output_node_info(logger, node, n_class=2):
    if node.is_leaf:
        logger.info("leaf id: {}".format(node.node_id))
        format_str = "depth: {:<3d}  sample_num: {:<4d}  class_dist: {:>4d}:{:<4d}"
        format_str += "  impurity: {:.4f}  reason: {}"
        args = [node.depth, node.sample_num]
        args += list(node.class_dist.values())
        args += [node.impurity, node.reason]
        output_str = format_str.format(*args)
        logger.info(output_str)
    else:
        logger.info("node id: {}".format(node.node_id))
        format_str = "depth: {:<3d}  sample_num: {:<4d}  class_dist: {:>4d}:{:<4d}"
        format_str += "  impurity: {:.4f}  gain: {:.4f}  sel_fea: {:<4d}"
        args = [node.depth, node.sample_num, list(node.class_dist.values())]
        args += [node.impurity, node.gain, node.split_index]
        output_str = format_str.format(*args)
        logger.info(output_str)
    return


def to_pairing_vec(x, pair_num=2):
    n_feature = int(x.shape[1] / pair_num)
    paired_x = np.zeros((x.shape[0], n_feature, pair_num))
    for i in range(x.shape[0]):
        for j in range(n_feature):
            paired_x[i, j, 0], paired_x[i, j, 1] = x[i, j], x[i, j + n_feature]
    return paired_x


