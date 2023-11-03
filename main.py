# -*- coding: gbk -*-

import datetime
import os
import pickle
import sys

import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold

from Datasets.data_loader import load_data
from function import get_dirs, get_logger, get_metric_dict, make_dir, process_result
from PairedRandomForest import PairedRandomForest

sys.setrecursionlimit(10000)

# Experiment parameters
n_repeat = 1        # Repeat Experiment
n_fold = 5          # k-fold cross valiadation
random_state = 42   # Random seed
n_process = n_repeat * n_fold

# Datasets
file_names = ['SL_exp']
version = 'CPRF'

# Data directory
ori_dir = 'Datasets'
dat_dir = make_dir("data")
src_dir = make_dir("%s/source_data" % dat_dir)
spl_dir = make_dir("%s/split_indice" % dat_dir)

# Result directory
res_path = make_dir("Results")
log_name = '%s/%s.txt' % (res_path, version)
LOGGER = get_logger('%s' % version, log_name)


def do_exp(task_id, x, y, file_name, dir_dict):

    LOGGER.info("Run task {:>2d}, pid is {}...".format(task_id, os.getpid()))
    start = datetime.datetime.now()

    # Prepare data
    split_indices = np.loadtxt(f'{spl_dir}/split_indice_{file_name}_{task_id}.csv', int, delimiter=',')
    train_id, test_id = np.where(split_indices[0])[0], np.where(split_indices[1])[0]
    x_train, y_train = x[train_id], y[train_id]
    x_test, y_test = x[test_id], y[test_id]

    # Train model
    model = PairedRandomForest(file_name=file_name, exp_id=task_id, random_state=random_state,
                               log_dir=dir_dict['log_dir'], pic_dir=dir_dict['pic_dir'])
    model.fit(x_train, y_train)
    pickle.dump(model, open(f"{dir_dict['mod_dir']}/model_{task_id}.dat", "wb"))

    # model = pickle.load(open(f"{dir_dict['mod_dir']}/model_{task_id}.dat", "rb"))

    feature_importance = model.feature_importance
    np.savetxt(f"{dir_dict['fi_dir']}/fi_{task_id}.csv", feature_importance, fmt='%s', delimiter=',')

    end = datetime.datetime.now()
    time_cost = end - start
    LOGGER.info("Task {:>2d}, finished! Cost time: {}".format(task_id, time_cost))

    # Performance on test data
    y_proba = model.predict_proba(x_test)
    y_pred = np.argmax(y_proba, axis=1)
    perf_dict = get_metric_dict(y_test, y_pred, y_proba)
    output_str = "Task {:>2d} on TestSet:  ".format(task_id)
    for met, perf in perf_dict.items():
        if met not in ['REC0', 'PRE0']:
            output_str += "{}={:.4f}  ".format(met, perf)
    LOGGER.info(output_str)

    del x_train, y_train, x_test, y_test, model, y_proba, y_pred

    return list(perf_dict.keys()), list(perf_dict.values())


def main():

    for file_name in file_names:

        # Load data
        data_path = '%s/%s.csv' % (src_dir, file_name)
        data = load_data(ori_dir, file_name)
        np.savetxt(data_path, data, fmt='%s', delimiter=',')
        x, y = data[:, :-1], data[:, -1].astype(int)

        # Perform 5-fold cross-validation
        skf = RepeatedStratifiedKFold(n_splits=n_fold, n_repeats=n_repeat, random_state=random_state)
        task_id = 0
        for train_id, test_id in skf.split(x, y):
            split_indice = np.zeros((2, len(y)))
            split_indice[0, train_id], split_indice[1, test_id] = 1, 1
            np.savetxt(f'{spl_dir}/split_indice_{file_name}_{task_id}.csv', split_indice.astype(int),
                       fmt='%s', delimiter=',')
            task_id += 1

    for file_name in file_names:
        LOGGER.info("########################################################################################")
        LOGGER.info("DataSet: {}".format(file_name))

        dir_dict = get_dirs(f"{res_path}/{file_name}")

        # Load data
        data_path = '%s/%s.csv' % (src_dir, file_name)
        data = np.loadtxt(data_path, float, delimiter=',')
        x, y = data[:, :-1], data[:, -1].astype(int)

        # Perform 5-fold cross-validation
        LOGGER.info("Perform experiments for each task...")
        results = [do_exp(task_id, x, y, file_name, dir_dict) for task_id in range(n_process)]

        # Collate experimental results
        met_name, perf_s = None, []
        for task_id in range(n_process):
            met_name, perf = results[task_id]
            perf_s.append(perf)

        process_result(LOGGER, f"{file_name}", perf_s, met_name, res_path)


if __name__ == '__main__':

    main()


