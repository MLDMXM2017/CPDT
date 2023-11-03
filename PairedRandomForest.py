
import random

import multiprocessing as mp
import numpy as np

from function import get_logger, make_dir, to_pairing_vec
from PairedDecisionTree import PairedDecisionTree


class PairedRandomForest:
    def __init__(self,
                 file_name='Unknown', exp_id=-1, random_state=42, use_mp=True,
                 log_dir='ForestInfo', pic_dir='Picture',
                 n_estimators=25, n_jobs=-1, max_features=0.05):

        self.file_name = file_name
        self.exp_id = exp_id
        self.random_state = random_state
        self.use_mp = use_mp

        self.log_dir = log_dir
        self.pic_dir = pic_dir

        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.max_features = max_features

        self.feature_indexs = None
        self.sample_indexs = None
        self.exchange_indexs = None

        self.estimators = None
        self.feature_importance = None

        self.log_path = make_dir('%s/forest_%d' % (log_dir, exp_id))
        self.pic_path = make_dir('%s/forest_%d' % (pic_dir, exp_id))

        log_name = "%s/forest_%d.txt" % (log_dir, exp_id)
        self.logger = get_logger('CPRF_%d' % exp_id, log_name)

        random.seed(random_state)
        np.random.seed(random_state)

    def fit(self, x, y):

        x = to_pairing_vec(x)

        sample_num, feature_num, pair_num = x.shape

        if isinstance(self.max_features, str):
            if self.max_features == 'all':
                sub_feature_num = feature_num
            elif self.max_features == 'sqrt':
                sub_feature_num = int(np.sqrt(feature_num))
            elif self.max_features == 'log2':
                sub_feature_num = int(np.log2(feature_num))
            else:
                print("Error max_features type!")
        elif isinstance(self.max_features, float):
            sub_feature_num = int(feature_num * self.max_features)
        else:
            raise TypeError("Error max_features type!")

        feature_indexs = np.array([
            np.random.choice(np.arange(0, feature_num, 1), sub_feature_num, replace=False)
            for _ in range(self.n_estimators)])
        self.feature_indexs = feature_indexs

        sub_sample_num = int(sample_num * 1.0)
        class_labels, class_counts = np.unique(y, return_counts=True)
        class_weights = (class_counts / np.sum(class_counts))[::-1]
        sample_weights = np.zeros(sample_num)
        for class_label, class_weight in zip(class_labels, class_weights):
            sample_weights[np.where(y == class_label)[0]] = class_weight
        sample_weights /= sample_weights.sum()
        sample_indexs = [
            np.random.choice(np.arange(0, sample_num, 1), sub_sample_num, replace=True, p=sample_weights)
            for _ in range(self.n_estimators)]
        self.sample_indexs = sample_indexs

        if self.use_mp:
            self.logger.info("Assign process for trees in forest...")
            pool = mp.Pool()
            tasks = [pool.apply_async(self.fit_async, args=(t_i, x, y, feature_indexs[t_i], sample_indexs[t_i]))
                     for t_i in range(self.n_estimators)]
            pool.close()
            pool.join()
            results = [task.get() for task in tasks]
        else:
            self.logger.info("Train trees in forest one by one...")
            results = [self.fit_async(t_i, x, y, feature_indexs[t_i], sample_indexs[t_i])
                       for t_i in range(self.n_estimators)]

        estimators = []
        self.feature_importance = np.zeros(feature_num)
        for tree_id in range(self.n_estimators):
            estimator = results[tree_id]
            estimators.append(estimator)
            self.feature_importance[self.feature_indexs[tree_id]] += estimator.feature_importance
        self.estimators = estimators

    def fit_async(self, tree_id, x, y, feature_index, sample_index):

        x_train, y_train = x[:, feature_index, :][sample_index], y[sample_index]

        pdt = PairedDecisionTree(
            file_name=self.file_name, exp_id=tree_id, random_state=self.random_state,
            log_dir=self.log_path, pic_dir=self.pic_path
        )
        pdt.fit(x_train, y_train)

        return pdt

    def predict(self, x):
        proba = self.predict_proba(x)
        return np.argmax(proba, axis=1)

    def predict_proba(self, x):

        if len(x.shape) == 2:
            x = to_pairing_vec(x)

        results = [self.predict_proba_async(tree_id, x) for tree_id in range(self.n_estimators)]

        probas = []
        for tree_id in range(self.n_estimators):
            proba = results[tree_id]
            probas.append(proba)

        final_proba = np.mean(np.array(probas), axis=0)
        return final_proba

    def predict_proba_async(self, tree_id, x):
        return self.estimators[tree_id].predict_proba(x[:, self.feature_indexs[tree_id], :])





