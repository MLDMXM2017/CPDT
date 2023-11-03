
import numpy as np
import warnings

from function import calc_entropy_gain, calc_gini_index, calc_impurity, get_logger, to_pairing_vec
from function import adapt_split, grid_search, make_dir, output_node_info
from Grid import Grid
from TreeNode import TreeNode

np.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")


class PairedDecisionTree:
    def __init__(self,
                 file_name='Unknown', exp_id=-1, random_state=42,
                 log_dir='TreeInfo', pic_dir='Picture',
                 criterion='gini', splitter='best', feature_select='sequence', max_features='all',
                 max_depth=None, min_samples_split=2, min_samples_leaf=1, minor_class_num=1, min_impurity=0.0):

        self.file_name = file_name
        self.exp_id = exp_id
        self.random_state = random_state

        self.log_dir = log_dir
        self.pic_dir = pic_dir

        self.criterion = criterion
        self.splitter = splitter
        self.feature_select = feature_select
        self.max_features = max_features

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.minor_class_num = minor_class_num
        self.min_impurity = min_impurity

        self.head_node = None
        self.depth = 0
        self.node_num = 0
        self.leaf_num = 0
        self.feature_importance = None

        self.sample_num = None
        self.class_label = None
        self.n_class = None

        self.pic_path = make_dir('%s/tree_%d' % (pic_dir, exp_id))

        log_name = "%s/tree_%d.txt" % (log_dir, exp_id)
        self.logger = get_logger('CPDT_%d' % exp_id, log_name, output=False)

    def fit(self, x, y):

        if len(x.shape) == 2:
            x = to_pairing_vec(x)

        self.head_node = TreeNode(node_id="h")
        self.depth = 0
        self.feature_importance = np.zeros(x.shape[1])

        self.sample_num = x.shape[0]
        self.class_label = np.unique(y)
        self.n_class = len(self.class_label)

        self.grow(x, y, self.head_node, 0)

    def grow(self, node_x, node_y, node, depth):

        if depth > self.depth:
            self.depth = depth
        self.node_num += 1

        class_dist = {label: len(np.where(node_y == label)[0]) for label in self.class_label}
        distribution = sorted(class_dist.items(), key=lambda d: d[0])
        class_prob = np.array([v / len(node_y) for (k, v) in distribution])

        node.depth = depth
        node.class_dist = class_dist
        node.class_prob = class_prob
        node.sample_num = len(node_y)
        node.impurity = calc_impurity(node_y, self.criterion)

        if self.is_leaf(node, node_y):
            node.is_leaf = True
            self.leaf_num += 1
            output_node_info(self.logger, node)
            return

        left_index, right_index, split_index, gain, grid_array, grid_info = self.split(node_x, node_y)

        if len(left_index) == 0 or len(right_index) == 0:
            node.is_leaf = True
            self.leaf_num += 1
            node.reason = "Failed cluster!"
            output_node_info(self.logger, node)

            # if split_index is None:
            #     split_index = 1
            #     try:
            #         save_path = f"{self.pic_path}/{node.node_id}_none.png"
            #         draw_data_dist(node_x[:, split_index, :], node_y, save_path=save_path)
            #     except np.linalg.LinAlgError:
            #         return
            # else:
            #     try:
            #         save_path = f"{self.pic_path}/{node.node_id}_single.png"
            #         draw_grid_info(node_x[:, split_index, :], node_y, grid_array, grid_info, save_path=save_path)
            #     except np.linalg.LinAlgError:
            #         return
            return

        left_y = node_y[left_index]
        right_y = node_y[right_index]
        left_impurity = calc_impurity(left_y, self.criterion)
        right_impurity = calc_impurity(right_y, self.criterion)
        self.logger.info("node id: {}".format(node.node_id, node.reason))
        self.logger.info(
            "depth: {:<3d}  sample_num: {:<4d}  class_dist: {:>4d}:{:<4d}  impurity: {:.4f}    "
            "child: {:>4d}:{:<4d}  child_impurity: {:.4f}:{:.4f}  gain: {:.4f}  feature_index: {:<4d}"
            .format(node.depth, node_x.shape[0], distribution[0][1], distribution[1][1], node.impurity,
                    len(left_index), len(right_index), left_impurity, right_impurity, gain, split_index))

        fi = gain * node.sample_num / self.sample_num
        self.feature_importance[split_index] += fi

        node.split_index = split_index
        node.gain = gain
        node.grid_array = grid_array
        node.grid_info = grid_info

        # Grow left subtree
        left_node = TreeNode(node_id=node.node_id + "l")
        node.left_node = left_node
        self.grow(node_x[left_index], left_y, left_node, depth + 1)

        # Grow right subtree
        right_node = TreeNode(node_id=node.node_id + "r")
        node.right_node = right_node
        self.grow(node_x[right_index], right_y, right_node, depth + 1)

    def is_leaf(self, node, node_y):
        if self.max_depth is not None:
            if node.depth == self.max_depth:
                node.reason = 'max_depth'
                return True
        if len(np.unique(node_y)) == 1:
            node.reason = 'one_class'
            return True
        if node.sample_num < self.min_samples_split:
            node.reason = 'min_sample_split'
            return True
        minor_class_num = min([len(np.where(node_y == class_label)[0]) for class_label in self.class_label])
        if minor_class_num < self.minor_class_num:
            node.reason = 'minor_class_num'
            return True
        if node.impurity < self.min_impurity:
            node.reason = 'min_impurity'
            return True
        return False

    def split(self, node_x, node_y):

        sample_num, feature_num, pair_num = node_x.shape

        if isinstance(self.max_features, str):
            if self.max_features == 'all':
                sub_feature_num = feature_num
            elif self.max_features == 'sqrt':
                sub_feature_num = int(np.sqrt(feature_num))
            elif self.max_features == 'log2':
                sub_feature_num = int(np.log2(feature_num))
            else:
                raise TypeError("Error max_features type!")
        elif isinstance(self.max_features, float):
            sub_feature_num = int(feature_num * self.max_features)
        else:
            raise TypeError("Error max_features type!")

        if self.feature_select == 'random':
            feature_index = np.random.choice(np.arange(0, feature_num, 1), sub_feature_num, replace=False)
        elif self.feature_select == 'sequence':
            feature_index = np.arange(0, feature_num, 1)
        else:
            raise TypeError("Error feature_select type!")

        best_left_index, best_right_index = [], []
        best_i, best_grid_array, best_grid_info = None, None, None
        best_gain = 0
        for i in feature_index:

            left_index, right_index, grid_array, grid_info = self.grid_cluster(node_x[:, i, :], node_y)

            if len(left_index) == 0 or len(right_index) == 0:
                continue

            if self.criterion == 'entropy':
                gain = calc_entropy_gain(node_y, node_y[left_index], node_y[right_index], self.class_label)
            elif self.criterion == 'gini':
                gain = 1.0 - calc_gini_index(node_y, node_y[left_index], node_y[right_index], self.class_label)
            else:
                raise TypeError("Error Criterion for Split")

            if gain > best_gain:
                best_left_index, best_right_index = left_index, right_index
                best_i, best_gain = i, gain
                best_grid_array, best_grid_info = grid_array, grid_info

        return best_left_index, best_right_index, best_i, best_gain, best_grid_array, best_grid_info

    def grid_cluster(self, node_x, node_y):

        try:
            x_a, sort_a, unique_a, tmp_a, density_a, divs_a, centers_a = adapt_split(node_x[:, 0])
            x_b, sort_b, unique_b, tmp_b, density_b, divs_b, centers_b = adapt_split(node_x[:, 1])
        except np.linalg.LinAlgError:
            return [], [], None, None

        min_v, max_v = np.min(node_x), np.max(node_x)
        lines_a = np.sort(np.hstack((divs_a, np.array([min_v - 0.5, max_v + 0.5]))))
        lines_b = np.sort(np.hstack((divs_b, np.array([min_v - 0.5, max_v + 0.5]))))
        grid_length = [len(divs_a) + 1, len(divs_b) + 1]
        grid_info = {
            'min_v': min_v,
            'max_v': max_v,
            'divs': [divs_a, divs_b],
            'lines': [lines_a, lines_b],
            'centers': [centers_a, centers_b],
            'grid_length': grid_length
        }

        # Initial grid array
        grid_array = np.array([[
            Grid(grid_id="%d-%d" % (i, j), class_labels=self.class_label)
            for j in range(grid_length[1])] for i in range(grid_length[0])], dtype=Grid)
        grid_sample_num = np.full(grid_array.shape, 0)
        g_i_s = np.searchsorted(divs_a, node_x[:, 0])
        g_j_s = np.searchsorted(divs_b, node_x[:, 1])
        for g_i in range(grid_length[0]):
            for g_j in range(grid_length[1]):
                sample_index = np.where((g_i_s == g_i) & (g_j_s == g_j))[0]
                grid_array[g_i, g_j].class_num = np.array([len(np.where(node_y[sample_index] == class_label)[0])
                                                           for class_label in self.class_label])
                grid_array[g_i, g_j].n_sample = len(sample_index)
                grid_array[g_i, g_j].sample_index = sample_index
                grid_sample_num[g_i, g_j] = len(sample_index)

                if grid_array[g_i, g_j].n_sample == 0:
                    grid_array[g_i, g_j].center = np.array([(lines_a[g_i] + lines_a[g_i + 1]) / 2.,
                                                            (lines_b[g_j] + lines_b[g_j + 1]) / 2.])
                    # grid_array[g_i, g_j].class_dist = np.repeat(1. / len(self.class_label), len(self.class_label))
                else:
                    grid_array[g_i, g_j].center = np.mean(node_x[sample_index], axis=0)
                    # grid_array[g_i, g_j].class_dist = grid_array[g_i, g_j].class_num / len(sample_index)

        cluster_labels = ['L', 'R']
        max_indices = np.unravel_index(np.argsort(grid_sample_num.flatten())[-len(cluster_labels):],
                                       grid_sample_num.shape)
        if len(max_indices[0]) < 2:
            return [], [], None, None

        # Generate two clusters
        grid_label = np.full(grid_array.shape, 'O')
        cluster_info, cluster_index, cluster_center = [], [], []
        for k in range(len(cluster_labels)):
            g_i, g_j = max_indices[0][k], max_indices[1][k]
            cluster_info_k = [[g_i, g_j]]
            grid_array[g_i, g_j].cluster_label = cluster_labels[k]
            grid_label[g_i, g_j] = cluster_labels[k]
            grid_search(g_i, g_j, grid_array, grid_label, cluster_info_k)

            cluster_info.append(cluster_info_k)
            cluster_index_k = []
            for g_i in range(grid_length[0]):
                for g_j in range(grid_length[1]):
                    if grid_label[g_i, g_j] == cluster_labels[k]:
                        grid_array[g_i, g_j].cluster_label = cluster_labels[k]
                        cluster_index_k.extend(grid_array[g_i, g_j].sample_index)
            cluster_index.append(cluster_index_k)
            cluster_center.append(np.mean(node_x[cluster_index_k], axis=0))

        # Assign unlabeled grids into clusters
        for g_i in range(grid_length[0]):
            for g_j in range(grid_length[1]):
                if grid_array[g_i, g_j].cluster_label not in cluster_labels:
                    left_dis = np.linalg.norm(grid_array[g_i, g_j].center - cluster_center[0])
                    right_dis = np.linalg.norm(grid_array[g_i, g_j].center - cluster_center[1])
                    if left_dis < right_dis:
                        grid_array[g_i, g_j].cluster_label = 'L'
                        grid_label[g_i, g_j] = 'L'
                    else:
                        grid_array[g_i, g_j].cluster_label = 'R'
                        grid_label[g_i, g_j] = 'R'

        # Assign samples to left or right clusters
        left_index, right_index = [], []
        for g_i in range(grid_length[0]):
            for g_j in range(grid_length[1]):
                if grid_label[g_i, g_j] == 'L':
                    left_index.extend(grid_array[g_i, g_j].sample_index)
                elif grid_label[g_i, g_j] == 'R':
                    right_index.extend(grid_array[g_i, g_j].sample_index)
                else:
                    raise TypeError("Unknown grid label!")
        left_index, right_index = sorted(left_index), sorted(right_index)

        return left_index, right_index, grid_array, grid_info

    def predict(self, x):
        proba = self.predict_proba(x)
        return np.argmax(proba, axis=1)

    def predict_proba(self, x):
        if len(x.shape) == 2:
            x = to_pairing_vec(x)
        proba = self.find_leaf(self.head_node, x)
        return proba

    def find_leaf(self, node, x):
        # Leaf
        if (node.left_node is None) and (node.right_node is None):
            return np.repeat(np.array([node.class_prob]), repeats=x.shape[0], axis=0)
        # Branch
        else:
            x_tmp = x[:, node.split_index, :]

            grid_array = node.grid_array
            grid_info = node.grid_info

            g_i_s = np.searchsorted(grid_info['divs'][0], x_tmp[:, 0])
            g_j_s = np.searchsorted(grid_info['divs'][1], x_tmp[:, 1])
            sample_cluster_s = np.array([grid.cluster_label for grid in grid_array[g_i_s, g_j_s]])

            left_index = np.where(sample_cluster_s == 'L')[0]
            right_index = np.where(sample_cluster_s == 'R')[0]

            if len(left_index) == 0:
                return self.find_leaf(node.right_node, x[right_index])
            if len(right_index) == 0:
                return self.find_leaf(node.left_node, x[left_index])

            left_proba = self.find_leaf(node.left_node, x[left_index])
            right_proba = self.find_leaf(node.right_node, x[right_index])

            index = np.hstack((left_index, right_index))
            proba = np.vstack((left_proba, right_proba))

            sort_index = np.argsort(index)
            return proba[sort_index]


