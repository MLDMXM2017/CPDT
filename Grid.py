
class Grid:
    def __init__(self, grid_id=None, class_labels=None):

        self.grid_id = grid_id

        self.class_num = {class_label: 0 for class_label in class_labels}
        self.n_sample = 0
        self.sample_index = []

        self.center = None
        self.class_dist = None

        self.cluster_label = 'O'

        self.class_prob = None




