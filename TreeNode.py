
class TreeNode:
    def __init__(self, node_id=None):

        self.node_id = node_id

        self.depth = None
        self.class_dist = None
        self.class_prob = None
        self.sample_num = None
        self.impurity = None

        self.is_leaf = False

        self.split_index = None
        self.gain = None
        self.grid_array = None
        self.grid_info = None

        self.left_node = None
        self.right_node = None

        self.reason = None


        


