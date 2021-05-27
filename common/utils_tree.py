import numpy as np
import math


class Node:
    def __init__(self, axis=None):
        self.axis = axis
        self.leaf = False
        self.c = Node
        self.left = None
        self.right = None

    def copy(self):
        n = Node()
        n.axis = self.axis
        n.leaf = self.leaf
        n.c = self.c
        if self.left is not None:
            n.left = self.left.copy()
        if self.right is not None:
            n.right = self.right.copy()
        return n


def GenerateTree(data, label, feature_selected, thresh):
    """
    返回一颗通过data和label所生成的决策树
    :param data:
    :param label:
    :param thresh:
    :return:
    """
    try:
        data_left, label_left, data_right, label_right, \
        new_feature_selected, axis = SplitNode(data, label, feature_selected, thresh)
    except TypeError:
        # 如果出现了TypeError, 表明分支已经无法继续进行下去，因此就是叶子节点
        n = Node(axis=None)
        n.leaf = True
        n.c = max_class(label)
        return n
    n = Node(axis=axis)
    n.c = max_class(label)
    n.left = GenerateTree(data_left, label_left, new_feature_selected, thresh)
    n.right = GenerateTree(data_right, label_right, new_feature_selected, thresh)
    return n


def SplitNode(data, label, feature_selected, thresh):
    """
    用于从当前数据中进行特征分割
    :param data:
    :param label:
    :param feature_selected:
    :param thresh:
    :return:
    """
    # 如果是单一类,或者所有特征都已经用完(此时axix_chosen是None),停止分裂,直接返回None
    axis, max_gr = SelectFeature(data, label, feature_selected)
    # 或者此时的信息增益率已经不能够满足阈值要求
    if max_gr < thresh or axis is None:
        return None
    # 左子树元素
    data_left = data[data[:, axis] == 0, :]
    label_left = label[data[:, axis] == 0, :]
    # 右子树元素
    data_right = data[data[:, axis] == 1, :]
    label_right = label[data[:, axis] == 1, :]
    new_feature_selected = feature_selected.copy()
    new_feature_selected.add(axis)
    # print("axis %d chose, gr is %.3f" % (axis, max_gr))
    return data_left, label_left, data_right, label_right, new_feature_selected, axis


def SelectFeature(data, label, feature_selected):
    """
    计算所有特征中信息增益率最大的特征,返回该特征对应的维度和信息增益率
    :param data:
    :param label:
    :param feature_selected: set类型,表示当前已经选择过的分支
    :return:
    """
    n, m = data.shape
    max_gr = 0
    axis_chosen = None
    entropy = cross_entropy(label, 0)
    if entropy == 0:
        return None, 0
    for axis in range(m):
        if axis not in feature_selected:
            gr = gain_ratio(data, label, axis, entropy_before=entropy)
            if gr > max_gr:
                max_gr = gr
                axis_chosen = axis
    return axis_chosen, max_gr


def Decision(Tree, X_for_predict):
    node = Tree
    while not node.leaf:
        if X_for_predict[node.axis] == 0:
            node = node.left
        else:
            node = node.right
    return node.c


def Impurity(data, axis):
    # 计算某一类的熵,如标签或者某一个特征
    n = data.shape[0]
    values = list(set(data[:, axis]))
    n_class = len(values)
    num_each_class = [0] * n_class
    for i, value in enumerate(values):
        num_each_class[i] = np.sum(data[:, axis] == value)
    entropy = 0.0
    for num in num_each_class:
        if num != 0:
            entropy -= num / n * math.log2(num / n)
    return entropy


def Prune(Tree, data, label, prune_thresh=0):
    """
    利用验证集数据进行剪枝，通过反复调用prune_leaf,直至不再有可以剪掉的叶子节点为止
    :param Tree:
    :param data:
    :param label:
    :param aug_thresh: 用于防止过度剪枝，只有当剪枝后准确率提升大于aug_thresh时才进行剪枝
    :return:
    """
    while True:
        res = prune_leaf(Tree, data, label, prune_thresh)
        if not res:
            break
    return Tree


def prune_leaf(node, data, label, prune_thresh=0):
    """
    利用data和label对Tree进行一次剪枝,逐一将每个叶子节点
    :param Tree:
    :param data:
    :param test:
    :return:
    """
    # 通过将node.leaf设置为True即可名义上进行节点删除
    acc_before_prune = evaluate_accuracy(node, data, label)
    for n in get_leaf_parent(node):
        n.leaf = True
        acc_after_prune = evaluate_accuracy(node, data, label)
        if acc_after_prune - acc_before_prune >= prune_thresh:
            return True
        else:
            n.leaf = False
    return False


def evaluate_accuracy(Tree, data, label):
    n = label.shape[0]
    acc_num = 0
    for i in range(n):
        if Decision(Tree, data[i, :]) == label[i]:
            acc_num += 1
    return acc_num / n


def cross_entropy(data, axis):
    # 计算某一类的熵,如标签或者某一个特征
    n = data.shape[0]
    values = list(set(data[:, axis]))
    n_class = len(values)
    num_each_class = [0] * n_class
    for i, value in enumerate(values):
        num_each_class[i] = np.sum(data[:, axis] == value)
    entropy = 0.0
    for num in num_each_class:
        if num != 0:
            entropy -= num / n * math.log2(num / n)
    return entropy


def cross_entropy_given_x(data, label, axis):
    """
    计算某一特征的条件熵
    :param data:
    :param label:
    :param axis:
    :return:
    """
    n = data.shape[0]
    data_axis = data[:, axis].reshape(-1, 1)

    label_x_0 = label[data[:, axis] == 0, :]
    label_x_1 = label[data[:, axis] == 1, :]
    p0 = label_x_0.shape[0] / n
    p1 = label_x_1.shape[0] / n
    # 0类：
    entropy_0 = cross_entropy(label_x_0, 0)
    # 1类：
    entropy_1 = cross_entropy(label_x_1, 0)
    return p0 * entropy_0 + p1 * entropy_1


def gain_ratio(data, label, axis, entropy_before=None):
    """
    计算信息增益率
    :param data:
    :param label:
    :param axis:
    :param entropy_before:
    :return:
    """
    if entropy_before is None:
        entropy_before = cross_entropy(label, 0)
    entropy_after = cross_entropy_given_x(data, label, axis)
    gain = entropy_before - entropy_after
    cross_entropy_axis = cross_entropy(data, axis)
    # 如果特征都一样，那么表明这一特征完全没有提供任何信息，返回0即可
    if cross_entropy_axis == 0:
        return 0
    return gain / cross_entropy_axis


def max_class(label):
    max_num, c = 0, 0
    for i in range(9):
        s = np.sum(label == i)
        if s > max_num:
            c, max_num = i, s
    return c


def walk_tree(node):
    if node is None: return
    print("axis = ", node.axis, "leaf: ", node.leaf, " class :", node.c)
    if node.leaf: return
    walk_tree(node.left)
    walk_tree(node.right)


def get_leaf_parent(node):
    if not node.leaf:
        if node.left.leaf and node.right.leaf:
            yield node
        if not node.left.leaf and not node.right.leaf:
            for n in get_leaf_parent(node.left):
                yield n
            for n in get_leaf_parent(node.right):
                yield n