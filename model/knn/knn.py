import numpy as np
import model.heap as h
import numpy.linalg as la

def distance(a, b):
    return la.norm(a - b)

class kd_note:
    def __init__(self, vector=None, label=None, split=None, left=None, right=None, parent=None):
        self.vector = vector
        self.label = label
        self.split = split
        self.left = left
        self.right = right
        self.parent = parent
        self.visited = False

class kd_tree:
    def __init__(self, data, label):
        if data.ndim != 2:
            print("Invalid data!")
            return

        self.dim = data.shape[1]
        self.tree = self._build_kd_tree(0, data, label)

    def _build_kd_tree(self, split, data, label):
        if data.shape[0] == 0:
            return None
        n = data.shape[0] // 2
        indices = data[:, split].argsort()
        data = data[indices]
        label = label[indices]
        #data = sorted(data, key = lambda item : item[split])
        #data = np.array(data)
        while n > 0 and data[n, split] == data[n - 1, split]:
            n -= 1
        left = self._build_kd_tree((split + 1) % self.dim, data[: n], label[: n])
        right = self._build_kd_tree((split + 1) % self.dim, data[n + 1:], label[n + 1:])
        node = kd_note(data[n], label[n], split, left, right)
        if left is not None:
            left.parent = node
        if right is not None:
            right.parent = node
        return node

    def k_neighbor(self, point, k=5):
        if k < 1:
            return None
        node = self.search(point, self.tree)
        heap = h.MaxHeap()

        while node is not None:
            if node.visited:
                node = node.parent
                continue

            node.visited = True
            if heap.length() < k:
                #if node.visited == False:
                tmp = (distance(point, node.vector), list(node.vector), node.label)
                heap.push(tmp)
                #node.visited = True
                #print(heap.length())

                if node.left is not None and node.left.visited == False:
                    node = self.search(point, node.left)
                elif node.right is not None and node.right.visited == False:
                    node = self.search(point, node.right)
                else:
                    node = node.parent
            else:
                if distance(point, node.vector) < heap.max():
                    tmp = (distance(point, node.vector), list(node.vector), node.label)
                    heap.push(tmp)
                    heap.pop()
                    #node.visited = True

                #print(heap.max())
                #print(abs(point[node.split] - node.vector[node.split]))
                condition = heap.max() > abs(point[node.split] - node.vector[node.split])
                if node.left is not None and node.left.visited == False and condition:
                    node = self.search(point, node.left)
                elif node.right is not None and node.right.visited == False and condition:
                    #print("aaa")
                    node = self.search(point, node.right)
                else:
                    node = node.parent

        #print(heap.to_list())
        self.clear_visited(self.tree)
        return heap.to_list()

    def search(self, point, node):
        #if node.right is None and node.left is None:
        #    return node
        split = node.split

        if point[split] < node.vector[split]:
            if node.left is None:
                return node
            else:
                return self.search(point, node.left)
        else:
            if node.right is None:
                return node
            else:
                return self.search(point, node.right)

    def clear_visited(self, root):
        if root is None:
            return
        root.visited = False
        self.clear_visited(root.left)
        self.clear_visited(root.right)

    def pre_order(self, root):
        if root is None:
            return
        print(root.vector)
        self.pre_order(root.left)
        self.pre_order(root.right)

    def in_order(self, root):
        if root is None:
            return
        self.in_order(root.left)
        print(root.vector)
        self.in_order(root.right)

class knn:
    def __init__(self):
        self.x = None
        self.y = None
        self.use_kd_tree = None

    def _votes(self, labels):
        votes = dict()
        for label in labels:
            votes[label] = votes.get(label, 0) + 1
        sorted_votes = sorted(votes.items(), key=lambda item : item[1], reverse=True)
        return sorted_votes[0][0]

    def fit(self, x, y, use_kd_tree = False):
        self.x = x
        self.y = y
        self.kd_tree = None
        self.use_kd_tree = use_kd_tree
        if self.use_kd_tree:
            self.kd_tree = kd_tree(x, y)
            #self.kd_tree.pre_order(self.kd_tree.tree)
            #self.kd_tree.in_order(self.kd_tree.tree)

    def predict(self, x, k=5, detailed=False):
        if self.use_kd_tree:
            if x.ndim == 1:
                k_neighbor = self.kd_tree.k_neighbor(x, k=k)
                #print(k_neighbor)
                [_, vectors, labels] = zip(*k_neighbor)
                #print(vectors, type(labels))

                if detailed:
                    print("For {}, the {} nearest neighbor are:".format(x, k))
                    for i in range(len(labels)):
                        print("x:{}, y:{}".format(vectors[i], labels[i]))

                votes = dict()
                for label in labels:
                    votes[label] = votes.get(label, 0) + 1
                sorted_votes = sorted(votes.items(), key=lambda item: item[1], reverse=True)
                return sorted_votes[0][0]
                #return self._votes(labels)
            else:
                return np.array([self.predict(i, k=k, detailed=detailed) for i in x])
        else:
            if x.ndim == 1:
                heap = h.MaxHeap()
                for i in range(self.x.shape[0]):
                    #tmp = (-la.norm(self.x[i, :] - x, ord = distance), i)
                    tmp = (distance(self.x[i, :], x), i)
                    #hp.heappush(heap, tmp)
                    heap.push(tmp)
                    if heap.length() > k:
                        heap.pop()
                [_, indices] = zip(*heap.to_list())
                if detailed:
                    print("For {}, the {} nearest neighbor are:".format(x, k))
                    for index in indices:
                        print("x:{}, y:{}".format(self.x[index], self.y[index]))

                votes = dict()
                for i in indices:
                    label = self.y[i]
                    votes[label] = votes.get(label, 0) + 1
                sorted_votes = sorted(votes.items(), key=lambda item: item[1], reverse=True)
                return sorted_votes[0][0]
            else:
                return np.array([self.predict(i, k=k, detailed=detailed) for i in x])
