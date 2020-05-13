import numpy as np
import pandas as pd
import sys

# Hacky solution to circumvent recursion limit :(
sys.setrecursionlimit(10 ** 6)

# Calculate the Gini impurity of the given count
def impurity(n, counts):
    gini = 0
    for i in counts.keys():
        p = counts[i] / n
        gini += p * (1 - p)
    return gini

def count(y, rows):
    counts = dict()
    for i in y[rows]:
        if i in counts.keys():
            counts[i] += 1
        else:
            counts[i] = 1
    return counts

def split_feature(X, y, d, rows):
    n = len(rows)
    H = 999
    ind = -1
    thr = None

    # sort data by the column
    X_sorted = np.argsort(X[:,d])
    data = X[X_sorted]
    
    # initialize the counts for this feature
    counts_a = dict()
    counts_b = count(y, rows)

    i = 0
    while i < n-1:
        v = y[data[i,-1]]

        counts_b[v] -= 1
        if v in counts_a.keys():
            counts_a[v] += 1
        else:
            counts_a[v] = 1

        # Prevent branching between points with same values
        while i < n-2 and data[i+1, d] == data[i, d]:
            v = y[data[i,-1]]
            counts_b[v] -= 1
            if v in counts_a.keys():
                counts_a[v] += 1
            else:
                counts_a[v] = 1
            i += 1

        H_a = impurity(i + 1, counts_a) * (i + 1) / n
        H_b = impurity(n- i - 1, counts_b) * (n - i - 1) / n
        
        if H_a + H_b < H:
            H = H_a + H_b
            ind = i
            thr = data[i,d]

        i += 1

    return H, data[:ind+1,-1], data[ind+1:,-1], thr

# Return the best split given a dataset, features and rows
def split(dataset, y, features, rows):
    # append the row number as a column
    X = np.hstack((dataset[rows], rows.reshape(len(rows), 1)))
    H = 999
    thr = -1
    d   = -1
    S_a = None
    S_b = None

    for d_ in features:
        H_, S_a_, S_b_, thr_ = split_feature(X, y, d_, rows)
        if H_ < H:
            d   = d_
            S_a = S_a_
            S_b = S_b_
            thr = thr_

    return S_a, S_b, d, thr 

# Node in a decision tree. Rows is a list containing the indices
# corresponding to this node, and k is a list of features.
# Since the tree is part of a random forest, we will make it maximally
# deep.
class TreeNode:
    def __init__(self, dataset, rows, features, y, f=-1):
        self.leaf = False
        self.label = None
        self.left = None
        self.right = None

        rows = rows.astype(int)
        n = len(rows)

        # if this subset is pure, split no further
        if impurity(n, count(y, rows)) == 0:
            self.leaf = True
            self.label = y[rows[0]]
            return

        # otherwise, make an optimal (greedy) split
        if f != None:
            features_ = [x for x in features if x != f]
        else:
            features_ = features
        S_a, S_b, d, thr = split(dataset, y, features_, rows)

        self.feature = d
        self.threshold = thr
        self.left = TreeNode(dataset, S_a, features, y, f=d)
        self.right = TreeNode(dataset, S_b, features, y, f=d)

    # classify the given point according to this tree
    def classify(self, x):
        if self.leaf == True:
            return self.label
        else:
            if x[self.feature] <= self.threshold:
                return self.left.classify(x)
            else:
                return self.right.classify(x)

# A random forest class applies bagging (bootstrap aggregating) to the
# decision tree class (TreeNode) defined above
class RandomForest:
    # m is the number of trees
    def __init__(self, X, y, n, m, p=0.5):
        self.p     = p
        self.trees = []
        
        # d - number of features, k - number of features in each tree
        d = len(X[0])
        k = int(np.sqrt(d))

        df = pd.DataFrame(X)
        df['Survived'] = y
        for i in range(m):
            # Sampling a dataset and choosing a set of features
            sample = df.sample(n, replace=True).to_numpy()
            Xi = sample[:,:-1]
            yi = sample[:,-1]
            features = np.random.choice(np.arange(d), k, replace=False)
            tree = TreeNode(Xi, np.arange(n), features, yi)

            self.trees.append(tree)
            print('Tree', i, 'built.')
        
    def classify(self, x):
        votes = []
        for tree in self.trees:
            votes.append(tree.classify(x))
        return int(np.mean(np.array(votes)) > self.p)