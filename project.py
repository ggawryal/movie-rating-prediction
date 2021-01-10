import numpy as np
import math
from functools import partial
from sklearn.metrics import mean_squared_error, r2_score
import sklearn.preprocessing 


import data_preparation

#data = data_preparation.get_data(True)
data = data_preparation.get_data_from_tmp()

data_train, data_test = data[:-5000], data[-5000:]

scaler = sklearn.preprocessing.StandardScaler()
data_train = scaler.fit_transform(data_train)
data_test  = scaler.transform(data_test)


d_X_train = data_train[:,:-1]
d_y_train = data_train[:,-1]

d_X_test = data_test[:,:-1]
d_y_test = data_test[:,-1]

class Decision_Tree_Regression:
    class TreeNode:
        def set_params(self, j, s, left, right):
            self.j = j
            self.s = s
            self.left = left
            self.right = right
    
    def __init__(self, max_depth=40):
        self.max_depth = max_depth

    def find_split_point(self,Xy):
        n,m = Xy.shape[1]-1, Xy.shape[0]
        best = (1e18,0,0)
        for j in range(n):
            Xs = Xy[np.argsort(Xy[:,j])]
            
            sumL, sumR = 0, Xy[:,-1].sum()
            sumL2, sumR2 = 0,(Xy[:,-1]**2).sum()
            for i in range(1,m):
                sumL += Xs[i-1][-1]
                sumR -= Xs[i-1][-1]
                sumL2 += Xs[i-1][-1]**2
                sumR2 -= Xs[i-1][-1]**2
                if Xs[i-1][j] == Xs[i][j]:
                    continue
                s = (Xs[i-1][j] + Xs[i][j])/2

                c1,c2 = sumL/i, sumR/(m-i)
                J = sumL2 - 2*c1*sumL + i*c1*c1 + sumR2 - 2*c2*sumR + (m-i)*c2*c2
                best = min(best,(J,j,s))

        return (best[1], best[2])
    
    def fit_recursive(self, Xy, node, depth):
        if len(Xy) == 1 or depth == self.max_depth:
            node.set_params(None,None,None,None)
            node.c = Xy[:,-1].sum()/len(Xy)
            return 

        j,s = self.find_split_point(Xy)
        #print("split by x[",j,"] <=",s)
        XyL, XyR = Xy[Xy[:,j] <= s, :], Xy[Xy[:,j] > s, :]
        node.set_params(j,s,  Decision_Tree_Regression.TreeNode(),  Decision_Tree_Regression.TreeNode())
        self.fit_recursive(XyL,node.left,depth+1)
        self.fit_recursive(XyR,node.right,depth+1)

    def fit(self,X,y):
        self.root = Decision_Tree_Regression.TreeNode()
        self.fit_recursive(np.append(X,y.reshape(-1,1),axis=1), self.root, 0)

    def predict_single(self,x):
        node = self.root
        while node.j is not None:
            if x[node.j] <= node.s:
                node = node.left
            else:
                node = node.right
        return node.c

    def predict(self,X):
        return np.array([self.predict_single(x) for x in X])

regr = Decision_Tree_Regression(max_depth=4)

regr.fit(d_X_train, d_y_train)

d_y_pred = regr.predict(d_X_test)
d_y_pred_tr = regr.predict(d_X_train)

print('MSE on train: %.2f' % mean_squared_error(d_y_train, d_y_pred_tr))
print('MSE on test: %.2f' % mean_squared_error(d_y_test, d_y_pred))
print('R2: %.2f'     % r2_score(d_y_test, d_y_pred))


