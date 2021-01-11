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
        XyL, XyR = Xy[Xy[:,j] <= s, :], Xy[Xy[:,j] > s, :]
        #print("split by x[",j,"] <=",s,len(XyL),len(XyR))
        node.set_params(j,s,  Decision_Tree_Regression.TreeNode(),  Decision_Tree_Regression.TreeNode())
        self.fit_recursive(XyL,node.left,depth+1)
        self.fit_recursive(XyR,node.right,depth+1)

    def fit(self,X,y):
        self.root = Decision_Tree_Regression.TreeNode()
        self.fit_recursive(np.append(X,y.reshape(-1,1),axis=1), self.root, 0)

    def predict_single(self,x):
        if(x.ndim != 1):
            raise RuntimeError("predict single should take single observation")
        node = self.root
        while node.j is not None:
            if x[node.j] <= node.s:
                node = node.left
            else:
                node = node.right
        return node.c

    def predict(self,X):
        return np.array([self.predict_single(x) for x in X])

class Gradient_Boosting_Regression:
    def __init__(self, iters=100, max_tree_depth=4, eta=0.1, sample_fraction=1.0):
        """
        Parameters
        ----------
        iters 
            number of iterations in boosting
        max_tree_depth 
            depth of each of decision trees
        eta gradient 
            learing rate
        sample_fraction 
            fraction of random sample of rows used in each iteration, 
            1.0 means using whole set of rows, smaller values can be used for stochastic gradient boosting
        """
        self.iters = iters
        self.max_tree_depth = max_tree_depth
        self.eta = eta
        self.sample_fraction = sample_fraction

    def update_function(self,f,tree):
        return lambda x: f(x) + self.eta*tree.predict_single(x)

    def fit(self,X,y):
        n,m = X.shape[1], X.shape[0]
        
        mean_y = sum(y)/len(y)
        fs = [lambda _ : mean_y] + [None] * self.iters

        for t in range(self.iters):
            random_indexes = np.random.choice(X.shape[0], max(1,int(m*self.sample_fraction)), replace=False)
            X2,y2 = X[random_indexes], y[random_indexes]
            g = np.array([y2[i] - fs[t](X2[i]) for i in range(len(y2))])

            tree = Decision_Tree_Regression(self.max_tree_depth)
            tree.fit(X2,g)
            fs[t+1] = self.update_function(fs[t],tree)
            
            #y_pred = np.array([fs[t+1](x) for x in X])
            #print("t =",t,"mse =",mean_squared_error(y_pred,y))

        self.h = fs[-1]

    def predict_single(self,x):
        return self.h(x)

    def predict(self,X):
        return np.array([self.predict_single(x) for x in X])

#regr = Decision_Tree_Regression(max_depth=6)
regr = Gradient_Boosting_Regression(iters=30, max_tree_depth=4,sample_fraction=0.6)
regr.fit(d_X_train, d_y_train)

d_y_pred = regr.predict(d_X_test)
d_y_pred_tr = regr.predict(d_X_train)

print('MSE on train: %.2f' % mean_squared_error(d_y_train, d_y_pred_tr))
print('MSE on test: %.2f' % mean_squared_error(d_y_test, d_y_pred))
print('R2: %.2f'     % r2_score(d_y_test, d_y_pred))


