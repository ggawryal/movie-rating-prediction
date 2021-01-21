import numpy as np
import math
from sklearn.metrics import mean_squared_error, r2_score
import data_preparation

np.random.seed(4325534)
data_train_org, data_test_org = data_preparation.get_data(0.7,save_to_file=False, test_set_fraction=0.3)

class Scaler:
    def fit_transform(self,Xy_train):
        self.mean, self.stddev = [],[]
        for column in Xy_train.T:
            self.mean.append(sum(column)/len(column))
            self.stddev.append((sum(column**2)/len(column) - self.mean[-1]**2))
            if self.stddev[-1] >= 0:
                self.stddev[-1] = np.sqrt(self.stddev[-1])
            if self.stddev[-1] < 1e-6 or np.isnan(self.stddev[-1]):
                self.stddev[-1] = 1
        return self.transform(Xy_train)
    
    def transform(self, Xy):
        return np.array([(row - self.mean)/self.stddev for row in Xy])

    def inverse_transform_column(self, col, i):
        return col*self.stddev[i]+self.mean[i] 


#Kernel version of ridge regression
class LinearRegression:
    def __init__(self, C=1.0, kernel="linear", gaussian_gamma=1, poly_degree=3, poly_const=1):
        #kernel should be either linear (default), gaussian or polynomial
        self.C = C
        self.make_kernel_matrix = lambda A,B: A @ B.T

        if kernel == "gaussian":
            self.make_kernel_matrix = lambda A,B: np.exp(-gaussian_gamma*(
                -2*A@B.T + np.tile((A**2).sum(axis=1).reshape(-1,1), (1,B.shape[0])) + np.tile((B**2).sum(axis=1), (A.shape[0],1))
            ))

        elif kernel == "polynomial" or kernel == "poly":
            self.make_kernel_matrix = lambda A,B: (A @ B.T + poly_const)**poly_degree

    def fit(self,X,y):
        n,m = X.shape[1], X.shape[0] 
        K = self.make_kernel_matrix(X,X)

        self.w = np.linalg.solve(K + self.C*np.identity(m), y)
        self.X = X
    
    def predict(self,x):
        return self.make_kernel_matrix(x,self.X) @ self.w

class Decision_Tree_Regression:
    class TreeNode:
        def set_params(self, j, s, left, right):
            self.j = j
            self.s = s
            self.left = left
            self.right = right
    
    def __init__(self, max_depth=40,leaf_prediction_function = "mean"):
        self.max_depth = max_depth
        if leaf_prediction_function == "mean":
            self.leaf_prediction_function = lambda Xy: np.mean(Xy[:,-1])
        elif leaf_prediction_function == "median":
            self.leaf_prediction_function = lambda Xy: np.median(Xy[:,-1])
        elif callable(leaf_prediction_function):
            self.leaf_prediction_function = leaf_prediction_function
        else:
            raise RuntimeError("unsupported leaf prediction function")

    def find_split_point(self,Xy,ignored_last_features_n):
        n,m = Xy.shape[1]-1, Xy.shape[0]
        best = (1e18,None,None)
        for j in range(n-ignored_last_features_n):
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
    
    def fit_recursive(self, Xy, node, depth, ignored_last_features_n):
        leaf = False
        if len(Xy) <= 5 or depth == self.max_depth:
            leaf = True
        else:
            j,s = self.find_split_point(Xy,ignored_last_features_n)
            if j is None:
                leaf = True
            else:
                XyL, XyR = Xy[Xy[:,j] <= s, :], Xy[Xy[:,j] > s, :]
                node.set_params(j,s,  Decision_Tree_Regression.TreeNode(),  Decision_Tree_Regression.TreeNode())
                self.fit_recursive(XyL,node.left,depth+1,ignored_last_features_n)
                self.fit_recursive(XyR,node.right,depth+1,ignored_last_features_n)
        if leaf:
            assert(len(Xy) > 0)
            node.set_params(None,None,None,None)
            #node.c = np.mean(Xy[:,-1])
            node.c = self.leaf_prediction_function(Xy)
            return 

    def fit(self,X,y, ignored_last_features_n = 0):
        self.root = Decision_Tree_Regression.TreeNode()
        self.fit_recursive(np.append(X,y.reshape(-1,1),axis=1), self.root, 0,ignored_last_features_n)

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
    def __init__(self, iters=100, max_tree_depth=4, eta=0.1, sample_fraction=1.0, loss = "L2", huber_alpha_quantile=0.6):
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
        loss
            loss function for gradient boosting, could be either L2 (squared error, LS_Boost), L1 (abs error, LAD_TreeBost) or huber(M_TreeBost)
        huber_alpha_quantile
            quantile of how many values are considered as outliers in huber loss function, in each iteration huber delta parameter is calculated based on this
        """
        self.iters = iters
        self.max_tree_depth = max_tree_depth
        self.eta = eta
        self.sample_fraction = sample_fraction
        self.huber_alpha_quantile = huber_alpha_quantile
        self.loss = loss
        if loss == "L2":
            self.gradient = lambda x,y,: x-y
        elif loss == "L1":
            self.gradient = lambda x,y: np.sign(x-y)
        elif loss == "huber":
            self.gradient = lambda x,y,delta: x-y if np.abs(x-y) <= delta else delta*np.sign(x-y)
        else:
            raise("unsupported loss function")

    def update_function(self,f,tree):
        return lambda x: f(x) + self.eta*tree.predict_single(x)

    def fit(self,X,y):
        n,m = X.shape[1], X.shape[0]
        starting_value_function = np.mean if self.loss == "lse" else np.median

        fs = [lambda _ : starting_value_function(y)] + [None] * self.iters

        for t in range(self.iters):
            random_indexes = np.random.choice(X.shape[0], max(1,int(m*self.sample_fraction)), replace=False)
            X2,y2 = X[random_indexes], y[random_indexes]

            if self.loss == "huber":
                r = y2 - np.apply_along_axis(fs[t],1,X2)
                delta = np.quantile(r,self.huber_alpha_quantile)
                g = np.array([self.gradient(y2[i], fs[t](X2[i]),delta) for i in range(len(y2))])
                X2 = np.append(X2, r.reshape(-1,1),axis=1)

                def leaf_prediction_function(Xy2):
                    r_tilde = np.median(Xy2[:,-2])
                    return r_tilde + np.mean(np.sign(Xy2[:,-2]-r_tilde)*np.minimum(np.abs(Xy2[:,-2] - r_tilde),delta))    

                tree = Decision_Tree_Regression(self.max_tree_depth,leaf_prediction_function)
                tree.fit(X2,g, ignored_last_features_n=1)

            else:
                g = np.array([self.gradient(y2[i], fs[t](X2[i])) for i in range(len(y2))])
                tree = Decision_Tree_Regression(self.max_tree_depth, "mean" if self.loss == "L2" else "median")
                tree.fit(X2,g)

            fs[t+1] = self.update_function(fs[t],tree)
            #y_pred = np.array([fs[t+1](x) for x in X])
            #print("t =",t,"mse =",mean_squared_error(y_pred,y))
        self.h = fs[-1]

    def predict_single(self,x):
        return self.h(x)

    def predict(self,X):
        return np.array([self.predict_single(x) for x in X])

class Random_Forest_Regression:
    def __init__(self, n_trees, max_depth, bootstrap=True, max_features="all", alpha_prunning=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.alpha_prunning = alpha_prunning

        if isinstance(max_features,int):
            self.n_features = lambda _ : max_features
        elif max_features == "sqrt":
            self.n_features = math.sqrt
        else:
            max_features = lambda x : x

        self.trees = []
    
    def fit(self,X,y):
        n,m = X.shape[1], X.shape[0]
        for i in range(self.n_trees):
            random_columns = np.random.choice(X.shape[1],np.clip(int(self.n_features(X.shape[1])), X.shape[1],X.shape[1]), replace=False)
            random_rows    = np.random.choice(X.shape[0], X.shape[0], replace=self.bootstrap) #if bootstrap is False, then rows will be only permuted
            Xs = np.array([X.T[i] if i in random_columns else np.zeros(m) for i in range(n)]).T
            Xs,ys = Xs[random_rows], y[random_rows]
            tree = Decision_Tree_Regression(max_depth=self.max_depth)
            tree.fit(Xs,ys)
            self.trees.append(tree)
    
    def predict_single(self,x):
        y = 0
        for tree in self.trees:
            y += tree.predict_single(x)
        return y/self.n_trees
    
    def predict(self,X):
        return np.array([self.predict_single(x) for x in X])


def outliers_fraction(y_pred, y, c):
    return (np.abs(y-y_pred) > c).sum()/len(y_pred)



for model,name in (
    (Gradient_Boosting_Regression(iters=30, max_tree_depth=4, sample_fraction=0.6, loss="L2"),"GB L2"),
    (Gradient_Boosting_Regression(iters=100, max_tree_depth=5, sample_fraction=0.6,eta=0.05, loss="huber",huber_alpha_quantile=0.8),"GB huber"),
    (LinearRegression(C=1),"LR"),
    (LinearRegression(C=15000*10,kernel="polynomial",poly_degree=3,poly_const=1),"LR poly kernel"),
    (LinearRegression(C=1,kernel="gaussian",gaussian_gamma=0.009), "LR gaussian kernel"),
    (Decision_Tree_Regression(max_depth=6),"Decision tree"),
    (Random_Forest_Regression(n_trees = 50,max_depth=4,bootstrap=True,max_features="sqrt"),"Random Forest")):


    data_train, data_test = np.copy(data_train_org), np.copy(data_test_org)
    
    np.random.shuffle(data_test)
    data_val,data_test = data_test[:len(data_test)//2], data_test[len(data_test)//2:]

    scaler = Scaler()

    data_train = scaler.fit_transform(data_train)
    data_val  = scaler.transform(np.copy(data_val))
    data_test  = scaler.transform(np.copy(data_test))
    d_X_train = data_train[:,:-1]
    d_y_train = data_train[:,-1]

    d_X_test = data_test[:,:-1]
    d_y_test = data_test[:,-1]

    d_X_val = data_val[:,:-1]
    d_y_val = data_val[:,-1]

            
    model.fit(d_X_train, d_y_train)
    d_y_pred = model.predict(d_X_val)
    d_y_pred_tr = model.predict(d_X_train)

    train_set_fraction=1
    print(train_set_fraction,name,'MSE on train: %.2f' % mean_squared_error(d_y_train, d_y_pred_tr))
    print(train_set_fraction,name,'MSE on test: %.2f' % mean_squared_error(d_y_val, d_y_pred))
    print(train_set_fraction,name,'Outliers fraction 1: %.2f' % outliers_fraction(d_y_val, d_y_pred,0.5))
    print(train_set_fraction,name,'R2: %.2f'     % r2_score(d_y_val, d_y_pred))
