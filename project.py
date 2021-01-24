import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys
import config
import data_preparation
from datetime import datetime

#data_train_org, data_test_org = data_preparation.get_data(0.1,save_to_file=False, test_set_fraction=0.3)

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
    def _kernel_matrix_creator(self):
        self.make_kernel_matrix = lambda A,B: A @ B.T
        if self.kernel_name == "gaussian":
            self.make_kernel_matrix = lambda A,B: np.exp(-self.gaussian_gamma*(
                -2*A@B.T + np.tile((A**2).sum(axis=1).reshape(-1,1), (1,B.shape[0])) + np.tile((B**2).sum(axis=1), (A.shape[0],1))
            ))

        elif self.kernel_name == "polynomial" or self.kernel_name == "poly":
            self.make_kernel_matrix = lambda A,B: (A @ B.T + self.poly_const)**self.poly_degree

    def __init__(self, C=1.0, kernel="linear", gaussian_gamma=1, poly_degree=3, poly_const=1):
        #kernel should be either linear (default), gaussian or polynomial
        self.C = C
        self.kernel_name = kernel
        self.gaussian_gamma = gaussian_gamma
        self.poly_degree = poly_degree
        self.poly_const = poly_const
        self._kernel_matrix_creator()

      
    def fit(self,X,y):
        n,m = X.shape[1], X.shape[0] 
        K = self.make_kernel_matrix(X,X)

        self.w = np.linalg.solve(K + self.C*np.identity(m), y)
        self.X = X
    
    def predict(self,x):
        return self.make_kernel_matrix(x,self.X) @ self.w

    #for saving state to file after training
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('make_kernel_matrix',None)
        return state

    def __setstate__(self, newstate):
        self.__dict__.update(newstate)
        self._kernel_matrix_creator()

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

    #for saving state to file after training
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('leaf_prediction_function',None)
        return state

    def __setstate__(self, newstate):
        self.__dict__.update(newstate)

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
            loss function for gradient boosting, could be either MSE (squared error, LS_Boost), ABS (abs error, LAD_TreeBost) or huber(M_TreeBost)
        huber_alpha_quantile
            quantile of how many values are considered as outliers in huber loss function, in each iteration huber delta parameter is calculated based on this
        """
        self.iters = iters
        self.max_tree_depth = max_tree_depth
        self.eta = eta
        self.sample_fraction = sample_fraction
        self.huber_alpha_quantile = huber_alpha_quantile
        self.loss = loss
        if loss == "MSE":
            self.gradient = lambda x,y,: x-y
        elif loss == "ABS":
            self.gradient = lambda x,y: np.sign(x-y)
        elif loss == "huber":
            self.gradient = lambda x,y,delta: x-y if np.abs(x-y) <= delta else delta*np.sign(x-y)
        else:
            raise("unsupported loss function")

    def update_function(self,f,tree):
        return lambda x: f(x) + self.eta*tree.predict_single(x)

    def _h(self,x):
        return self.start_value + self.eta*sum(tree.predict_single(x) for tree in self.trees)

    def fit(self,X,y):
        n,m = X.shape[1], X.shape[0]
        self.start_value = np.mean(y) if self.loss == "MSE" else np.median(y)
        self.trees = []

        for t in range(self.iters):
            random_indexes = np.random.choice(X.shape[0], max(1,int(m*self.sample_fraction)), replace=False)
            X2,y2 = X[random_indexes], y[random_indexes]

            if self.loss == "huber":
                r = y2 - np.apply_along_axis(self._h,1,X2)
                delta = np.quantile(r,self.huber_alpha_quantile)
                g = np.array([self.gradient(y2[i], self._h(X2[i]),delta) for i in range(len(y2))])
                X2 = np.append(X2, r.reshape(-1,1),axis=1)

                def leaf_prediction_function(Xy2):
                    r_tilde = np.median(Xy2[:,-2])
                    return r_tilde + np.mean(np.sign(Xy2[:,-2]-r_tilde)*np.minimum(np.abs(Xy2[:,-2] - r_tilde),delta))    

                tree = Decision_Tree_Regression(self.max_tree_depth,leaf_prediction_function)
                tree.fit(X2,g, ignored_last_features_n=1)

            else:
                g = np.array([self.gradient(y2[i], self._h(X2[i])) for i in range(len(y2))])
                tree = Decision_Tree_Regression(self.max_tree_depth, "mean" if self.loss == "MSE" else "median")
                tree.fit(X2,g)

            self.trees.append(tree)
            #y_pred = np.array([self._h(x) for x in X])
            #print("t =",t,"mse =",mse(y_pred,y), "abs =",mean_abs_error(y_pred,y))

    def predict_single(self,x):
        return self._h(x)

    def predict(self,X):
        return np.array([self.predict_single(x) for x in X])

    #for saving state to file after training
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('gradient',None)
        return state

    def __setstate__(self, newstate):
        self.__dict__.update(newstate)

class Random_Forest_Regression:
    def __init__(self, n_trees, max_depth, bootstrap=True, max_features="half"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.bootstrap = bootstrap

        if isinstance(max_features,int):
            self.n_features = lambda _ : max_features
        elif max_features == "sqrt":
            self.n_features = math.sqrt
        elif max_features == "half":
            self.n_features = lambda x : x/2
        else:
            self.n_features = lambda x : x
        
    
    def fit(self,X,y):
        self.trees = []
        n,m = X.shape[1], X.shape[0]
        for i in range(self.n_trees):
            random_columns = np.random.choice(X.shape[1],np.clip(int(self.n_features(X.shape[1])), 1 ,X.shape[1]), replace=False)
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

    #for saving state to file after training
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('n_features',None)
        return state

    def __setstate__(self, newstate):
        self.__dict__.update(newstate)

def outliers_fraction(y, y_pred, c):
    return (np.abs(y-y_pred) > c).sum()/len(y_pred)

def mean_abs_error(y, y_pred):
    return np.abs(y_pred-y).mean()

def mse(y, y_pred):
    return ((y_pred-y)**2).mean()

def r2_score(y, y_pred):
    y_mean = y.mean()
    s1 = sum((y-y_mean)**2)
    s2 = sum((y-y_pred)**2)
    return 1-s2/s1

def permutation_feature_importance(trained_model, X, y, iters):
    score = mse(y, trained_model.predict(X))
    feature_importance = np.array([0.]*X.shape[1])
    for i in range(X.shape[1]):
        for _ in range(iters):
            Xcp = np.copy(X)
            Xcp.T[i] = np.random.shuffle(Xcp.T[i])
            feature_importance[i] += np.abs(score-mse(y, trained_model.predict(Xcp)))/iters #predict on matrix with shuffled column
    return feature_importance / np.sum(feature_importance) 

if __name__ == '__main__':
    np.random.seed(config.seed)
    sns.set()
    kde_plots, legend_labels = [],[]

    for model,name in (
        (Gradient_Boosting_Regression(iters=100, max_tree_depth=4, eta=0.05, sample_fraction=0.6, loss="MSE"),"GB MSE"),
        (Gradient_Boosting_Regression(iters=100, max_tree_depth=5, eta=0.05,sample_fraction=0.6, loss="ABS"),"GB ABS"),
        (Gradient_Boosting_Regression(iters=100, max_tree_depth=5, eta=0.05, sample_fraction=0.6, loss="huber",huber_alpha_quantile=0.8),"GB huber"),
        (LinearRegression(C=1),"LR linear"),
        (LinearRegression(C=15000*10,kernel="polynomial",poly_degree=3,poly_const=1),"LR poly kernel"),
        (LinearRegression(C=1,kernel="gaussian",gaussian_gamma=0.009), "LR gaussian kernel"),
        (Decision_Tree_Regression(max_depth=8),"Decision tree"),
        (Random_Forest_Regression(n_trees = 30,max_depth=8,bootstrap=True),"Random Forest"),):
        

        train_mses, test_mses, test_abs_errs, r2s = [],[],[],[]
        for train_set_fraction in config.train_set_fractions:
            outliers_X = np.linspace(0,2,100)
            train_mse, test_mse,test_abs, r2, outliers_Y = 0.,0.,0.,0.,np.zeros(len(outliers_X))
            for it in range(config.iters):
                data_train_org, data_test_org = data_preparation.get_saved_data_from(filename_suffix=str(it)+"_"+str(train_set_fraction))
                data_train, data_test = np.copy(data_train_org), np.copy(data_test_org)
        
                np.random.shuffle(data_test)
                data_val,data_test = data_test[:len(data_test)//2], data_test[len(data_test)//2:]

                scaler = Scaler()

                data_train = scaler.fit_transform(data_train)
                data_val  = scaler.transform(np.copy(data_val))
                data_test  = scaler.transform(np.copy(data_test))

                pickle.dump(scaler, open('models/scaler.p', "wb" ))

                d_X_train = data_train[:,:-1]
                d_y_train = data_train[:,-1]

                d_X_test = data_test[:,:-1]
                d_y_test = data_test[:,-1]

                d_X_val = data_val[:,:-1]
                d_y_val = data_val[:,-1]

                model.fit(d_X_train, d_y_train)
                d_y_pred = model.predict(d_X_test)
                d_y_pred_tr = model.predict(d_X_train)

                train_mse += mse(d_y_train, d_y_pred_tr)/config.iters
                test_mse  += mse(d_y_test, d_y_pred)/config.iters
                test_abs  += mean_abs_error(d_y_test, d_y_pred)/config.iters

                r2 += r2_score(d_y_test, d_y_pred)/config.iters
                if train_set_fraction == 1:
                    outliers_Y += np.array([outliers_fraction(d_y_test, d_y_pred,x) for x in outliers_X])/config.iters
                    if it == 0:
                        plt.figure(1)
                        if len(kde_plots) == 0:
                            print("feature importances = ",permutation_feature_importance(model, d_X_test, d_y_test,iters=100))
                            sns.histplot(data=scaler.inverse_transform_column(np.array(d_y_test),-1),stat="density",kde=True,bins=10,binrange=(0,10),color="lightgreen")
                        kde_plots.append(sns.kdeplot(data=scaler.inverse_transform_column(np.array(d_y_pred),-1), palette=sns.color_palette()))     
                        legend_labels.append(name)
            

            pickle.dump(model, open('models/'+name+'.p', "wb" ))
            train_mses.append(train_mse)
            test_mses.append(test_mse)
            test_abs_errs.append(test_abs)
            r2s.append(r2)

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time, train_set_fraction, 'MSE on train: %.2f, MSE on test: %.2f, ABS on test: %.2f, R2: %.2f' % (train_mse, test_mse, test_abs, r2))

            if train_set_fraction == 1:
                plt.figure(2)
                plt.plot(outliers_X, outliers_Y)

        plt.figure(3)
        plt.plot(config.train_set_fractions, r2s)

        plt.figure(4)
        plt.plot(config.train_set_fractions, train_mses)
        plt.figure(5)
        plt.plot(config.train_set_fractions, test_mses)
        plt.figure(6)
        plt.plot(config.train_set_fractions, test_abs_errs)

    plt.figure(1)
    plt.legend(labels=['y']+legend_labels)

    for i in range(2,7):
        plt.figure(i)
        plt.legend(labels=legend_labels)

    for i in range(1,7):
        plt.figure(i)
        plt.savefig('plots/plot'+str(i)+'.png')

    plt.show()
