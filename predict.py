import numpy as np
import pickle
import config
import data_preparation
from project import LinearRegression, Decision_Tree_Regression, Gradient_Boosting_Regression, Random_Forest_Regression, Scaler, mse, permutation_feature_importance

np.random.seed(config.seed)


model = pickle.load(open("models/GB huber.p", 'rb'))

data_train_org, data_test_org = data_preparation.get_saved_data_from(filename_suffix="0_1")
data_train, data_test = np.copy(data_train_org), np.copy(data_test_org)

np.random.shuffle(data_test)
data_val,data_test = data_test[:len(data_test)//2], data_test[len(data_test)//2:]

scaler = pickle.load(open("models/scaler.p", 'rb'))

data_test  = scaler.transform(np.copy(data_test))
d_X_test = data_test[:,:-1]
d_y_test = data_test[:,-1]

print("mse on test data = ",mse(d_y_test, model.predict(d_X_test)))
