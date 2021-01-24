import numpy as np
import pickle
import config
import data_preparation
import pandas as pd
from project import LinearRegression, Decision_Tree_Regression, Gradient_Boosting_Regression, Random_Forest_Regression, Scaler, mse, permutation_feature_importance
from data_preparation import transform_test, One_Hot_Encoder, Target_Encoder_K_Fold, Target_Encoder_Leave_Out1, Count_Mean_Statistics

#enter your movie data here and run this script
movies_data = [{
        'year': 2013,
        'actors': "Brett Beoubay, Jamie Bernstein, Ashley Braud, Philippe Brenninkmeyer, Rebecca Collins, Jenn Foreman, Andrea Frankle, Anthony Michael Frederick, Derrick Freeman, Pell James, Arabella Landrum, Ever Eloise Landrum, Grace LaRocca, Michael Patrick Rogers, Johnathon Schaech",
        'director':'Paul Soter',
        'genre':'Horror',
        'duration': 87,
        'country': 'USA',
        'language': 'English',
        'budget':  '10000',
        'description': "Alex and Penny are sick of the hectic city life and decide a move out to the country to raise their newborn child. But they don't expect the horrors threatening their relationship and family"
    },
]



df = pd.DataFrame(movies_data, index=range(len(movies_data)))
X = transform_test(df)
model = pickle.load(open("models/GB huber.p", 'rb'))
scaler = pickle.load(open("models/scaler.p", 'rb'))
X = scaler.tranform_without_y(X)
print('estimated rating: ', scaler.inverse_transform_column(model.predict(X), -1))
