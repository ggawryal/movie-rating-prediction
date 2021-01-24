# movie-rating-prediction

Aim of this project is to predict movie rating basing on metadata, like production year, budget, actors, director, duration, genre, description, language and country.
The dataset for this project were get from IMDb and can be downloaded from [kaggle](https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset?select=IMDb+movies.csv)

# Required python modules
* numpy
* matplotlib
* seaborn
* pandas
* nltk

# Usage
Trained models and prepared datasets are availible in the release section. Simply download this release, install required python modules, and run ```predict.py``` file.
It is also possible to edit this file and change or add more movies to ```movies_data``` list. Note that dictionary should have all keys like in example and none of values could be null. 

# Training models
To train models, download dataset from kaggle and extract files to ```data/``` directory. Then, run ```data_preparation.py``` script to generate and process training and test data from csv file. Then, run ```project.py``` script. It will draw learning curves, print statistics like
mean squared error or R2 measure and save models using pickle to ```models/``` directory. 

You can edit ```config.py``` file and change ```iters``` to ```1``` if you want to generate just one train and test dataset and ```train_set_fractions``` to ```[1.0]``` if you don't want to plot learning curves. 

# Results
Best model achieves 0.5 R2 score on test data
