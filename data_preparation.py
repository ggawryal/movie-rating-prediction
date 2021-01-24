import numpy as np
import pandas as pd
from itertools import chain
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from string import punctuation
import pickle

import config

np.random.seed(config.seed)


def move_column_to_end(df, col):
    cols = df.columns.tolist()
    cols.remove(col)
    cols.append(col)
    return df[cols].copy()


#change most popluar currencies to dollars and convert to int
def parse_currency(s):
    if not isinstance(s, str):
        return None
    if s.isnumeric():
        return int(s)
    currency_converter = {'$': 1., 'EUR': 1.23, 'INR': 0.014 , 'GBP': 1.36, 'CAD': 0.79, 'PLN': 0.27} #exchange rates from day 7.01.2021
    for name ,v in currency_converter.items():
        if name in s:
            return int(int(re.sub(r'\D+','',s))*v)
    return None

class One_Hot_Encoder:
    def __init__(self, max_classes=-1):
        self.max_classes = max_classes

    def fit(self, df_train, col):
        counts = pd.Series(chain(*df_train[col].str.lower().str.split(', '))).value_counts()

        if self.max_classes == -1 or self.max_classes >= len(counts):
            self.max_classes = len(counts)
        
        self.classes = counts.index.tolist()[:self.max_classes]
        
    def transform(self, df, col):
        for x in self.classes:
            df[col+"_"+x] = df[col].str.lower().str.contains(x,regex=False).astype(float)
        return df

class Count_Mean_Statistics:
    def __init__(self, alpha, avg_func="mean"):
        self.alpha = alpha
        self.avg_func = avg_func

    def fit(self, df_train, col, y):
        self.counts = pd.Series(chain(*df_train[col].str.lower().str.split(', '))).value_counts()
        self.means = pd.Series(index=self.counts.index, data=[df_train[df_train[col].str.lower().str.contains(name,regex=False)][y].mean() for name in self.counts.index])
        self.mean_all = df_train[y].mean()

    def transform_test(self, df, col,col_suffix):
        acc = (lambda x,y: x+y) if self.avg_func == "mean" else max
        vs = []
        for cell in df[col]:
            v = 0
            for word in cell.split(', '):
                word = word.lower()
                if word in self.counts:
                    v = acc(v, (self.counts[word] * self.means[word] + self.alpha * self.mean_all)/(self.counts[word] + self.alpha))
                else:
                    v = acc(v, self.mean_all)

            if self.avg_func == "mean":
                v /= len(cell.split(', '))
            vs.append(v)

        df[col+col_suffix] = vs
        return df

class Target_Encoder_Leave_Out1:
    def __init__(self, alpha, avg_func="mean"):
        self.alpha = alpha
        self.avg_func = avg_func
        self.cms = Count_Mean_Statistics(alpha,avg_func)

    def fit_transform_train(self, df_train, col, y):
        df = move_column_to_end(df_train,y)
        self.cms.fit(df, col, y)

        acc = (lambda x,y: x+y) if self.avg_func == "mean" else max

        vs = []
        for cell, y_val in zip(df[col],df[y]):
            v = 0
            for word in cell.split(', '):
                word = word.lower()
                if self.cms.counts[word] > 1:
                    others_mean = ((self.cms.means[word]*self.cms.counts[word])-y_val)/(self.cms.counts[word]-1)
                    v = acc(v, ((self.cms.counts[word]-1) * others_mean + self.alpha * self.cms.mean_all)/(self.cms.counts[word]-1 + self.alpha))
                else:
                    v = acc(v, self.cms.mean_all)

            if self.avg_func == "mean":
                v /= len(cell.split(', '))
            vs.append(v)

        df[col+"_target"] = vs
        return df

    def fit_transform_both(self, df, col, y, train_set_size):
        df_train = df.iloc[:train_set_size].copy()
        df_test  = df.iloc[train_set_size:].copy()

        df_train = self.fit_transform_train(df_train,col,y)
        df_test =  self.cms.transform_test(df_test,col,'_target')
        return pd.concat((df_train, df_test))

    def transform(self, df, col):
        return self.cms.transform_test(df,col,'_target')

class Target_Encoder_K_Fold:
    def __init__(self, alpha,k, avg_func="mean"):
        self.alpha = alpha
        self.avg_func = avg_func
        self.k = k
        self.cms = Count_Mean_Statistics(alpha,avg_func)
    
    def fit_transform_train(self, df, col, y):
        self.cms.fit(df,col,y)
        M = len(df)
        df[col+'_k_folds_target'] = [0.0]*M
        df = move_column_to_end(df,y)

        #then, encode each group in train set using counts and means from other folds 
        for group in range(self.k):
            group_bounds = (group*M//self.k, (group+1)*M//self.k)
            out_of_group = [i < group_bounds[0] or i >= group_bounds[1] for i in range(M)]

            c = Count_Mean_Statistics(alpha=self.alpha, avg_func=self.avg_func)
            c.fit(df.iloc[out_of_group],col,y)

            cp = c.transform_test(df.iloc[group_bounds[0] : group_bounds[1]].copy(),col,'_k_folds_target')
            df.iloc[group_bounds[0] : group_bounds[1],-2] = cp[col+'_k_folds_target'] 

        return df

    def fit_transform_both(self, df, col, y, train_set_size):
        df_train = df.iloc[:train_set_size].copy()
        df_test  = df.iloc[train_set_size:].copy()

        df_train = self.fit_transform_train(df_train,col,y)
        df_test =  self.cms.transform_test(df_test,col,'_k_folds_target')
        return pd.concat((df_train, df_test))
    
    def transform(self, df, col):
        return self.cms.transform_test(df,col,'_k_folds_target')

def stem_description(text):
    ps = PorterStemmer()
    return ', '.join(filter(lambda x : len(x) >= 4, set(re.sub(r'[^a-z]', '', ps.stem(word.lower())) for word in word_tokenize(text))))

def transform_test(df):
    #df = df['year','actors','director','genre','duration','country','language','budget','description'].dropna()
    df['year'] = df['year'].apply(pd.to_numeric, errors='coerce').dropna().astype(int)
    df['budget'] = df['budget'].apply(parse_currency)
    df = df[df['budget'].notnull()]
    nltk.download('punkt')
    df['description'] = df['description'].apply(stem_description)

    for x in ['genre','language','country','description','director','actors']:
        encoder = pickle.load(open('models/'+x+'_encoder.p', 'rb' ))
        df = encoder.transform(df,x)

    df['number_of_actors'] = df['actors'].str.count(',').add(1)
    df = df.drop(columns=['description', 'actors', 'director','genre','language','country'])
    return df.to_numpy()

def get_data(train_set_fraction, save_to_file = False,filename_suffix='', test_set_fraction = None):
    df = pd.read_csv("data/IMDB movies.csv")[['year','actors','director','genre','duration','country','language','budget','avg_vote','description']].dropna()
    df['year'] = df['year'].apply(pd.to_numeric, errors='coerce').dropna().astype(int)
    df['budget'] = df['budget'].apply(parse_currency)
    df = df[df['budget'].notnull()]

    nltk.download('punkt')
    df['description'] = df['description'].apply(stem_description)

    df = df.sample(frac=1,random_state=config.seed) #shuffle
    train_set_size = int(train_set_fraction*len(df))

    if test_set_fraction is not None and train_set_fraction + test_set_fraction < 1:
        df = df.iloc[:int((train_set_fraction + test_set_fraction)*len(df))]    
    
    genre_encoder = One_Hot_Encoder()
    genre_encoder.fit(df.iloc[:train_set_size],'genre')
    df = genre_encoder.transform(df,'genre')
    pickle.dump(genre_encoder, open('models/genre_encoder.p', 'wb' ))

    lang_encoder = Target_Encoder_K_Fold(alpha=10,k=5)
    df = lang_encoder.fit_transform_both(df,'language','avg_vote',train_set_size)
    pickle.dump(lang_encoder, open('models/language_encoder.p', 'wb' ))

    country_encoder = Target_Encoder_K_Fold(alpha=10,k=5)
    df = country_encoder.fit_transform_both(df,'country','avg_vote',train_set_size)
    pickle.dump(country_encoder, open('models/country_encoder.p', 'wb' ))

    description_encoder = Target_Encoder_K_Fold(alpha=5,k=5,avg_func="max")
    df = description_encoder.fit_transform_both(df,'description','avg_vote',train_set_size)
    pickle.dump(description_encoder, open('models/description_encoder.p', 'wb' ))

    director_encoder = Target_Encoder_Leave_Out1(alpha=10)
    df = director_encoder.fit_transform_both(df,'director','avg_vote', train_set_size)
    pickle.dump(director_encoder, open('models/director_encoder.p', 'wb' ))


    actors_encoder = Target_Encoder_Leave_Out1(alpha=5)
    df = actors_encoder.fit_transform_both(df,'actors','avg_vote', train_set_size)
    pickle.dump(actors_encoder, open('models/actors_encoder.p', 'wb' ))


    df['number_of_actors'] = df['actors'].str.count(',').add(1)
    df = move_column_to_end(df,'avg_vote')
    df = df.drop(columns=['description', 'actors', 'director','genre','language','country'])

    if save_to_file:
        df.iloc[:train_set_size].to_csv('frames/frame_train'+filename_suffix+'.dataframe',index=False)
        df.iloc[train_set_size:].to_csv('frames/frame_test'+filename_suffix+'.dataframe',index=False)
    return df.iloc[:train_set_size].to_numpy(), df.iloc[train_set_size:].to_numpy()

def get_saved_data_from(filename_suffix=''):
    return pd.read_csv('frames/frame_train'+filename_suffix+'.dataframe').to_numpy(), pd.read_csv('frames/frame_test'+filename_suffix+'.dataframe').to_numpy(), 





if __name__ == '__main__':
    print("Are you sure you want to re generate train and test dataframes from csv file (this may take a few hours)?")
    print("Type 'YES' to confirm")
    txt = input()

    if txt == "YES":
        print("re generating dataframes...")
        for it in range(config.iters):
            for f in config.train_set_fractions:
                data_train_org, data_test_org = get_data(f*config.train_fraction,save_to_file=True, test_set_fraction=(1-config.train_fraction),filename_suffix=str(it)+"_"+str(f))
    else:
        print('canceled')