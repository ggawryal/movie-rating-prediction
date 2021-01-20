import numpy as np
import pandas as pd
from itertools import chain
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from string import punctuation

import nltk
nltk.download('punkt')


def move_column_to_end(df, col):
    cols = df.columns.tolist()
    cols.remove(col)
    cols.append(col)
    return df[cols]


#change most popluar currencies to dollars and convert to int
def parse_currency(s):
    if not isinstance(s, str):
        return None
    currency_converter = {'$': 1., 'EUR': 1.23, 'INR': 0.014 , 'GBP': 1.36, 'CAD': 0.79, 'PLN': 0.27} #exchange rates from day 7.01.2021
    for name ,v in currency_converter.items():
        if name in s:
            return int(int(re.sub(r'\D+','',s))*v)
    return None

def make_dummies_from_list(col,  train_set_size, max_classes = -1, make_others_class = False):
    name = col.name
    
    counts = pd.Series(chain(*col.iloc[:train_set_size].str.lower().str.split(', '))).value_counts()
    lim = -1
    if max_classes != -1 and max_classes+1 < len(counts):
        lim = counts[max_classes]
    
    col = pd.get_dummies(col.str.lower().str.split(', ').apply(pd.Series).applymap(lambda x : x if x in counts and counts[x] > lim else 'other').stack()).sum(level=0)
    if make_others_class:
        col['other'] = col['other'].apply(lambda x: min(1,x))
    else:
        col = col.drop(columns=['other'])
    return col.add_prefix(name+"_")


def target_encoding_leave_out1(df, col, y, alpha, train_set_size):
    """
    For each observation and selected categorical feature col calculate mean of all other observations in trainig set having same value.
    If observation have multiple values from this feature (which are separeated by comma in df[col]), then calculate mean of these.
    """
    mask = [i < train_set_size for i in range(len(df))]
    counts = pd.Series(chain(*df[col][mask].str.lower().str.split(', '))).value_counts()
    means = pd.Series(index=counts.index, data=[df[mask][df[col][mask].str.lower().str.contains(name)][y].mean() for name in counts.index])

    mean_all = df[y][mask].mean()

    vs = []
    for is_in_train, cell, y_val in zip(mask,df[col],df[y]):
        v = 0
        for word in cell.split(', '):
            word = word.lower()
            if is_in_train:
                if counts[word] > 1:
                    others_mean = ((means[word]*counts[word])-y_val)/(counts[word]-1)
                    v += ((counts[word]-1) * others_mean + alpha * mean_all)/(counts[word]-1 + alpha)
                else:
                    v += mean_all
            else:
                if word in counts:
                    v += (counts[word] * means[word] + alpha * mean_all)/(counts[word] + alpha)
                else:
                    v += mean_all
        
        vs.append(v / len(cell.split(', ')))

    df[col+"_target"] = vs

def target_encoding_k_fold(df, col, y, alpha, train_set_size, k=5):
    #first, make target encoding of test set, which is the same as in leave out one target encoding (using whole training set)
    target_encoding_leave_out1(df,col,y,alpha,train_set_size)
    df = move_column_to_end(df,y)
    df.insert(len(df.columns)-1, col+'_k_folds_target', df[col+"_target"])
    df = df.drop(columns=[col+'_target'])


    #then, encode each group in train set using counts and means from other folds 
    for group in range(k):
        group_bounds = (group*train_set_size//k, (group+1)*train_set_size//k)
        out_of_group = [i < group_bounds[0] or (i >= group_bounds[1] and i < train_set_size) for i in range(len(df))]
        counts = pd.Series(chain(*df.iloc[out_of_group][col].str.lower().str.split(', '))).value_counts()
        means = pd.Series(index=counts.index, 
            data=[df.iloc[out_of_group,-1] [df.iloc[out_of_group][col].str.lower().str.contains(name)].mean() for name in counts.index])

        mean_all_others = df.iloc[out_of_group,-1].mean()
        vs = []

        for cell in df.iloc[group_bounds[0] : group_bounds[1]][col]:
            v = 0.0
            for word in cell.split(', '):
                word = word.lower()
                if word in counts:
                    v += (counts[word] * means[word] + alpha * mean_all_others)/(counts[word] + alpha)
                else:
                    v += mean_all_others
            vs.append(v/len(cell.split(', ')))

        df.iloc[group_bounds[0] : group_bounds[1],-2] = vs 

def count_max_occurences(col, train_set_size, pdSeriesMethod=pd.Series.max):
    vc = pd.Series(chain(*col.iloc[:train_set_size].str.lower().str.split(', '))).value_counts()
    return pdSeriesMethod(col.str.lower().str.split(', ').apply(pd.Series).applymap(lambda x : 0 if x not in vc else vc[x]),axis=1)

def stem_description(text):
    ps = PorterStemmer()
    return ', '.join(filter(lambda x : len(x) >= 4, set(re.sub(r'[^a-z]', '', ps.stem(word.lower())) for word in word_tokenize(text))))

def get_data(train_set_fraction, target_encoding=False, save_to_file = False):
    df = pd.read_csv("data/IMDB movies.csv")[['year','actors','director','genre','duration','country','language','budget','avg_vote','description']].dropna()
    df['year'] = df['year'].apply(pd.to_numeric, errors='coerce').dropna().astype(int)
    df['budget'] = df['budget'].apply(parse_currency)
    df = df[df['budget'].notnull()]
    df['description'] = df['description'].apply(stem_description)

    df = df.sample(frac=1) #shuffle
    train_set_size = int(train_set_fraction*len(df))

    df = pd.concat([df, make_dummies_from_list(df['genre'],train_set_size=train_set_size)],axis=1).drop(columns=['genre'])
    df = pd.concat([df, make_dummies_from_list(df['language'],train_set_size=train_set_size, max_classes=10,make_others_class=True)],axis=1).drop(columns=['language'])
    df = pd.concat([df, make_dummies_from_list(df['country'],train_set_size=train_set_size, max_classes=10,make_others_class=True)],axis=1).drop(columns=['country'])

    target_encoding_leave_out1(df,'director','avg_vote', alpha=10.0,train_set_size=train_set_size)
    target_encoding_leave_out1(df,'actors','avg_vote', alpha=10.0,train_set_size=train_set_size)
    target_encoding_k_fold(df,'description','avg_vote',alpha=5.0,train_set_size=train_set_size,k=3)

    df['number_of_actors'] = df['actors'].str.count(',').add(1)
    df['director_total_movies'] = count_max_occurences(df['director'],train_set_size=train_set_size)
    df['max_total_movies_actor'] = count_max_occurences(df['actors'],train_set_size=train_set_size)
    df['avg_total_movies_actor'] = count_max_occurences(df['actors'],train_set_size=train_set_size, pdSeriesMethod=pd.Series.mean)

    
    df = move_column_to_end(df,'avg_vote')
    df = df.drop(columns=['description', 'actors', 'director'])

    if save_to_file:
        df.iloc[:train_set_size].to_csv('frame_train.tmp',index=False)
        df.iloc[train_set_size:].to_csv('frame_test.tmp',index=False)
    return df.iloc[:train_set_size].to_numpy(), df.iloc[train_set_size:].to_numpy()

def get_data_from_tmp():
    print(pd.read_csv('frame_train.tmp').columns.tolist())
    return pd.read_csv('frame_train.tmp').to_numpy(), pd.read_csv('frame_test.tmp').to_numpy(), 

