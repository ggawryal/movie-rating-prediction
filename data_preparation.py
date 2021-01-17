import numpy as np
import pandas as pd
from itertools import chain
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from string import punctuation

import nltk
nltk.download('punkt')

#change most popluar currencies to dollars and convert to int
def parse_currency(s):
    if not isinstance(s, str):
        return None
    currency_converter = {'$': 1., 'EUR': 1.23, 'INR': 0.014 , 'GBP': 1.36, 'CAD': 0.79, 'PLN': 0.27} #exchange rates from day 7.01.2021
    for name ,v in currency_converter.items():
        if name in s:
            return int(int(re.sub(r'\D+','',s))*v)
    return None


def make_dummies_from_list(col,  mask=None, max_classes = -1, make_others_class = False):
    name = col.name

    if mask is None:
        mask = [True]*len(col)
    
    counts = pd.Series(chain(*col[mask].str.lower().str.split(', '))).value_counts()
    lim = -1
    if max_classes != -1 and max_classes+1 < len(counts):
        lim = counts[max_classes]
    
    col = pd.get_dummies(col.str.lower().str.split(', ').apply(pd.Series).applymap(lambda x : x if x in counts and counts[x] > lim else 'other').stack()).sum(level=0)
    if make_others_class:
        col['other'] = col['other'].apply(lambda x: min(1,x))
    else:
        col = col.drop(columns=['other'])
    return col.add_prefix(name+"_")

def count_max_occurences(col, train_set_mask, pdSeriesMethod=pd.Series.max):
    vc = pd.Series(chain(*col[train_set_mask].str.lower().str.split(', '))).value_counts()
    return pdSeriesMethod(col.str.lower().str.split(', ').apply(pd.Series).applymap(lambda x : 0 if x not in vc else vc[x]),axis=1)


def stem_description(text):
    ps = PorterStemmer()
    return ', '.join(filter(lambda x : len(x) >= 4, set(re.sub(r'[^a-z]', '', ps.stem(word.lower())) for word in word_tokenize(text))))


def get_data(train_set_fraction, target_encoding=False, save_processed_df = False):
    df = pd.read_csv("data/IMDB movies.csv")[['year','actors','director','genre','duration','country','language','budget','avg_vote','description']].dropna()

    df['year'] = df['year'].apply(pd.to_numeric, errors='coerce').dropna().astype(int)
    df['budget'] = df['budget'].apply(parse_currency)
    df = df[df['budget'].notnull()]
    df['description'] = df['description'].apply(stem_description)

    train_set_size = int(train_set_fraction*len(df))
    train_set_mask = np.array([True] * train_set_size + [False] * (len(df) - train_set_size))
    np.random.shuffle(train_set_mask)

    df = pd.concat([df, make_dummies_from_list(df['genre'])],axis=1).drop(columns=['genre'])
    df = pd.concat([df, make_dummies_from_list(df['language'],mask=train_set_mask, max_classes=10,make_others_class=True)],axis=1).drop(columns=['language'])
    df = pd.concat([df, make_dummies_from_list(df['country'],mask=train_set_mask, max_classes=10,make_others_class=True)],axis=1).drop(columns=['country'])
    df = pd.concat([df, make_dummies_from_list(df['description'],mask=train_set_mask,max_classes=200,make_others_class=False)],axis=1)
   
    df['number_of_actors'] = df['actors'].str.count(',').add(1)
    df['director_total_movies'] = count_max_occurences(df['director'],train_set_mask)
    df['max_total_movies_actor'] = count_max_occurences(df['actors'],train_set_mask)
    df['avg_total_movies_actor'] = count_max_occurences(df['actors'],train_set_mask, pd.Series.mean)

    cols = df.columns.tolist()
    cols.remove('description')
    cols.remove('actors')
    cols.remove('director')
    cols.remove('avg_vote')
    cols.append('avg_vote')
    df = df[cols]
    if save_processed_df:
        df.to_csv('frame.tmp',index=False)
    return df.to_numpy()

def get_data_from_tmp():
    print(pd.read_csv('frame.tmp').columns.tolist())
    return pd.read_csv('frame.tmp').to_numpy()

