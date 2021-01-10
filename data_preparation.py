import numpy as np
import pandas as pd
from itertools import chain
import re


#change most popluar currencies to dollars and convert to int
def parse_currency(s):
    if not isinstance(s, str):
        return None
    currency_converter = {'$': 1., 'EUR': 1.23, 'INR': 0.014 , 'GBP': 1.36, 'CAD': 0.79, 'PLN': 0.27} #exchange rates from day 7.01.2021
    for name ,v in currency_converter.items():
        if name in s:
            return int(int(re.sub(r'\D+','',s))*v)
    return None


def make_dummies_from_list(col, max_classes = -1, make_others_class = False):
    name = col.name
    counts = pd.Series(chain(*col.str.lower().str.split(', '))).value_counts()
    lim = -1
    if max_classes != -1 and max_classes+1 < len(counts):
        lim = counts[max_classes]
    
    col = pd.get_dummies(col.str.lower().str.split(', ').apply(pd.Series).applymap(lambda x : x if x in counts and counts[x] > lim else 'other').stack()).sum(level=0)
    if make_others_class:
        col['other'] = col['other'].apply(lambda x: min(1,x))
    else:
        col = col.drop(columns=['other'])
    return col.add_prefix(name+"_")


def get_data(save_processed_df = False):
    df = pd.read_csv("data/IMDB movies.csv")[['year','genre','duration','country','language','budget','reviews_from_users','reviews_from_critics','votes','avg_vote']].dropna()
    df['year'] = df['year'].apply(pd.to_numeric, errors='coerce').dropna().astype(int)
    df['budget'] = df['budget'].apply(parse_currency)
    df = df[df['budget'].notnull()]

    df = pd.concat([df, make_dummies_from_list(df['genre'])],axis=1).drop(columns=['genre'])
    df = pd.concat([df, make_dummies_from_list(df['language'],max_classes=10,make_others_class=True)],axis=1).drop(columns=['language'])
    df = pd.concat([df, make_dummies_from_list(df['country'],max_classes=10,make_others_class=True)],axis=1).drop(columns=['country'])

    cols = df.columns.tolist()
    cols.remove('avg_vote')
    cols.append('avg_vote')
    df = df[cols]

    if save_processed_df:
        df.to_csv('frame.tmp',index=False)
    return df.to_numpy()

def get_data_from_tmp():
    return pd.read_csv('frame.tmp').to_numpy()

