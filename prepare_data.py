# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 05:35:31 2019

@author: Raymond
"""

import pandas as pd
import numpy as np
import os

cols=['gameId', 'playId', 'nflId', 'start_x', 'start_y', 'delta_x', 'delta_y',
       'frameid', 'cat_dir']

if not os.path.isdir('./data/processed/'):
    os.mkdir('./data/processed/')
    
df = pd.read_msgpack('./input/feature_df.mp')
df.columns=cols
features=['start_x', 'start_y', 'delta_x', 'delta_y', 'cat_dir']
all_dfs=[]
all_ids=[]
count=0
for feature in features:
    select_cols=['frameid', 'gameId', 'playId', 'nflId'] + [feature] 
    df1=df[select_cols]
    df1.set_index(['frameid', 'gameId', 'playId', 'nflId'], inplace=True)
    df1=df1.unstack(level=0)
    if count==0:
        df2=df1
        count+=1
        df2.reset_index(inplace=True, drop=False)
        np.save('./data/processed/gameid.npy', df2['gameId'].values)
        np.save('./data/processed/playid.npy', df2['playId'].values)
        np.save('./data/processed/nflid.npy', df2['nflId'].values)
        all_ids=[ df2['gameId'].values, df2['playId'].values, df2['nflId'].values]
    
    out=df1.iloc[:,16:50]
    out = out.iloc[:, ::-1]
    print(out.shape)
    print(out.min(), out.max())
    #print(out.columns)
    out.fillna(0, inplace=True)
    all_dfs.append(out.values)
    np.save('./data/processed/{}_zeros.npy'.format(feature), out.values)
    


all_df=[np.expand_dims(i, axis=-1) for i in all_dfs]
all_id=[np.expand_dims(i, axis=-1) for i in all_ids]
all_id=np.concatenate(all_id, axis=-1)
all_df=np.concatenate(all_df, axis=-1)

print(np.isnan(all_df.sum().sum()))
print(np.isnan(all_id.sum().sum()))
print(all_id.shape, all_df.shape)    
np.save('./data/processed/all_id.npy'.format(feature), all_id)
np.save('./data/processed/all_df.npy'.format(feature), all_df)