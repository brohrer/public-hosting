"""
An example illustrating how to index pandas DataFrames. 
"""
import numpy as np
import pandas as pd

mtns = pd.DataFrame([
    {'name': 'Mount Everest',
        'height (m)': 8848,
        'summited': 1953,
        'mountain range': 'Mahalangur Himalaya'},
    {'name': 'K2',
        'height (m)': 8611,
        'summited': 1954,
        'mountain range': 'Baltoro Karakoram'},
    {'name': 'Kangchenjunga',
        'height (m)': 8586,
        'summited': 1955,
        'mountain range': 'Kangchenjunga Himalaya'},
    {'name': 'Lhotse',
        'height (m)': 8516,
        'summited': 1956,
        'mountain range': 'Mahalangur Himalaya'},
])
mtns.set_index('name', inplace=True)

print(mtns)
print(mtns.loc[:, 'height (m)'])
print(mtns.loc[:, 'height (m)'].values)
print(mtns.loc[:, 'mountain range'])
print(mtns.loc['K2', :])
print(mtns.loc['K2', 'mountain range'])
print(mtns.loc[:, 'height (m)': 'summited'])
print(mtns.loc[:, ['height (m)', 'summited']])
print(mtns.loc[mtns.loc[:, 'summited'] > 1954, :])
print(mtns.iloc[0, :])
print(mtns.iloc[:, 2])
print(mtns.iloc[0, 2])
print(mtns.iloc[[1, 3], :])
print(mtns.iloc[:, 0:2])
print(mtns.iloc[:, 0:2].loc['K2', :])
print(mtns.iloc[:, 0].loc['K2', :])
