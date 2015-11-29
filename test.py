import numpy as np
import pandas as pd
file = 'Data/bbg/transaction.csv'
data = pd.read_csv(file)

uid = data['uid']
iid = pd.Series([int(i) for i in data['iid'].values])
time = data['time']
rating = data['rating']
review = data['review']
print iid
data.to_csv('transaction.csv', index=False, header=['uid','iid','time','rating','review'])