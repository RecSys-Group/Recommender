import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
import pandas as pd

data = pd.read_csv('Data/bbg/transaction.csv')
samples = [[int(i[0]), int(i[1])] for i in data.values[:,0:2]]
targets = [1 for i in samples]
parameters = {'n':[5], 'neighbornum':[5]}
labels = [int(i[0]) for i in data.values[:,0:2]]
rec_cv =  StratifiedKFold(labels, 5)
for line in rec_cv:
    print line[0][:25], line[1][:10]
    print len(line[0]), len(line[1])
    raw_input()