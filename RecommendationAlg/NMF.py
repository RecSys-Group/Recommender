__author__ = 'Jerry'

import numpy as np
from sklearn.decomposition import ProjectedGradientNMF
import pandas as pd
import scipy.sparse as spr
import math
import random
from DataModel.MemeryDataModel import *
from sklearn.base import BaseEstimator
from sklearn import cross_validation
from sklearn import metrics
from sklearn import grid_search
from Eval.Evaluation import *

class NMF(BaseEstimator):
    def __init__(self, n=5, factors=50):
        self.n = n
        self.factors = factors

    def predict(self, testSamples):
        recommend_lists = []
        for user_item in testSamples:
            uid = self.dataModel.getUidByUser(user_item[0])
            iid = self.dataModel.getUidByUser(user_item[1])
            recommend_lists.append(np.dot(self.pu[uid], self.qi[iid]))
        return recommend_lists

    def fit(self, trainSamples, trainTargets):
        self.dataModel = MemeryDataModel(trainSamples, trainTargets)
        V = self.dataModel.getData()
        model = ProjectedGradientNMF(n_components=self.factors, max_iter=100, nls_max_iter=100)
        self.pu = model.fit_transform(V)
        self.qi = model.fit(V).components_.transpose()

    def predict_single(self, uid, iid):
        return np.dot(self.pu[uid], self.qi[iid])
    def recommend(self, uid):
        predict_scores = []
        for i in range(self.dataModel.getItemsNum()):
            predict_scores.append(self.predict_single(uid, i))
        topN = np.argsort(np.array(predict_scores))[-1:-self.n-1:-1]
        return topN
    def score(self, testSamples, trueLabels):
        print 'NMF scoring ...'
        trueList = []
        recommendList= []
        user_unique = list(set(np.array(testSamples)[:,0]))
        for u in user_unique:
            uTrueIndex = np.argwhere(np.array(testSamples)[:,0] == u)[:,0]
            true = [self.dataModel.getIidByItem(i) for i in list(np.array(testSamples)[uTrueIndex][:,1])]
            #true = list(np.array(testSamples)[uTrueIndex][:,1])
            trueList.append(true)
            uid = self.dataModel.getUidByUser(u)
            recommendList.append(self.recommend(uid))
        e = Eval()
        result = e.evalAll(trueList, recommendList)
        print 'NMF result:'+'('+str(self.n) + str(self.factors)+')' + str((result)['F1'])
        return (result)['F1']

if __name__ == '__main__':
     nmf = NMF()
     data = pd.read_csv('../Data/tinytest/format.csv')
     samples = [[int(i[0]), int(i[1])] for i in data.values[:,0:2]]
     targets = [int(i) for i in data.values[:,3]]
     parameters = {'n':[5], 'factors':[50]}

     clf = grid_search.GridSearchCV(nmf, parameters,cv=5)
     clf.fit(samples, targets)
     print(clf.grid_scores_)










