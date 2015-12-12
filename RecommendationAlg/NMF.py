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
        print 'nmf begin'
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
        #print 'train user:' + str(self.dataModel.getUsersNum())
        V = self.dataModel.getData()
        model = ProjectedGradientNMF(n_components=self.factors, max_iter=1000, nls_max_iter=1000)
        self.pu = model.fit_transform(V)
        self.qi = model.fit(V).components_.transpose()

    def predict_single(self, uid, iid):
        return np.dot(self.pu[uid], self.qi[iid])
    def recommend(self, u):
        uid = self.dataModel.getUidByUser(u)
        if uid == -1:
            print 'not in test'
            return []
        else:
            predict_scores = []
            for i in range(self.dataModel.getItemsNum()):
                predict_scores.append(self.predict_single(uid, i))
            topN = np.argsort(np.array(predict_scores))[-1:-self.n-1:-1]
            return [self.dataModel.getItemByIid(i) for i in topN]
    def score(self, testSamples, trueLabels):
        print 'NMF scoring ...'
        trueList = []
        recommendList= []
        user_unique = list(set(np.array(testSamples)[:,0]))
        #print 'test user:' + str(len(user_unique))
        for u in user_unique:
            uTrueIndex = np.argwhere(np.array(testSamples)[:,0] == u)[:,0]
            #true = [self.dataModel.getIidByItem(i) for i in list(np.array(testSamples)[uTrueIndex][:,1])]
            true = list(np.array(testSamples)[uTrueIndex][:,1])
            trueList.append(true)
            pre = self.recommend(u)
            recommendList.append(pre)
        e = Eval()
        result = e.evalAll(recommendList, trueList)
        print 'NMF result:'+'('+str(self.get_params())+')' + str((result)['F1'])
        return (result)['F1']

if __name__ == '__main__':
     nmf = NMF()
     data = pd.read_csv('../Data/bbg/transaction.csv')
     samples = [[int(i[0]), int(i[1])] for i in data.values[:,0:2]]
     targets = [1 for i in samples]
     parameters = {'n':[5], 'factors':[50]}


     clf = grid_search.GridSearchCV(nmf, parameters,cv=5)
     clf.fit(samples, targets)
     print(clf.grid_scores_)










