import numpy as np
import pandas as pd
import os
from sklearn.base import BaseEstimator
from sklearn import cross_validation
from sklearn import metrics
from DataModel.MemeryDataModel import *
from Eval.Evaluation import *
from sklearn import grid_search

class TopN(BaseEstimator):

    def __init__(self, n=5):
        self.n = n

    def gen_items_popular(self, trainSamples, trainTargets, hasTimes=False):
        self.dataModel = MemeryDataModel(trainSamples, trainTargets)
        itempopular = np.zeros(self.dataModel.getItemsNum())
        uids = self.dataModel.getData().nonzero()[0]
        iids = self.dataModel.getData().nonzero()[1]
        for i in range(len(iids)):
            iid = iids[i]
            itempopular[iid] += 1
        self.popItems = itempopular

    def predict(self, testSamples):
        recommend_lists = []
        for user_item in testSamples:
            if self.dataModel.getIidByItem(user_item[1]) in self.topN[:self.n]:
                recommend_lists.append(1)
            else:
                recommend_lists.append(0)
        return recommend_lists

    def fit(self, trainSamples, trainTargets):
        self.gen_items_popular(trainSamples, trainTargets)
        self.topN = np.argsort(np.array(self.popItems))[-1::-1]
        return self

    def score(self, testSamples, trueLabels):
        trueList = []
        recommendList= []

        user_unique = list(set(np.array(testSamples)[:,0]))
        for u in user_unique:
            uTrueIndex = np.argwhere(np.array(testSamples)[:,0] == u)[:,0]
            true = [self.dataModel.getIidByItem(i) for i in list(np.array(testSamples)[uTrueIndex][:,1])]
            #true = list(np.array(testSamples)[uTrueIndex][:,1])
            trueList.append(true)
            recommendList.append(self.topN[:self.n])
        e = Eval()
        result = e.evalAll(trueList, recommendList)
        print 'TopN result:'+'('+str(self.n)+')'+str((result)['F1'])
        return (result)['F1']


if __name__ == '__main__':
    tp = TopN()
    data = pd.read_csv('../Data/tinytest/format.csv')
    samples = [[int(i[0]), int(i[1])] for i in data.values[:,0:2]]
    targets = [int(i) for i in data.values[:,3]]
    parameters = {'n':[5,10,15,20]}

    clf = grid_search.GridSearchCV(tp, parameters,cv=5)
    clf.fit(samples, targets)
    print(clf.grid_scores_)





