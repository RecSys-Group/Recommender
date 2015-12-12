import numpy as np
import pandas as pd
import os
from sklearn.base import BaseEstimator
from sklearn import cross_validation
from sklearn import metrics
from DataModel.MemeryDataModel import *
from Eval.Evaluation import *
from sklearn import grid_search
from sklearn.cross_validation import StratifiedKFold

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
        #print trainSamples, trainTargets
        #print len(trainSamples), len(trainTargets)
        self.gen_items_popular(trainSamples, trainTargets)
        self.topN = np.argsort(np.array(self.popItems))[-1::-1]
        return self

    def recommend(self, uid):
        return [self.dataModel.getItemByIid(i) for i in self.topN[:self.n]]
    def score(self, testSamples, trueLabels):
        #print testSamples
        #print len(testSamples)
        trueList = []
        recommendList= []

        user_unique = list(set(np.array(testSamples)[:,0]))
        for u in user_unique:
            uTrueIndex = np.argwhere(np.array(testSamples)[:,0] == u)[:,0]
            #true = [self.dataModel.getIidByItem(i) for i in list(np.array(testSamples)[uTrueIndex][:,1])]
            true = list(np.array(testSamples)[uTrueIndex][:,1])
            trueList.append(true)
            pre = [self.dataModel.getItemByIid(i) for i in self.topN[:self.n]]
            recommendList.append(pre)
        e = Eval()
        result = e.evalAll(recommendList, trueList)
        print 'TopN result:'+'('+str(self.get_params())+')'+str((result)['F1'])
        return (result)['F1']


if __name__ == '__main__':
    tp = TopN()
    #data = pd.read_csv('../Data/bbg/transaction.csv')
    data = pd.read_csv('../Data/tinytest/format.csv')
    samples = [[int(i[0]), int(i[1])] for i in data.values[:,0:2]]
    #targets = [1 for i in samples]
    targets = [i for i in data.values[:,3]]
    parameters = {'n':[5]}

    labels = [int(i[0]) for i in data.values[:,0:2]]

    rec_cv =  StratifiedKFold(labels, 5)

    clf = grid_search.GridSearchCV(tp, parameters,cv=rec_cv)

    clf.fit(samples, targets)
    print(clf.grid_scores_)





