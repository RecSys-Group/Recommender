__author__ = 'Jerry'

import numpy as np
import scipy as sp
from numpy.random import random
import math
import random
from DataModel.FileDataModel import *
from DataModel.MemeryDataModel import *
from sklearn.base import BaseEstimator
from sklearn import cross_validation
from sklearn import metrics
from sklearn import grid_search
from Eval.Evaluation import *

class LFM(BaseEstimator):
    def __init__(self, n=5, factors=50, learningrate=0.001, userregular=0.001, itemregular=0.001, iter = 5):
        self.factors = factors
        self.n = n
        self.learningrate = learningrate
        self.userregular = userregular
        self.itemregular = itemregular
        self.iter = iter

    def predict(self, testSamples):
        recList = []
        for user_item in testSamples:
            uid = self.dataModel.getUidByUser(user_item[0])
            recList.append(self.recommend(uid))
        return recList

    def fit(self, trainSamples, trainTargets):
        self.dataModel = MemeryDataModel(trainSamples, trainTargets)
        self.mu = np.array(trainTargets).mean()
        self.bu = np.zeros(self.dataModel.getUsersNum())
        self.bi = np.zeros(self.dataModel.getItemsNum())
        temp = math.sqrt(self.factors)
        self.qi = [[(0.1 * random.random() / temp) for j in range(self.factors)] for i in range(self.dataModel.getItemsNum())]
        self.pu = [[(0.1 * random.random() / temp) for j in range(self.factors)] for i in range(self.dataModel.getUsersNum())]
        lineData = self.dataModel.getLineData()
        lengthOfTrain = len(lineData)

        for step in range(self.iter):
            rmse_sum = 0.0
            hash = np.random.permutation(lengthOfTrain)
            for j in range(lengthOfTrain):
                n = hash[j]
                row = lineData[n]
                uid = self.dataModel.getUidByUser(row[0])
                iid = self.dataModel.getIidByItem(row[1])
                rating = row[2]
                eui = rating - self.predict_single(uid, iid)
                rmse_sum += eui**2
                self.bu[uid] += self.learningrate*(eui-self.userregular*self.bu[uid])
                self.bi[iid] += self.learningrate*(eui-self.itemregular*self.bi[iid])
                temp = self.qi[iid]
                self.qi[iid] += self.learningrate*(np.dot(eui, self.pu[uid]) - np.dot(self.itemregular, self.qi[iid]))
                self.pu[uid] += self.learningrate*(np.dot(eui, temp) - np.dot(self.userregular, self.pu[uid]))
            self.learningrate = self.learningrate * 0.93

    def predict_single(self, uid, iid):
        ans = self.mu + self.bi[iid] + self.bu[uid] + np.dot(self.qi[iid], self.pu[uid])
        if ans > 5:
            return 5
        elif ans < 1:
            return 1
        return ans
    def recommend(self, uid):
        predict_scores = []
        for i in range(self.dataModel.getItemsNum()):
            predict_scores.append(self.predict_single(uid, i))
        topN = np.argsort(np.array(predict_scores))[-1:-self.n-1:-1]
        return topN
    def score(self, testSamples, trueLabels):
        print 'LFM scoring ...'
        trueList = []
        recommendList= []
        user_unique = list(set(np.array(testSamples)[:,0]))
        for u in user_unique:
            uTrueIndex = np.argwhere(np.array(testSamples)[:,0] == u)[:,0]
            #true = [self.dataModel.getIidByItem(i) for i in list(np.array(testSamples)[uTrueIndex][:,1])]
            true = list(np.array(testSamples)[uTrueIndex][:,1])
            trueList.append(true)
            uid = self.dataModel.getUidByUser(u)
            pre = [self.dataModel.getItemByIid(i) for i in self.recommend(uid)]
            recommendList.append(pre)
        e = Eval()
        result = e.evalAll(trueList, recommendList)
        print 'LFM result:'+ '('+str(self.get_params())+')'+str((result)['F1'])
        return (result)['F1']


if __name__ == '__main__':
    nmf = LFM()
    data = pd.read_csv('../Data/bbg/transaction.csv')
    samples = [[int(i[0]), int(i[1])] for i in data.values[:,0:2]]
    targets = [1 for i in samples]
    parameters = {'n':[5]}

    clf = grid_search.GridSearchCV(nmf, parameters,cv=5)
    clf.fit(samples, targets)
    print(clf.grid_scores_)