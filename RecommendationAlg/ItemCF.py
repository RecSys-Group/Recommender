__author__ = 'Jerry'

import numpy as np

from DataModel.FileDataModel import *
from utils.Similarity import Similarity
from DataModel.MemeryDataModel import *
from sklearn.base import BaseEstimator
from sklearn import cross_validation
from sklearn import metrics
from sklearn import grid_search
from Eval.Evaluation import *

class ItemCF(BaseEstimator):

    def __init__(self, neighbornum=5, n=5):
        self.neighbornum = neighbornum
        self.similarity = Similarity('COSINE')
        self.n = n

    def predict(self,testSamples):
        recList = []
        for user_item in testSamples:
            uid = self.dataModel.getUidByUser(user_item[0])
            recList.append(self.recommend(uid))
        return recList

    def fit(self, trainSamples, trainTargets):
        self.dataModel = MemeryDataModel(trainSamples, trainTargets)
        itemsNum = self.dataModel.getItemsNum()
        self.simiMatrix = np.zeros((itemsNum, itemsNum))
        for i in range(itemsNum):
            for j in range(i+1, itemsNum):
                s = self.similarity.compute(self.dataModel.getUserIDsFromIid(i), self.dataModel.getUserIDsFromIid(j))
                self.simiMatrix[i][j] = self.simiMatrix[j][i] = s

    def neighborhood(self, itemID):
        neighbors = np.argsort(np.array(self.simiMatrix[itemID]))[-1:-self.neighbornum-1:-1]
        return neighbors

    def predict_single(self, userID, itemID):
        rating = 0.0
        for iid in self.neighborhood(itemID):
            if userID in self.dataModel.getUserIDsFromIid(iid):
                rating += self.simiMatrix[itemID][iid] * self.dataModel.getRating(userID, iid)
        return rating

    def recommend(self, u):
        if userID == -1:
            print 'not in test'
            return []
        else:
            userID = self.dataModel.getUidByUser(u)
            #interactedItems = self.dataModel.getItemIDsFromUid(userID)
            ratings = dict()
            for iid in self.dataModel.getItemIDsFromUid(userID):
                for niid in self.neighborhood(iid):
                    #if iid in interactedItems:
                        #continue
                    r = ratings.get(iid, 0)
                    ratings[iid] = r + self.simiMatrix[iid][niid] * self.dataModel.getRating(userID, niid)
            r = [x for (x, y) in sorted(ratings.items(), lambda a, b: cmp(a[1], b[1]), reverse=True)[:self.n]]
            return [self.dataModel.getItemByIid(i) for i in r]

    def score(self, testSamples, trueLabels):
        print 'Item_CF scoring ...'
        trueList = []
        recommendList= []
        user_unique = list(set(np.array(testSamples)[:,0]))
        for u in user_unique:
            uTrueIndex = np.argwhere(np.array(testSamples)[:,0] == u)[:,0]
            #true = [self.dataModel.getIidByItem(i) for i in list(np.array(testSamples)[uTrueIndex][:,1])]
            true = list(np.array(testSamples)[uTrueIndex][:,1])
            trueList.append(true)
            pre = self.recommend(u)
            recommendList.append(pre)
        e = Eval()
        result = e.evalAll(trueList, recommendList)
        print 'ItemCF result:'+'('+str(self.get_params())+')'+str((result)['F1'])
        return (result)['F1']


if __name__ == '__main__':
    itemcf = ItemCF('a')
    itemcf.train()
