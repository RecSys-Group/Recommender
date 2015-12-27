__author__ = 'Jerry'

import numpy as np

from DataModel.FileDataModel import *
from utils.Similarity import Similarity
from DataModel.MemeryDataModel import *
from sklearn.base import BaseEstimator
from sklearn import cross_validation
from sklearn import metrics
from sklearn import grid_search
from sklearn.cross_validation import StratifiedKFold
from Eval.Evaluation import *

class VSRank(BaseEstimator):

    def __init__(self, neighbornum=5, n=5):
        print 'vsrank begin'
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
        usersNum = self.dataModel.getUsersNum()
        itemsNum = self.dataModel.getItemsNum()

        self.dataModel = MemeryDataModel(trainSamples, trainTargets, isRating=False, hasTimes=True)
        self.T = [{} for i in range(usersNum)]
        for uid in range(usersNum):
            purchased_items = self.dataModel.getItemIDsFromUid(uid)
            for i in range(len(purchased_items)):
                for j in range(i+1, len(purchased_items)):
                    if self.dataModel.getRating(uid, i) > self.dataModel.getRating(uid, j):
                        key = str(purchased_items[i]) + " " + str(purchased_items[j])
                    elif self.dataModel.getRating(uid, i) < self.dataModel.getRating(uid, j):
                        key = str(purchased_items[j]) + " " + str(purchased_items[i])
                    self.T[uid][key] = 1

        idf = {}
        pair_sum = [[0]*itemsNum for i in range(itemsNum)]
        for uid in range(usersNum):
            for t, times in self.T[uid].iteritems():
                i1, i2 = t.split(" ")
                pair_sum[int(i1)][int(i2)] += 1
        for i1 in range(itemsNum):
            for i2 in range(itemsNum):
                if pair_sum[i1][i2] != 0:
                    key = str(i1) + ' ' + str(i2)
                    sum = pair_sum[i1][i2] + pair_sum[i2][i1]
                    alpha = log10(1+9.0*sum/usersNum)
                    idf[key] = alpha*log2(sum*1.0/pair_sum[i1][i2])+(1-alpha)

        W = [{} for i in range(usersNum)]
        for uid in range(usersNum):
            for t, times in self.T[uid].iteritems():
                i1, i2 = t.split(" ")
                diff = self.dataModel.getRating(uid, int(i1))-self.dataModel.getRating(uid, int(i2))
                tf = log2(1+abs(diff))
                if diff < 0:
                    tf = -tf
                W[uid][t] = tf * idf[t]

        self.simiMatrix = np.zeros((usersNum, usersNum))
        for i in range(usersNum):
            for j in range(i+1, usersNum):
                s = self.cos(W[i], W[j])
                self.simiMatrix[i][j] = self.simiMatrix[j][i] = s

    def cos(self, dict1, dict2):
        product = 0.0
        m1 = 0.0
        m2 = 0.0
        for k, v in dict1.iteritems():
            m1 += v**2
            if dict2.has_key(k):
                product += v * dict2[k]
        for k, v in dict2.iteritems():
            m2 += v**2
        if product == 0:
            return 0
        else:
            return product/sqrt(m1)/sqrt(m2)

    def tau(self, dict1, dict2, u1, u2):
        pass


    def neighborhood(self, userID):
        neighbors = np.argsort(np.array(self.simiMatrix[userID]))[-1:-self.neighbornum-1:-1]
        return neighbors

    def predict_single(self, userID, itemID):
        rating = 0.0
        for uid in self.neighborhood(userID):
            if itemID in self.dataModel.getItemIDsFromUid(uid):
                rating += self.simiMatrix[userID][uid] * self.dataModel.getRating(uid, itemID)
        return rating

    def recommend(self, u):
        userID = self.dataModel.getUidByUser(u)
        if userID == -1:
            print 'not in test'
            return []
        else:
            return self.recommend_listwise(userID)

    def recommend_pointwise(self, userID):
        #interactedItems = self.dataModel.getItemIDsFromUid(userID)
        ratings = dict()
        for uid in self.neighborhood(userID):
            for iid in self.dataModel.getItemIDsFromUid(uid):
                #if iid in interactedItems:
                    #continue
                r = ratings.get(iid, 0)
                ratings[iid] = r + self.simiMatrix[userID][uid] * self.dataModel.getRating(uid, iid)
        r = [x for (x, y) in sorted(ratings.items(), lambda a, b: cmp(a[1], b[1]), reverse=True)[:self.n]]
        return [self.dataModel.getItemByIid(i) for i in r]

    def recommend_pairwise(self, userID):
        pass

    def recommend_listwise(self, userID):
        itemsNum = self.dataModel.getItemsNum()
        M = [[0]*itemsNum for i in range(itemsNum)]
        for uid in self.neighborhood(userID):
            for t, times in self.T[uid].iteritems():
                i1, i2 = t.split(" ")
                M[int(i1)][int(i2)] += 1
        for m in xrange(itemsNum):
            for n in xrange(itemsNum):
                for k in xrange(itemsNum):
                    M[n][k] = max(M[n][k], min(M[n][m], M[m][k]))
        rank = [0]*itemsNum
        for m in range(itemsNum):
            for n in range(itemsNum):
                if n != m and M[m][n] > M[n][m]:
                    rank[m] += 1
        r = [x for (x, y) in sorted(zip(range(itemsNum), rank), lambda a, b: cmp(a[1], b[1]))[:self.n]]
        return [self.dataModel.getItemByIid(i) for i in r]

    def score(self, testSamples, trueLabels):
        print 'vsrank scoring ...'
        #print len(testSamples)
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
        result = e.evalAll(recommendList, trueList)
        print 'vsrank result:'+'('+str(self.get_params())+')'+str((result)['F1'])
        return (result)['F1']

if __name__ == '__main__':
    a = [1, 2]
    b = [2, 4]
    print cos(a, b)
