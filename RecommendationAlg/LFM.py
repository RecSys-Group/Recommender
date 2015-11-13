__author__ = 'Jerry'

import numpy as np
import scipy as sp
from numpy.random import random
import math
import random
from BaseAlg import BaseAlg
from DataModel.FileDataModel import *

class LFM(BaseAlg):
    def __init__(self, dataModel, paras):
        print "LFM begin"
        super(LFM, self).__init__(dataModel, 'LFM')
        if not isinstance(dataModel, FileDataModelInRow):
            raise TypeError(dataModel + ' is not an instance of FileDataModelInRow')
        self.feature_dims = int(float(paras['feature_dims']))
        self.mu = self.dataModel.getAve()
        self.bu = np.zeros(self.dataModel.getUsersNum())
        self.bi = np.zeros(self.dataModel.getItemsNum())
        temp = math.sqrt(self.feature_dims)
        self.qi = [[(0.1 * random.random() / temp) for j in range(self.feature_dims)] for i in range(self.dataModel.getItemsNum())]
        self.pu = [[(0.1 * random.random() / temp) for j in range(self.feature_dims)] for i in range(self.dataModel.getUsersNum())]
        self.steps = int(float(paras['steps']))
        self.gamma = int(float(paras['gamma']))
        self.Lambda = int(float(paras['lambda']))


    def predict(self, uid, iid):
        ans = self.mu + self.bi[iid] + self.bu[uid] + np.dot(self.qi[iid], self.pu[uid])
        if ans > 5:
            return 5
        elif ans < 1:
            return 1
        return ans

    def recommend(self, uid, N=5):
        predict_scores = []
        for i in range(self.dataModel.getItemsNum()):
            predict_scores.append(self.predict(uid, i))
        topN = np.argsort(np.array(predict_scores))[-1:-N-1:-1]
        return topN

    def recommendAllUserInTest(self, N=5):
        recList = []
        for uid in self.dataModel.getUserIDsInTest():
            recList.append(self.recommend(uid, N))
        return recList

    def train(self):
        lengthOfTrain = self.dataModel.getLenOfTrain()
        gamma = self.gamma
        for step in range(self.steps):
            print 'the ',step,'-th  step is running'
            rmse_sum = 0.0
            hash = np.random.permutation(lengthOfTrain)
            for j in range(lengthOfTrain):
                n = hash[j]
                row = self.dataModel.getRow(n)
                uid = row[0]
                iid = row[1]
                rating = row[2]
                eui = rating - self.predict(uid, iid)
                rmse_sum += eui**2
                self.bu[uid] += gamma*(eui-self.Lambda*self.bu[uid])
                self.bi[iid] += gamma*(eui-self.Lambda*self.bi[iid])
                temp = self.qi[iid]
                self.qi[iid] += gamma*(np.dot(eui, self.pu[uid]) - np.dot(self.Lambda, self.qi[iid]))
                self.pu[uid] += gamma*(np.dot(eui, temp) - np.dot(self.Lambda, self.pu[uid]))
            gamma = gamma * 0.93
            print "the rmse of this step on train data is ", np.sqrt(rmse_sum/lengthOfTrain)
