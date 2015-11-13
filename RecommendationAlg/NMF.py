__author__ = 'Jerry'

import numpy as np
from sklearn.decomposition import ProjectedGradientNMF
import pandas as pd
import scipy.sparse as spr
import math
import random
from DataModel.FileDataModel import *
from BaseAlg import BaseAlg

class NMF(BaseAlg):
    def __init__(self, dataModel, paras):
        print "NMF begin"
        super(NMF, self).__init__(dataModel, 'NMF')
        self.feature_dims = int(float(paras['feature_dims']))
        temp = math.sqrt(self.feature_dims)
        self.qi = [[(0.1 * random.random() / temp) for j in range(self.feature_dims)] for i in range(self.dataModel.getItemsNum())]
        self.pu = [[(0.1 * random.random() / temp) for j in range(self.feature_dims)] for i in range(self.dataModel.getUsersNum())]

    def predict(self, uid, iid):
        return np.dot(self.pu[uid], self.qi[iid])

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
        print 'NMF training'
        V = self.dataModel.getTrain()
        model = ProjectedGradientNMF(n_components=self.feature_dims, max_iter=1000, nls_max_iter=10000)
        self.pu = model.fit_transform(V)
        self.qi = model.fit(V).components_.transpose()
        print model.reconstruction_err_
        t = pd.DataFrame(np.array(self.pu))
        t.to_csv(self.dataModel.conf.get('initpu'))
        t = pd.DataFrame(np.array(self.qi))
        t.to_csv(self.dataModel.conf.get('initqi'))

if __name__ == '__main__':
    pass










