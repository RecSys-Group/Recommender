__author__ = 'Jerry'

import numpy as np
from sklearn.decomposition import ProjectedGradientNMF
import pandas as pd
import scipy.sparse as spr
import math
import random
from DataModel.FileDataModel import FileDataModel

class NMF:
    def __init__(self, dataModel, feature_dims=10):
        print "NMF begin"
        self.dataModel = dataModel
        self.feature_dims = feature_dims
        temp = math.sqrt(self.feature_dims)
        self.qi = [[(0.1 * random.random() / temp) for j in range(self.feature_dims)] for i in range(self.dataModel.getItemsNum())]
        self.pu = [[(0.1 * random.random() / temp) for j in range(self.feature_dims)] for i in range(self.dataModel.getUsersNum())]

    def predict(self, uid, iid):
        return self.InerProduct(self.pu[uid], self.qi[iid])

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

    def InerProduct(self, v1, v2):
        result = 0
        for i in range(len(v1)):
            result += v1[i] * v2[i]
        return result

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
        print("model generation over")

if __name__ == '__main__':
    users = 'D:/Desktop/recommender/Data/v3/v3_users'
    items = 'D:/Desktop/recommender/Data/v3/v3_items'
    train = 'D:/Desktop/recommender/Data/v3/v3_train_records'
    test = 'D:/Desktop/recommender/Data/v3/v3_test_records'
    fileDataModel = FileDataModel(users, items, train, test)
    alg = NMF(fileDataModel)
    alg.train()
    print alg.recommend(81)










