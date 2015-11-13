__author__ = 'Jerry'

import numpy as np
from scipy.sparse.linalg import svds
import scipy.sparse as spr
from BaseAlg import BaseAlg

class SVD(BaseAlg):
    def __init__(self, dataModel, paras):
        print "SVD begin"
        super(SVD, self).__init__(dataModel, 'SVD')
        self.dataModel = dataModel
        self.feature_dims = int(float(paras['feature_dims']))

    def predict(self, uid, iid):
        return self.result[uid][iid]

    def recommend(self, uid, N=5):
        # rated = self.dataModel.getItemIDsFromUserInTrain(uid)
        # self.result[uid][rated] = 0.0
        topN = np.argsort(np.array(self.result[uid]))[-1:-N-1:-1]
        return topN

    def recommendAllUserInTest(self, N=5):
        recList = []
        for uid in self.dataModel.getUserIDsInTest():
            recList.append(self.recommend(uid, N))
        return recList

    def vector_to_diag(self, vector):
        if (isinstance(vector, np.ndarray) and vector.ndim == 1) or isinstance(vector, list):
            length = len(vector)
            diag = np.zeros((length, length))
            np.fill_diagonal(diag, vector)
            return diag
        return None

    def train(self):
        print 'SVD training'
        U, S, VT = svds(self.dataModel.getTrain(), self.feature_dims, maxiter=200)
        S = self.vector_to_diag(S)
        # print U.shape
        # print S.shape
        # print VT.shape
        self.result = np.dot(np.dot(U, S), VT)

if __name__ == '__main__':
    i = np.eye(2)
    j = np.array([[1, 2], [3, 4]])
    l = spr.csr_matrix(j)
    a = [1, 2]
    b = 2
    print np.dot(a, b)










