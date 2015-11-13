from __future__ import division
from numpy import *
import pandas as pd
from DataModel import FileDataModel
from RecommendationAlg import TopN

class Eval:
    def __init__(self, recAlg):
        self.recommend_list = recAlg.recommendAllUserInTest()
        self.purchased_list = recAlg.dataModel.getItemIDsForEachUserInTest()

    def F1_score_Hit_ratio(self):
        user_number = len(self.recommend_list)
        correct = []
        co_length = []
        re_length = []
        pu_length = []
        p = []
        r = []
        f = []
        hit_number = 0
        for i in range(user_number):
            temp = []
            for j in self.recommend_list[i]:
                if j in self.purchased_list[i]:
                    temp.append(j)
            if len(temp):
                hit_number = hit_number + 1
            co_length.append(len(temp))
            re_length.append(len(self.recommend_list[i]))
            pu_length.append(len(self.purchased_list[i]))
            correct.append(temp)

        for i in range(user_number):
            p_t = co_length[i] / re_length[i]
            r_t = co_length[i] / pu_length[i]
            p.append(p_t)
            r.append(r_t)
            if p_t != 0 or r_t != 0:
                f.append(2*p_t*r_t / (p_t+r_t))
            else:
                f.append(0)

        self.hit_ratio = hit_number / user_number
        # print 'Precisions are :' + str(p)
        # print len(p)
        # print 'Recalls are :' + str(r)
        self.f1 = array(f).mean()
        return self.f1, self.hit_ratio

    def NDGG_k(self):
        user_number = len(self.recommend_list)
        u_ndgg = []
        for i in range(user_number):
            temp = 0
            Z_u = 0
            for j in range(len(self.recommend_list[i])):
                Z_u = Z_u + 1 / log2(j + 2)
                if self.recommend_list[i][j] in self.purchased_list[i]:
                    temp = temp + 1 / log2(j + 2)
            temp = temp / Z_u
            u_ndgg.append(temp)
        self.NDCG = array(u_ndgg).mean()
        return self.NDCG

    def evalAll(self):
        self.F1_score_Hit_ratio()
        self.NDGG_k()
        return {'F1': self.f1, 'NDCG': self.NDCG}

if __name__ == '__main__':
    users = '../data/v3/v3_users'
    items = '../data/v3/v3_items'
    trainfile = '../data/v3/v3_train_records'
    testfile = '../data/v3/v3_test_records'
    popfile = '../data/v3/popfile'
    fileDataModel = FileDataModel.FileDataModel(users, items, trainfile, testfile)
    top = TopN.TopN(fileDataModel, 10, popfile)
    eval = Eval(top)
    eval.F1_score_Hit_ratio()
    eval.NDGG_k()

