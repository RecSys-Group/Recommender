__author__ = 'Jerry'
import pandas as pd
import numpy as np
import Queue

from multiprocessing import Process, Manager, freeze_support, Lock
import datetime

from sklearn.base import BaseEstimator
from sklearn import cross_validation
from sklearn import metrics
from sklearn import grid_search

from DataModel.FileDataModel import *
from utils.Config import Config
from utils.Similarity import *
import utils.MulThreading as Mul
from RecommendationAlg.AlgFactory import AlgFactory
from RecommendationAlg.TopN import TopN
from sklearn.cross_validation import StratifiedKFold

class App:
    def __init__(self):
        self.config = Config()
        self.config.from_ini('../Application/conf')
        self.data = pd.read_csv('../Data/ml-1m/newtransaction.csv')
        # self.data = pd.read_csv('../Data/bbg/transaction.csv')
        self.samples = [[int(i[0]), int(i[1])] for i in self.data.values[:,0:2]]
        self.labels = [int(i[0]) for i in self.data.values[:,0:2]]
        self.targets = [int(i[2]) for i in self.data.values]
        # self.targets = [1 for i in self.samples]
        self.Lock = Lock()
        m = Manager()
        self.result = m.list()

    def mulProcess(self,result,processParameters):
        algName = processParameters[0]
        parameters = processParameters[1]
        alg = AlgFactory.create(algName)

        rec_cv =  StratifiedKFold(self.labels, 2)
        clf = grid_search.GridSearchCV(alg, parameters, cv=rec_cv)
        clf.fit(self.samples, self.targets)
        print(clf.best_estimator_)
        print(clf.grid_scores_)
        self.Lock.acquire()
        result.append(algName)
        result.append([clf.best_estimator_,clf.best_score_])
        result.append(clf.grid_scores_)
        self.Lock.release()

    def fit(self):
        '''
        for algName in self.config.algList:
            for para in self.config.paras[algName].iter():
                s1 = datetime.datetime.now()
                threadParameters = [algName, para]
                self.mulThread(threadParameters)
                print algName + 'time consuming:' + str(datetime.datetime.now()-s1)
                raw_input()
        '''
        print 'Fitting begin'
        process_que = []
        s = datetime.datetime.now()
        for algName in self.config.algList:
            for para in self.config.paras[algName].iter():
                Parameters = [algName, para]
                process = Process(target=self.mulProcess, args=(self.result,Parameters))
                process_que.append(process)
        for t in process_que:
            t.start()
        for t in process_que:
            t.join()
        print 'time consuming:' + str(datetime.datetime.now()-s)
        t = pd.DataFrame(list(self.result))
        t.to_csv('all_result')

    def best_alg(self):
        print 'Selecting best Alg'
        self.fit()
        scores = np.array(self.result[1:3])[:,1]
        print scores
        index = np.argwhere(scores == scores.max())[0][0]
        best_method = self.result[3*index]
        print 'best is: ' + best_method
        return best_method

    def recommend(self, uids):
        print 'Recommending'
        bestMethod = self.best_alg()
        alg = AlgFactory.create(bestMethod)
        alg.fit(self.samples, self.targets)
        recommendList = []
        for u in uids:
            recommendList.append(alg.recommend(u))
        return recommendList


if __name__ == '__main__':

    app = App()
    '''
    uids = [1]
    print app.recommend(uids)
    '''
    # result = app.recommend(range(1729))
    # t = pd.DataFrame(result)
    # t.to_csv('rec_list')
    app.fit()
