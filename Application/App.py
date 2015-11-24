__author__ = 'Jerry'

import pandas as pd
import Queue
import threading
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
result = []

class App:

    def __init__(self):
        print 'begin'
        self.config = Config()
        self.config.from_ini('../Application/conf')
        self.dataModel = FileDataModelInMatrix(self.config.data)
        self.data = pd.read_csv('../Data/tinytest/format.csv')
        self.samples = [[int(i[0]), int(i[1])] for i in self.data.values[:,0:2]]
        self.targets = [int(i) for i in self.data.values[:,3]]
        self.threadLock = threading.Lock()

    def mulThread(self,threadParameters):
        algName = threadParameters[0]
        parameters = threadParameters[1]
        alg = AlgFactory.create(algName)
        clf = grid_search.GridSearchCV(alg, parameters,cv=5)
        clf.fit(self.samples, self.targets)
        print(clf.best_estimator_)
        print(clf.grid_scores_)
        self.threadLock.acquire()
        result.append(algName)
        result.append(clf.best_estimator_)
        result.append(clf.grid_scores_)
        self.threadLock.release()

    def start(self):
        '''
        for algName in self.config.algList:
            for para in self.config.paras[algName].iter():
                s1 = datetime.datetime.now()
                threadParameters = [algName, para]
                self.mulThread(threadParameters)
                print algName + 'time consuming:' + str(datetime.datetime.now()-s1)
                raw_input()
        '''

        print 'mul'
        thread_que = []
        s = datetime.datetime.now()
        for algName in self.config.algList:
            for para in self.config.paras[algName].iter():
                threadParameters = [algName, para]
                thread = threading.Thread(target=self.mulThread, args=(threadParameters,))
                thread_que.append(thread)
        for t in thread_que:
            t.start()
        for t in thread_que:
            t.join()
        print 'time consuming:' + str(datetime.datetime.now()-s)
        t = pd.DataFrame(result)
        t.to_csv('all_result')


if __name__ == '__main__':
    app = App()
    app.start()
