__author__ = 'Peter'
import threading
from DataModel.FileDataModel import *
from Eval.Evaluation import Eval
from utils.Config import Config
from RecommendationAlg.LFM import LFM
from RecommendationAlg.TopN import TopN
from RecommendationAlg.NMF import NMF
from RecommendationAlg.UserCF import UserCF
from utils.Similarity import *
import pandas as pd
from RecommendationAlg.AlgFactory import AlgFactory
import Queue
import datetime


class MulThreading(threading.Thread):
    def __init__(self, threadName, threadTarget, threadParameters, threadingLock, sharedVar):
        threading.Thread.__init__(self)
        self.threadName = threadName
        self.target = threadTarget
        self.parameters = threadParameters
        self.var = sharedVar
        self.lock = threadingLock
    def run(self):
        s1 = datetime.datetime.now()
        alg = self.target(self.parameters)
        alg.train()
        eval = Eval(alg)
        self.lock.acquire()
        print self.parameters[0] + str(eval.evalAll())
        self.lock.release()
        print self.parameters[0] + 'time consuming:' + str(datetime.datetime.now()-s1)



class ParallelModel():
    def __init__(self, taskQueue, taskParameters, sharedVar):
        self.queue =taskQueue
        self.parameters = taskParameters
        self.threads = []
        self.threadingLock = threading.Lock()
        self.sharedVar = sharedVar
        assert (taskQueue.qsize() == taskParameters.qsize())
    def run(self):
        while not self.queue.empty():
            target = self.queue.get()
            para = self.parameters.get()
            t = MulThreading('alg', target, para, self.threadingLock, self.sharedVar)
            self.threads.append(t)
            t.start()
        for th in self.threads:
            th.join()



