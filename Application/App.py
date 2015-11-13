__author__ = 'Jerry'

from DataModel.FileDataModel import *
from Eval.Evaluation import Eval
from utils.Config import Config
from RecommendationAlg.SVD import SVD
from RecommendationAlg.LFM import LFM
from RecommendationAlg.TopN import TopN
from RecommendationAlg.NMF import NMF
from RecommendationAlg.UserCF import UserCF
from utils.Similarity import *
import pandas as pd
from RecommendationAlg.AlgFactory import AlgFactory

class App:

    def __init__(self):
        self.config = Config()
        self.config.from_ini('../Application/conf')
        self.dataModel = FileDataModelInMatrix(self.config.data)
        # fileDataModelInRow = FileDataModelInRow(config)

    def start(self):
        results = []
        indexs = []
        for algName in self.config.algList:
            for para in self.config.paras[algName].iter():
                alg = AlgFactory.create(algName, self.dataModel, para)
                alg.train()
                eval = Eval(alg)
                results.append(eval.evalAll())
                indexs.append(algName+str(para))
        self.results = pd.DataFrame(results, index=indexs)
        print self.results

if __name__ == '__main__':
    app = App()
    app.start()
