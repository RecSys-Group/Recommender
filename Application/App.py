__author__ = 'Jerry'

from RecommendationAlg.TopN import TopN
from DataModel.FileDataModel import FileDataModel
from Eval.BasicMetric import Eval
from RecommendationAlg.NMF import NMF
from RecommendationAlg.UserCF import UserCF
from Similarity.Similarity import Similarity
import pandas as pd
from Config.Config import Config

class App:

    def start(self):
        config = Config('..\Config\config')

        fileDataModel = FileDataModel(config)

        # top = TopN(fileDataModel, popfile)
        # top.train()
        # eval = Eval(top)
        # eval.F1_score_Hit_ratio()
        # eval.NDGG_k()

        nmf = NMF(fileDataModel)
        nmf.train()
        eval = Eval(nmf)
        eval.F1_score_Hit_ratio()
        eval.NDGG_k()

        # sim = Similarity('COSINE')
        # usercf = UserCF(fileDataModel, sim)
        # usercf.train()
        # eval = Eval(usercf)
        # eval.F1_score_Hit_ratio()
        # eval.NDGG_k()

if __name__ == '__main__':
    app = App()
    app.start()
    print app.nmf.recommend(81)