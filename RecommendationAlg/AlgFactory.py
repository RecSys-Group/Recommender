__author__ = 'Jerry'

from RecommendationAlg.LFM import LFM
from RecommendationAlg.SVD import SVD
from RecommendationAlg.TopN import TopN
from RecommendationAlg.NMF import NMF
from RecommendationAlg.UserCF import UserCF
from RecommendationAlg.ItemCF import ItemCF

class AlgFactory:

    @staticmethod
    def create(name, dataModel, paras):
        if name == 'TopN':
            return TopN(dataModel, paras)
        elif name == 'SVD':
            return SVD(dataModel, paras)
        elif name == 'LFM':
            return LFM(dataModel, paras)
        elif name == 'NMF':
            return NMF(dataModel, paras)
        elif name == 'UserCF':
            return UserCF(dataModel, paras)
        elif name == 'ItemCF':
            return ItemCF(dataModel, paras)
