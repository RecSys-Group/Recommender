__author__ = 'Jerry'

from RecommendationAlg.LFM import LFM
from RecommendationAlg.TopN import TopN
from RecommendationAlg.NMF import NMF
from RecommendationAlg.UserCF import UserCF
from RecommendationAlg.ItemCF import ItemCF

class AlgFactory:

    @staticmethod
    def create(name):
        if name == 'TopN':
            return TopN()
        elif name == 'SVD':
            return SVD()
        elif name == 'LFM':
            return LFM()
        elif name == 'NMF':
            return NMF()
        elif name == 'UserCF':
            return UserCF()
        elif name == 'ItemCF':
            return ItemCF()
