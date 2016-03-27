__author__ = 'Jerry'

from RecommendationAlg.LFM import LFM
from RecommendationAlg.TopN import TopN
from RecommendationAlg.NMF import NMF
from RecommendationAlg.UserCF import UserCF
from RecommendationAlg.ItemCF import ItemCF
from RecommendationAlg.BPR import BPR
from RecommendationAlg.VSRank import VSRank
from RecommendationAlg.VSRankPlus import VSRankPlus

class AlgFactory:

    @staticmethod
    def create(name):
        if name == 'TopN':
            return TopN()
        elif name == 'LFM':
            return LFM()
        elif name == 'NMF':
            return NMF()
        elif name == 'UserCF':
            return UserCF()
        elif name == 'ItemCF':
            return ItemCF()
        elif name == 'BPR':
            return BPR()
        elif name == 'VSRank':
            return VSRank()
        elif name == 'VSRankPlus':
            return VSRankPlus()