__author__ = 'Jerry'

import math


class Similarity:

    def __init__(self, simiName):
        self.simiName = simiName

    @staticmethod
    def cosine(v1, v2):
        return len(set(v1) & set(v2)) / math.sqrt(len(v1) * len(v2) * 1.0)

    @staticmethod
    def jaccard(v1, v2):
        s1 = set(v1)
        s2 = set(v2)
        return len(s1.intersection(s2)) * 1.0 / len(s1.union(s2))

    def compute(self, v1, v2):
        return SIMILARITY.get(self.simiName)(v1, v2)

SIMILARITY = {'COSINE': Similarity.cosine, 'JACCARD': Similarity.jaccard}

if __name__ == "__main__":
    print Similarity.cosine([1,2], [2, 3])
    print Similarity.jaccard([1,2], [2, 3])
    s = Similarity('COSINE')
    print s.compute([1, 2], [7, 2])