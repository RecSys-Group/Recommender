import numpy as np
import pandas as pd
import os
from DataModel.FileDataModel import *
from BaseAlg import BaseAlg

class TopN(BaseAlg):

    def __init__(self, dataModel, paras):
        print "TopN begin"
        super(TopN, self).__init__(dataModel, 'TopN')
        self.popfile = self.dataModel.getPopFile()

    def gen_items_popular(self, hasTimes=False):
        print 'gen_popular!'
        if os.path.exists(self.popfile):
            print 'items_popularity has been generated!'
            self.popItems = pd.read_csv(self.popfile)
        else:
            itempopular = np.zeros(self.dataModel.getItemsNum())
            for row in self.dataModel.train.values:
                # print row
                iid = int(float(row[2]))
                times = int(float(row[3])) if hasTimes else 1
                itempopular[iid] += times
            self.popItems = pd.DataFrame(itempopular)
            self.popItems.to_csv(self.popfile)

    def recommend(self, userID=None, N=5):
        return self.topN[:N]

    def train(self):
        self.gen_items_popular()
        self.topN = np.argsort(np.array(self.popItems.iloc[:, 1]))[-1:-self.dataModel.getItemsNum()-1:-1]

    def recommendAllUserInTest(self, N=5):
        result = self.recommend(N)
        return [result] * self.dataModel.getUsersNumInTest()

if __name__ == '__main__':
    pass


