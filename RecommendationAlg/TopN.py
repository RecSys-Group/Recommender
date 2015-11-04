import numpy as np
import pandas as pd
import os
from DataModel.FileDataModel import FileDataModel

class TopN:

    def __init__(self, dataModel):
        print "TopN begin"
        self.dataModel = dataModel
        self.popfile = dataModel.conf.get('pop')

    def gen_items_popular(self, hasTimes=False):
        print 'gen_popular!'
        if os.path.exists(self.popfile):
            print 'items_popularity has been generated!'
            self.popItems = pd.read_csv(self.popfile)
        else:
            itempopular = np.zeros(self.dataModel.getItemsNum())
            train = pd.read_csv(self.dataModel.conf.get('train'))
            for row in train.values:
                print row
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
    users = 'D:/Desktop/recommender/Data/v3/v3_users'
    items = 'D:/Desktop/recommender/Data/v3/v3_items'
    train = 'D:/Desktop/recommender/Data/v3/v3_train_records'
    test = 'D:/Desktop/recommender/Data/v3/v3_test_records'
    popfile = 'D:/Desktop/recommender/Data/v3/item_popularity'
    fileDataModel = FileDataModel(users, items, train, test)
    alg = TopN(fileDataModel, popfile)
    alg.train()
    print alg.recommend()


