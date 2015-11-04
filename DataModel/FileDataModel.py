import numpy as np
import pandas as pd
import scipy.sparse as spr
from Config.Config import Config

class FileDataModel():

    def __init__(self, config, isRating=True):
        self.conf = config
        self.__users = pd.read_csv(self.conf.get('users'))
        self.__items = pd.read_csv(self.conf.get('items'))

        train = pd.read_csv(self.conf.get('train'))
        test = pd.read_csv(self.conf.get('test'))
        ratingMatrixOfTrain = np.zeros((self.getUsersNum(), self.getItemsNum()))
        self.__userIDsInTrain = set()
        for row in train.values:
            uid = int(float(row[1]))
            iid = int(float(row[2]))
            rating = float(row[3]) if isRating else 1
            ratingMatrixOfTrain[uid][iid] = rating
            self.__userIDsInTrain.add(uid)
        self.__train = spr.csr_matrix(ratingMatrixOfTrain)

        ratingMatrixOfTest = np.zeros((self.getUsersNum(), self.getItemsNum()))
        self.__userIDsInTest = set()
        for row in test.values:
            uid = int(float(row[1]))
            iid = int(float(row[2]))
            rating = float(row[3]) if isRating else 1
            ratingMatrixOfTest[uid][iid] = rating
            self.__userIDsInTest.add(uid)
        self.__test = spr.csr_matrix(ratingMatrixOfTest)

        if self.__userIDsInTrain != self.getUsersNum():
            print 'Error!!!!!!!!!!!!!\n some users do not appear in train'

    def getUserIDs(self):
        return self.__users.ix[:, 0]

    def getItemIDs(self):
        return self.__items.ix[:, 0]

    def getUserIDsInTrain(self):
        return self.__userIDsInTrain

    def getUserIDsInTest(self):
        return self.__userIDsInTest

    def getUserFromID(self, userID):
        return self.__users.ix[userID, 1]

    def getItemFromID(self, itemID):
        return self.__items.ix[itemID, 1]

    def getItemsNum(self):
        return self.__items.shape[0]

    def getUsersNum(self):
        return self.__users.shape[0]

    def getUsersNumInTrain(self):
        return len(self.__userIDsInTrain)

    def getUsersNumInTest(self):
        return len(self.__userIDsInTest)

    def getItemIDsFromUserInTest(self, userID):
        return self.__Test[userID].indices

    def getItemIDsFromUserInTrain(self, userID):
        return self.__train[userID].indices

    def getItemIDsForEachUserInTrain(self):
        itemIDs = []
        for uid in self.__userIDsInTrain:
            itemIDs.append(self.__train[uid].indices)
        return itemIDs

    def getItemIDsForEachUserInTest(self):
        itemIDs = []
        for uid in self.__userIDsInTest:
            itemIDs.append(self.__test[uid].indices)
        return itemIDs

    def getRatingInTrain(self, userID, itemID):
        return self.__train[userID, itemID]

    def getTrain(self):
        return self.__train

if __name__ == "__main__":
    conf = Config('D:\\Desktop\\recommender\\Config\\configOfYelp')
    fileDataModel = FileDataModel(conf)
    # print fileDataModel.getUserIDs()
    # print fileDataModel.getItemIDs()
    print fileDataModel.getUserFromID(0)
    print fileDataModel.getItemFromID(0)
    print fileDataModel.getItemsNum()
    print fileDataModel.getUsersNum()
    print fileDataModel.getItemIDsFromUserInTrain(81)
    # print fileDataModel.getItemIDsForEachUserInTrain()
    print fileDataModel.getRatingInTrain(81, 464)