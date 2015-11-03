import numpy as np
import pandas as pd
import scipy.sparse as spr

class FileDataModel():

    def __init__(self, config, isRating=True):
        # TODO:sparse matrix
        self.__users = pd.read_csv(config.get('users'))
        self.__items = pd.read_csv(config.get('items'))
        train = pd.read_csv(config.get('train'))
        test = pd.read_csv(config.get('test'))
        self.ratingMatrixOfTrain = np.zeros((self.getUsersNum(), self.getItemsNum()))
        for row in train.values:
            uid = int(float(row[1]))
            iid = int(float(row[2]))
            rating = float(row[3]) if isRating else 1
            self.ratingMatrixOfTrain[uid][iid] = rating
        
        self.ratingMatrixOfTest = np.zeros((self.getUsersNum(), self.getItemsNum()))
        for row in test.values:
            uid = int(float(row[1]))
            iid = int(float(row[2]))
            rating = float(row[3]) if isRating else 1
            self.ratingMatrixOfTest[uid][iid] = rating

    def getUserIDs(self):
        return self.users.ix[:, 0]

    def getItemIDs(self):
        return self.items.ix[:, 0]

    def getUserFromID(self, userID):
        return self.users.ix[userID, 1]

    def getItemFromID(self, itemID):
        return self.items.ix[itemID, 1]

    def getItemsNum(self):
        return self.__items.shape[0]

    def getUsersNum(self):
        return self.__users.shape[0]

    def getItemIDsFromUserInTest(self, userID):
        return self.ratingMatrixOfTest[userID]

    def getItemIDsFromUserInTrain(self, userID):
        itemIDs = set()
        for row in self.train.values:
            uid = int(float(row[1]))
            iid = int(float(row[2]))
            if uid == userID:
                itemIDs.add(iid)
        return itemIDs

    def getItemIDsForEachUserInTrain(self):
        itemsList = [[] for i in range(self.getUsersNum())]
        for row in self.train.values:
            uid = int(float(row[1]))
            iid = int(float(row[2]))
            itemsList[uid].append(iid)
        return itemsList

if __name__ == "__main__":
    users = 'D:/Desktop/recommender/Data/v3/v3_users'
    items = 'D:/Desktop/recommender/Data/v3/v3_items'
    train = 'D:/Desktop/recommender/Data/v3/v3_train_records'
    test = 'D:/Desktop/recommender/Data/v3/v3_test_records'
    fileDataModel = FileDataModel(users, items, train, test)
    print fileDataModel.getUserIDs()
    print fileDataModel.getItemIDs()
    print fileDataModel.getUserFromID(0)
    print fileDataModel.getItemFromID(0)
    print fileDataModel.getItemsNum()
    print fileDataModel.getUsersNum()
    print fileDataModel.getItemIDsFromUserInTrain(81)
    print fileDataModel.getItemIDsForEachUserInTrain()
    print fileDataModel.ratingMatrixOfTrain[81][324]