__author__ = 'Jerry'

import numpy as np
from DataModel.FileDataModel import FileDataModel
from Similarity.Similarity import Similarity

class UserCF:

    def __init__(self, dataModel, similarity, neighborNum=5):
        self.dataModel = dataModel
        self.similarity = similarity
        self.neighborNum = neighborNum

    def train(self):
        usersNum = self.dataModel.getUsersNum()
        self.simiMatrix = np.zeros((usersNum, usersNum))
        for i in range(usersNum):
            for j in range(i+1, usersNum):
                s = self.similarity.compute(self.dataModel.getItemIDsFromUserInTrain(i), self.dataModel.getItemIDsFromUserInTrain(j))
                self.simiMatrix[i][j] = self.simiMatrix[j][i] = s

    def neighborhood(self, userID):
        neighbors = np.argsort(np.array(self.simiMatrix[userID]))[-1:-self.neighborNum-1:-1]
        return neighbors

    def predict(self, userID, itemID):
        rating = 0.0
        for uid in self.neighborhood(userID):
            if itemID in self.dataModel.getItemIDsFromUserInTrain(uid):
                rating += self.simiMatrix[userID][uid] * self.dataModel.getRatingInTrain(uid, itemID)
        return rating

    def recommend(self, userID, N=5):
        interactedItems = self.dataModel.getItemIDsFromUserInTrain(userID)
        ratings = dict()
        for uid in self.neighborhood(userID):
            for iid in self.dataModel.getItemIDsFromUserInTrain(uid):
                if iid in interactedItems:
                    continue
                r = ratings.get(iid, 0)
                ratings[iid] = r + self.simiMatrix[userID][uid] * self.dataModel.getRatingInTrain(uid, iid)
        return [x for (x, y) in sorted(ratings.items(), lambda a, b: cmp(a[1], b[1]), reverse=True)[:N]]

    def recommendAllUserInTest(self, N=5):
        recList = []
        for uid in self.dataModel.getUserIDsInTest():
            recList.append(self.recommend(uid, N))
        return recList
if __name__ == '__main__':
    users = 'D:/Desktop/recommender/Data/v3/v3_users'
    items = 'D:/Desktop/recommender/Data/v3/v3_items'
    train = 'D:/Desktop/recommender/Data/v3/v3_train_records'
    test = 'D:/Desktop/recommender/Data/v3/v3_test_records'
    fileDataModel = FileDataModel(users, items, train, test)
    sim = Similarity('COSINE')
    alg = UserCF(fileDataModel, sim)
    alg.train()
    print fileDataModel.getItemIDsFromUserInTrain(243)
    print alg.recommend(81)
    print alg.predict(81, 169)
    print alg.predict(81, 364)