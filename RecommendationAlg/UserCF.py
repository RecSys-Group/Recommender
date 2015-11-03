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
        itemListForEachUser = self.dataModel.getItemIDsForEachUserInTrain()
        for i in range(usersNum):
            for j in range(i+1, usersNum):
                s = self.similarity.compute(itemListForEachUser[i], itemListForEachUser[j])
                self.simiMatrix[i][j] = self.simiMatrix[j][i] = s

    def neighborhoodAfterFiltering(self, userID, itemID):
        neighborsAfterFiltering = []
        neighbors = np.argsort(np.array(self.simiMatrix[userID]))[::-1]
        for uid in neighbors:
            if len(neighborsAfterFiltering) > self.neighborNum:
                break
            if itemID in self.dataModel.getItemIDsFromUserInTrain(uid):
                neighborsAfterFiltering.append(uid)
        return neighborsAfterFiltering

    def predict(self, userID, itemID):
        rating = 0.0
        neighbors = np.argsort(np.array(self.simiMatrix[userID]))[-1:-self.neighborNum-1:-1]
        for uid in neighbors:
            if itemID in self.dataModel.getItemIDsFromUserInTrain(uid):
                rating += self.simiMatrix[userID][uid] * self.dataModel.ratingMatrixOfTrain[uid][itemID]
        return rating

    def recommend(self, userID, N=5):
        interactedItems = self.dataModel.getItemIDsFromUserInTrain(userID)
        ratings = dict()
        neighbors = np.argsort(np.array(self.simiMatrix[userID]))[-1:-self.neighborNum-1:-1]
        for uid in neighbors:
            for iid in self.dataModel.getItemIDsFromUserInTrain(uid):
                if iid in interactedItems:
                    continue
                r = ratings.get(iid, 0)
                ratings[iid] = r + self.simiMatrix[userID][uid] * self.dataModel.ratingMatrixOfTrain[uid][iid]
        return [x for (x, y) in sorted(ratings.items(), lambda a, b: cmp(a[1], b[1]), reverse=True)[:N]]

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