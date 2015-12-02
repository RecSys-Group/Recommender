import numpy as np
import pandas as pd
import scipy.sparse as spr
import os
from BaseDataModel import BaseDataModel

class MemeryDataModel(BaseDataModel):

    def __init__(self, samples, targets, isRating=True):
        super(MemeryDataModel, self).__init__()
        self.samples = samples
        self.targets = targets
        self.__users = list(set(np.array(samples)[:,0]))
        self.__items = list(set(np.array(samples)[:,1]))
        ratingMatrix = np.zeros((len(self.__users),len(self.__items)))

        l = len(samples)
        for i in range(l):
            user = int(float(samples[i][0]))
            uid = self.getUidByUser(user)
            item = int(float(samples[i][1]))
            iid = self.getIidByItem(item)
            rating = float(targets[i]) if isRating else 1
            ratingMatrix[uid][iid] = rating
        self.__data = spr.csr_matrix(ratingMatrix)

    def getUidByUser(self, user):
        if user not in self.__users:
            return -1
        else:
            uid = np.argwhere(np.array(self.__users) == user)[0][0]
        return uid

    def getIidByItem(self, item):
        if item not in self.__items:
            return -1
        else:
            iid = np.argwhere(np.array(self.__items) == item)[0][0]
        return iid

    def getUserByUid(self, uid):
        return self.__users[uid]

    def getItemByIid(self, iid):
        return self.__items[iid]

    def getUsersNum(self):
        return len(self.__users)

    def getItemsNum(self):
        return len(self.__items)

    def getItemIDsFromUid(self, uid):
        return self.__data[uid].indices

    def getItemIDsForEachUser(self):
        itemIDs = []
        for uid in range(len(self.__users)):
            itemIDs.append(self.__data[uid].indices)
        return itemIDs

    def getRating(self, userID, itemID):
        return self.__data[userID, itemID]

    def getData(self):
        return self.__data

    def getLineData(self):
        lineData = [[self.samples[i][0], self.samples[i][1], self.targets[i]] for i in range(len(self.samples))]
        return lineData




if __name__ == "__main__":
    samples = [[0,0],[1,3],[0,1],[1,2]]
    targets = [1,2,3,4]
    p = MemeryDataModel(samples, targets, isRating=True)
    print p.getItemIDsFromUid(0)
