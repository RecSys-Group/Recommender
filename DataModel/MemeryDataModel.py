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
        self.__users = list(set(np.array(self.samples)[:,0]))
        self.__items = list(set(np.array(self.samples)[:,1]))
        self.ratingMatrix = np.zeros((len(self.__users),len(self.__items)))

        l = len(samples)
        for i in range(l):
            user = int(float(samples[i][0]))
            uid = self.getUidByUser(user)
            item = int(float(samples[i][1]))
            iid = self.getIidByItem(item)
            rating = float(targets[i]) if isRating else 1
            self.ratingMatrix[uid][iid] = rating
        self.__data = spr.csr_matrix(self.ratingMatrix)
        self.__data_T = spr.csr_matrix(self.ratingMatrix.transpose())

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

    def getUserIDsFromIid(self, iid):
        return self.__data_T[iid].indices

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


class MemeryDataModelPreprocess():

    def __init__(self):
        print 'Begin Preprocess!'

    def getLineDataByRemoveDuplicate(self, samples, targets):
        result = []
        lineData = [[samples[i][0], samples[i][1], targets[i]] for i in range(len(samples))]
        for line in lineData:
            print line
            if len(result) == 0:
                result.append(line)
            else:
                userItemPairs = [list(i) for i in np.array(result)[:,:2]]
                if line[:2] not in userItemPairs:
                    result.append(line)
        new_samples = [list(i) for i in np.array(result)[:,:2]]
        new_targets = list(np.array(result)[:,2])
        return new_samples, new_targets

    def getItemPurchasedNumDistribute(self, samples, targets):
        result = dict()
        lineData = [[samples[i][0], samples[i][1], targets[i]] for i in range(len(samples))]
        for line in lineData:
            item = line[1]
            if result.has_key(item):
                result[item] += 1
            else:
                result[item] = 1
        return result

    def getUserPurchaseNumDistribute(self, samples, targets):
        result = dict()
        lineData = [[samples[i][0], samples[i][1], targets[i]] for i in range(len(samples))]
        for line in lineData:
            user = line[0]
            if result.has_key(user):
                result[user] += 1
            else:
                result[user] = 1
        return result

    def hasLowFrequencyUser(self, samples, targets, n=5):
        UserF = self.getUserPurchaseNumDistribute(samples, targets)
        f = UserF.values()
        for i in f:
            if i < n:
                return 1
        return 0

    def hasLowFrequencyItem(self, samples, targets, n=5):
        ItemF = self.getItemPurchasedNumDistribute(samples, targets)
        f = ItemF.values()
        for i in f:
            if i < n:
                return 1
        return 0

    def removeLowFrequencyUser(self, samples, targets, n=5):
        print 'low'
        UserF = self.getUserPurchaseNumDistribute(samples, targets)
        result = []
        lineData = [[samples[i][0], samples[i][1], targets[i]] for i in range(len(samples))]
        for line in lineData:
            user = line[0]
            if UserF[user] > n:
                result.append(line)
        new_samples = [list(i) for i in np.array(result)[:,:2]]
        new_targets = list(np.array(result)[:,2])
        return new_samples, new_targets

    def removeLowFrequencyItem(self, samples, targets, n=5):
        print 'low'
        ItemF = self.getItemPurchasedNumDistribute(samples, targets)
        result = []
        lineData = [[samples[i][0], samples[i][1], targets[i]] for i in range(len(samples))]
        for line in lineData:
            item = line[1]
            if ItemF[item] > n:
                result.append(line)
        new_samples = [list(i) for i in np.array(result)[:,:2]]
        new_targets = list(np.array(result)[:,2])
        return new_samples, new_targets



if __name__  ==  "__main__":
    data = pd.read_csv('../Data/bbg/transaction.csv')
    samples = [[int(i[0]), int(i[1])] for i in data.values[:, 0:2]]
    targets = [1 for i in samples]
    p = MemeryDataModelPreprocess()
    print p.hasLowFrequencyUser(samples, targets, 5)
