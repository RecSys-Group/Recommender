import numpy as np
from math import exp
import scipy as sp
from scipy.sparse import coo_matrix
import scipy.sparse as spr
from numpy.random import random
import math
import random
from DataModel.FileDataModel import *
from DataModel.MemeryDataModel import *
from sklearn.base import BaseEstimator
from sklearn import cross_validation
from sklearn import metrics
from sklearn import grid_search
from Eval.Evaluation import *

class BPR(BaseEstimator):
    def __init__(self, n=5, factors=50, learning_rate=0.001, bias_regularization=0.001, user_regularization=0.001,
                 positive_item_regularization=0.001, negative_item_regularization=0.001,iter = 5):
        """initialise BPR matrix factorization model
        D: number of factors
        """
        self.n = n
        self.factors = factors
        self.learning_rate = learning_rate
        self.bias_regularization = bias_regularization
        self.user_regularization = user_regularization
        self.positive_item_regularization = positive_item_regularization
        self.negative_item_regularization = negative_item_regularization
        self.iter = iter

    def predict(self, testSamples):
        recList = []
        for user_item in testSamples:
            uid = self.dataModel.getUidByUser(user_item[0])
            recList.append(self.recommend(uid))
        return recList

    def update_factors(self, u, i, j, update_u=True, update_i=True):
        """apply SGD update"""
        update_j = True
        x = self.item_bias[i] - self.item_bias[j] \
            + np.dot(self.user_factors[u],self.item_factors[i]-self.item_factors[j])
        z = 1.0/(1.0+exp(x))
        # update bias terms
        if update_i:
            d = z - self.bias_regularization * self.item_bias[i]
            self.item_bias[i] += self.learning_rate * d
        if update_j:
            d = -z - self.bias_regularization * self.item_bias[j]
            self.item_bias[j] += self.learning_rate * d

        if update_u:
            d = (self.item_factors[i]-self.item_factors[j])*z - self.user_regularization*self.user_factors[u]
            self.user_factors[u,:] += self.learning_rate*d
        if update_i:
            d = self.user_factors[u]*z - self.positive_item_regularization*self.item_factors[i]
            self.item_factors[i,:] += self.learning_rate*d
        if update_j:
            d = -self.user_factors[u]*z - self.negative_item_regularization*self.item_factors[j]
            self.item_factors[j] += self.learning_rate*d
    def loss(self):
        ranking_loss = 0
        for u,i,j in self.loss_samples:
            x = self.predict_single(u,i) - self.predict_single(u,j)
            ranking_loss += math.log(1.0+exp(-x))

        complexity = 0
        for u,i,j in self.loss_samples:
            complexity += self.user_regularization * np.dot(self.user_factors[u],self.user_factors[u])
            complexity += self.positive_item_regularization * np.dot(self.item_factors[i],self.item_factors[i])
            complexity += self.negative_item_regularization * np.dot(self.item_factors[j],self.item_factors[j])
            complexity += self.bias_regularization * self.item_bias[i]**2
            complexity += self.bias_regularization * self.item_bias[j]**2
        return ranking_loss + 0.5*complexity
    def fit(self, trainSamples, trainTargets):
        self.dataModel = MemeryDataModel(trainSamples, trainTargets)
        temp = math.sqrt(self.factors)
        self.item_bias = np.zeros(self.dataModel.getItemsNum())
        self.user_factors = np.array([[(0.1 * random.random() / temp) for j in range(self.factors)] for i in range(self.dataModel.getUsersNum())])
        self.item_factors = np.array([[(0.1 * random.random() / temp) for j in range(self.factors)] for i in range(self.dataModel.getItemsNum())])
        '''
        user_file = 'pu'
        item_file = 'qi'
        self.user_factors = np.array(pd.read_csv(user_file).values)[:, 1:]
        self.item_factors = np.array(pd.read_csv(item_file).values)[:, 1:]
        '''
        num_loss_samples = int(100*self.dataModel.getUsersNum()**0.5)
        #print 'sampling {0} <user,item i,item j> triples...'.format(num_loss_samples)
        loss_sampler = UniformUserUniformItem(True)
        self.loss_samples = [t for t in loss_sampler.generate_samples(self.dataModel, num_loss_samples)]
        old_loss = self.loss()

        update_sampler = UniformPairWithoutReplacement(True)
        #print 'initial loss = {0}'.format(self.loss())
        for it in xrange(self.iter):
            #print 'starting iteration {0}'.format(it)
            for u, i, j in update_sampler.generate_samples(self.dataModel):
                self.update_factors(u, i, j)
            if abs(self.loss() - old_loss) < 0.1 or self.loss() - old_loss > 0:
                #print 'iteration {0}: loss = {1}'.format(it, self.loss())
                #print 'converge!!'
                break
            else:
                old_loss = self.loss()
                self.learning_rate *= 0.9
                #print 'iteration {0}: loss = {1}'.format(it, self.loss())

    def predict_single(self,uid,iid):
        return self.item_bias[iid] + np.dot(self.user_factors[uid],self.item_factors[iid])
    def recommend(self, uid):
        predict_scores = []
        for i in range(self.dataModel.getItemsNum()):
            s = self.predict_single(uid, i)
            predict_scores.append(s)
        topN = np.argsort(np.array(predict_scores))[-1:-self.n - 1:-1]
        return topN
    def score(self, testSamples, trueLabels):
        print 'BPR scoring ...'
        trueList = []
        recommendList= []
        user_unique = list(set(np.array(testSamples)[:,0]))
        for u in user_unique:
            uTrueIndex = np.argwhere(np.array(testSamples)[:,0] == u)[:,0]
            #true = [self.dataModel.getIidByItem(i) for i in list(np.array(testSamples)[uTrueIndex][:,1])]
            true = list(np.array(testSamples)[uTrueIndex][:,1])
            trueList.append(true)
            uid = self.dataModel.getUidByUser(u)
            pre = [self.dataModel.getItemByIid(i) for i in self.recommend(uid)]
            recommendList.append(pre)
        e = Eval()
        result = e.evalAll(trueList, recommendList)
        print 'BPR result:'+ '('+str(self.get_params())+')'+str((result)['F1'])
        return (result)['F1']


class Sampler(object):

    def __init__(self,sample_negative_items_empirically):
        self.sample_negative_items_empirically = sample_negative_items_empirically

    def init(self,dataModel,max_samples=None):
        self.dataModel = dataModel
        self.num_users = self.dataModel.getUsersNum()
        self.num_items = self.dataModel.getItemsNum()
        self.max_samples = max_samples
    def num_samples(self,n):
        if self.max_samples is None:
            return n
        return min(n,self.max_samples)

    def uniform_user(self):
        return random.randint(0,self.num_users-1)
    def random_item(self):
        """sample an item uniformly or from the empirical distribution
           observed in the training data
        """
        if self.sample_negative_items_empirically:
            # just pick something someone rated!
            u = self.uniform_user()
            i = random.choice(self.dataModel.getItemIDsFromUid(u))
        else:
            i = random.randint(0,self.num_items-1)
        return i
    def sample_negative_item(self,user_items):
        j = self.random_item()
        while j in user_items:
            j = self.random_item()
        return j

class UniformUserUniformItem(Sampler):

    def generate_samples(self, dataModel, max_samples=None):
        self.init(dataModel, max_samples)
        for _ in xrange(self.num_samples(self.dataModel.getData().nnz)):
            uid = self.uniform_user()
            iid_i = random.choice(self.dataModel.getItemIDsFromUid(uid))
            iid_j = self.sample_negative_item(self.dataModel.getItemIDsFromUid(uid))
            yield uid, iid_i, iid_j

class UniformPairWithoutReplacement(Sampler):

    def generate_samples(self,dataModel,max_samples=None):
        self.init(dataModel, max_samples)
        idxs = range(self.dataModel.getData().nnz)
        random.shuffle(idxs)
        self.users, self.items = self.dataModel.getData().nonzero()
        self.users = self.users[idxs]
        self.items = self.items[idxs]
        self.idx = 0
        for _ in xrange(self.num_samples(self.dataModel.getData().nnz)):
            u = self.users[self.idx]
            i = self.items[self.idx]
            j = self.sample_negative_item(self.dataModel.getItemIDsFromUid(u))
            self.idx += 1
            yield u, i, j

if __name__ == '__main__':
    bpr = BPR()
    data = pd.read_csv('../Data/bbg/transaction.csv')
    samples = [[int(i[0]), int(i[1])] for i in data.values[:,0:2]]
    targets = [1 for i in samples]
    parameters = {'n':[5]}

    clf = grid_search.GridSearchCV(bpr, parameters,cv=5)
    clf.fit(samples, targets)
    print(clf.grid_scores_)

