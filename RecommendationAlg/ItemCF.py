__author__ = 'Jerry'

from BaseAlg import BaseAlg

class ItemCF(BaseAlg):

    def __init__(self, dataModel):
        super(ItemCF, self).__init__(dataModel, 'ItemCF')

    def recommend(self, userID=None, N=5):
        pass

    def train(self):
        print 'ItemCF train'

    def recommendAllUserInTest(self, N=5):
        pass

if __name__ == '__main__':
    itemcf = ItemCF('a')
    itemcf.train()
