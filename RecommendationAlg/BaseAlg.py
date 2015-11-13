__author__ = 'Jerry'
from abc import ABCMeta, abstractmethod, abstractproperty
from DataModel.BaseDataModel import BaseDataModel

class BaseAlg:
    __metaclass__ = ABCMeta

    def __init__(self, dataModel, name):
        if not isinstance(dataModel, BaseDataModel):
            raise TypeError(dataModel + ' is not an instance of BaseDataModel')
        self.dataModel = dataModel
        self.name = name

    @abstractmethod
    def recommend(self, userID=None, N=5):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def recommendAllUserInTest(self, N=5):
        pass

    def getName(self):
        return self.name