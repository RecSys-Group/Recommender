__author__ = 'Jerry'

from abc import ABCMeta, abstractmethod
import pandas as pd
from utils.Config import Config

class BaseDataModel:
    __metaclass__ = ABCMeta

    def __init__(self, config):
        if not isinstance(config, dict):
            raise TypeError(config + ' is not an instance of dict')
        self.conf = config

    @abstractmethod
    def getUserIDs(self):
        pass

    @abstractmethod
    def getItemIDs(self):
        pass

    @abstractmethod
    def getUserFromID(self, userID):
        pass

    @abstractmethod
    def getItemFromID(self, itemID):
        pass

    @abstractmethod
    def getItemsNum(self):
        pass

    @abstractmethod
    def getUsersNum(self):
        pass

    @abstractmethod
    def getUserIDsInTrain(self):
        pass

    @abstractmethod
    def getUserIDsInTest(self):
        pass

    @abstractmethod
    def getUsersNumInTrain(self):
        pass

    @abstractmethod
    def getUsersNumInTest(self):
        pass

    @abstractmethod
    def getItemIDsForEachUserInTest(self):
        pass

    @abstractmethod
    def getTrain(self):
        pass

    @abstractmethod
    def getTest(self):
        pass