__author__ = 'Jerry'

from abc import ABCMeta, abstractmethod
import pandas as pd
from utils.Config import Config

class BaseDataModel:
    __metaclass__ = ABCMeta

    def __init__(self, config=dict()):
        if not isinstance(config, dict):
            raise TypeError(config + ' is not an instance of dict')
        self.conf = config

