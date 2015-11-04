__author__ = 'Jerry'

import pandas as pd

class Config:

    def __init__(self, config_path):
        data = pd.read_csv(config_path, header=None)
        self.__params = {}
        for row in data.values:
            self.__params[row[0]] = row[1]

    def get(self, name, default=None):
        return self.__params.get(name, default)

if __name__ == '__main__':
    config = Config('./configOfYelp')
    print config.get('test')
    print config.get('a')
    print config.get('a', 'b')