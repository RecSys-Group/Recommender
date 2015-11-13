__author__ = 'Jerry'

import pandas as pd
import ConfigParser

class Parameter:

    def __init__(self, paras):
        self.paras = paras
        self.lenList = {}
        self.currentPos = {}
        for k, values in paras.items():
            self.lenList[k] = len(values)
            self.currentPos[k] = 0

    def getNext(self):
        for k in self.lenList.keys():
            if self.currentPos[k] < self.lenList[k]-1:
                self.currentPos[k] += 1
                return True
        return False

    def iter(self):
        yield self.to_dict(self.currentPos)
        while self.getNext():
            yield self.to_dict(self.currentPos)
        raise StopIteration

    def to_dict(self, pos):
        result = {}
        for k, v in pos.items():
            result[k] = self.paras[k][v]
        return result

    def __iter__(self):
        return self

class Config:

    def __init__(self):
        self.paras = {}

    def from_ini(self, config_path, splitString='|'):
        conf = ConfigParser.ConfigParser()

        conf.read(config_path)

        secs = conf.sections()

        self.data = dict(conf.items('data'))
        self.algList = conf.get('algorithms', 'all').split(splitString)
        for alg in self.algList:
            if alg in secs:
                para = {}
                opts = conf.options(alg)
                for opt in opts:
                    para[opt] = conf.get(alg, opt).split(splitString)
                self.paras[alg] = Parameter(para)

        # #read by type
        # db_host = conf.get("db", "db_host")
        # db_port = conf.getint("db", "db_port")
        # db_user = conf.get("db", "db_user")
        # db_pass = conf.get("db", "db_pass")
        #
        # #read int
        # threads = conf.getint("concurrent", "thread")
        # processors = conf.getint("concurrent", "processor")

    # def from_csv(self, config_path):
    #     data = pd.read_csv(config_path, header=None)
    #     self.__params = {}
    #     for row in data.values:
    #         self.__params[row[0]] = row[1]
    #
    # def get(self, name, default=None):
    #     result = self.__params.get(name, default)
    #     if result is None:
    #         raise Exception(name + ' is not in config file: ' + self.path)
    #     return result

if __name__ == '__main__':
    config = Config()
    config.from_ini('../Application/conf')
    print config.algList
    print config.data
    for k, vs in config.paras.items():
        for v in vs.iter():
            print k, v
