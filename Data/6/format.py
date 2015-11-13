__author__ = 'Jerry'
import pandas as pd

if __name__ == '__main__':
    filename = './train.csv'
    data = pd.read_csv(filename)
    data = data.ix[:, ['user', 'item', 'rate']]
    data.to_csv(filename)

    filename = './test.csv'
    data = pd.read_csv(filename)
    data = data.ix[:, ['user', 'item', 'rate']]
    data.to_csv(filename)

    # filename = './item.csv'
    # data = pd.read_csv(filename)
    # data = data.ix[:, ['item']]
    # data.to_csv(filename)
    #
    # filename = './user.csv'
    # data = pd.read_csv(filename)
    # data = data.ix[:, ['user']]
    # data.to_csv(filename)