__author__ = 'Jerry'
import pandas as pd

if __name__ == '__main__':
    filename1 = './train.csv'
    filename2 = './test.csv'
    data1 = pd.read_csv(filename1)
    data1 = data1.ix[:, [0, 1, 3]]
    data1.to_csv(filename1)

    data2 = pd.read_csv(filename2)
    data2 = data2.ix[:, [0, 1, 3]]
    data2.to_csv(filename2)