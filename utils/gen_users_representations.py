import pandas as pd
import numpy as np
import operator
import os

user_representation = np.zeros(9238)

def get_products_from_string(string):
    p_array = string.split(' ')
    return p_array

def user_seq_representation(tran_array):
    user_dict = dict()
    length = len(tran_array)
    for i in range(length):
        for j in range(i+1, length):
            i_array = get_products_from_string(tran_array[i])
            j_array = get_products_from_string(tran_array[j])
            for x in i_array:
                for y in j_array:
                    key = x + '@' + y
                    if user_dict.has_key(key):
                        user_dict[key] += 1
                    else:
                        user_dict[key] = 1
    tmp = sorted(user_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
    user_dict = dict(tmp[:100])
    return str(user_dict)

def gen_seq_simi():
    file = 'new_format_seq.txt'
    data = pd.read_csv(file, names=np.array(range(1)))
    results = []
    h = 0
    for line in data.values:
        transactions = line[0]
        tran_array = transactions.split('-1')
        tran_array = [i.strip() for i in tran_array]
        tran_array = tran_array[:-1]
        u_seq_str = user_seq_representation(tran_array)
        results.append(u_seq_str)
        print h
        h += 1
    t = pd.DataFrame(results)
    t.to_csv('users_seq_representation')

def nor_seq_file():
    sum_dict = dict()
    filename = 'users_seq_representation'
    data = pd.read_csv(filename)
    for line in data.values:
        line_dict = eval(line[1])
        for (k, v) in line_dict.items():
            if sum_dict.has_key(k):
                sum_dict[k] += v**2
            else:
                sum_dict[k] = v**2
    print sum_dict['1405@17']
    for (k, v) in sum_dict.items():
        value = v**0.5
        sum_dict[k] = value
    print sum_dict['1405@17']
    results = []
    for line in data.values:
        user_nor = dict()
        line_dict = eval(line[1])
        for (k, v) in line_dict.items():
            user_nor[k] = v / sum_dict[k]
        results.append(str(user_nor))
    t = pd.DataFrame(results)
    t.to_csv('nor_seq_simi_file')


if __name__ == '__main__':
    print 'gg'
    if os.path.exists('users_seq_representation'):
        print 'users_seq_representation has been generated!'
        nor_seq_file()
    else:
        gen_seq_simi()
        nor_seq_file()


