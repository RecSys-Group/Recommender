from __future__ import division
import numpy as np
import pandas as pd
import scipy.sparse as spr
import datetime
from scipy import spatial



class SIMILAR:
    def __init__(self):
        print 'init'
    def load_data(self):
        print 'load_data'
        nn_file = '../Data/Tafeng/new_col_nor_ui_01.txt'
        seq_file = '../Data/Tafeng/nor_seq_simi_file'
        users_file = '../Data/Tafeng/users.txt'
        items_file = '../Data/Tafeng/item.txt'

        self.nn_data = pd.read_csv(nn_file, dtype='str')
        self.seq_data = pd.read_csv(seq_file, dtype='str')
        RATE_MATRIX = np.zeros((9238, 7973))
        for line in self.nn_data.values:
            uid = int(float(line[1]))
            iid = int(float(line[2]))
            RATE_MATRIX[uid][iid] = float(line[3])
        self.nn_data_sparse = spr.csr_matrix(RATE_MATRIX)

        self.users = pd.read_csv(users_file, dtype='str')
        self.items = pd.read_csv(items_file)

    def nn_simi(self, uid1, uid2):
        simi = 1 - spatial.distance.cosine(np.array(self.nn_data_sparse.getrow(uid1).todense()), np.array(self.nn_data_sparse.getrow(uid2).todense()) )
        return simi
    def gen_nn_similar(self):
        K = 50
        NN_similar = []
        for i in range(len(self.users)):
            s = datetime.datetime.now()
            tmp = []
            for j in range(len(self.users)):
                print i, j
                tmp.append([str(j), str(self.nn_simi(i, j))])
            topK_index = np.argsort(np.array(tmp)[:,1])[-1:-K-1:-1]
            tmp_tu = [dict({i[0]:i[1]}) for i in tmp]
            NN_similar.append(np.array(tmp_tu)[topK_index])
            print np.array(tmp_tu)[topK_index]
            print str(datetime.datetime.now() - s)
        t = pd.DataFrame(NN_similar)
        t.to_csv('tafeng_nor_NN_similar')


    def seq_simi(self, u_vector, v_vector):
        sum = 0
        u_abs = 0
        v_abs = 0
        for (k, v) in u_vector.items():
            u_abs += v ** 2
            if v_vector.has_key(k):
                sum += v*v_vector[k]
        for (k, v) in u_vector.items():
            v_abs += v ** 2
        simi = sum / (u_abs * v_abs)**0.5
        return simi
    def gen_seq_similar(self):
        print 'seq_simi'
        K = 50
        results = []
        for i in range(len(self.seq_data)):
            s = datetime.datetime.now()
            tmp = []
            for j in range(len(self.seq_data)):
                print i, j
                tmp.append([str(j), str(float(self.seq_simi(eval(self.seq_data.values[i][1]), eval(self.seq_data.values[j][1]))))])
            values = [float(h) for h in np.array(tmp)[:,1]]
            topK_index = np.argsort(np.array(values))[-1:-K-1:-1]
            tmp_tu = [dict({m[0]:m[1]}) for m in tmp]
            results.append(np.array(tmp_tu)[topK_index])
            print np.array(tmp_tu)[topK_index]
            print str(datetime.datetime.now() - s)
            raw_input()

        t = pd.DataFrame(results)
        t.to_csv('tafeng_nor_SEQ_similar_100')




if __name__ == '__main__':
    s = SIMILAR()
    s.load_data()
    s.gen_seq_similar()



