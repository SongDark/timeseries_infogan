
import numpy as np
from utils import one_hot_encode, shuffle_in_unison_scary

class datamanager(object):
    def __init__(self, datapath, train_ratio=None, fold_k=None, norm=None, expand_dim=None, seed=233):
        self.seed = seed
        data = np.load(datapath)
        self.data = np.concatenate([data["x_train"], data["x_test"]])
        self.labels = np.concatenate([data["y_train"], data["y_test"]])
        self.class_num = np.max(self.labels) + 1
        self.labels = one_hot_encode(self.labels, self.class_num)
        del data
        
        self.divide_train_test(train_ratio, fold_k)
        
        if norm is not None:
            self.data = self.data / norm # [0,1]
        
        self.data = self.data[:,:,:2]
        
        self.train_cur_pos, self.test_cur_pos = 0, 0

        self.expand_dim = expand_dim

    def get_train_test_id(self, train_ratio, fold_k, seed=None):
        self.train_id, self.test_id = [], []
        # normal
        if train_ratio and not fold_k:
            for v in self.dict_by_class:
                self.train_id += v[:int(train_ratio * len(v))]
                self.test_id += v[int(train_ratio * len(v)):]
        # cross validation
        if fold_k and not train_ratio:
            for i in range(10):
                self.test_id += list(self.dict_by_class[i][self.test_fold_id])
                for j in range(fold_k):
                    if j != self.test_fold_id:
                        self.train_id += list(self.dict_by_class[i][j])
        self.train_id = np.array(self.train_id)
        self.test_id = np.array(self.test_id)

        shuffle_in_unison_scary(self.train_id, self.test_id, seed=(seed or self.seed))
        self.train_num, self.test_num = len(self.train_id), len(self.test_id)

    def divide_train_test(self, train_ratio, fold_k, seed=None):
        self.dict_by_class = [[] for i in range(self.class_num)]

        for i, key in enumerate(np.argmax(self.labels, axis=1)):
            self.dict_by_class[key].append(i)
        for i in range(self.class_num):
            np.random.seed(i)
            np.random.shuffle(self.dict_by_class[i])
        
        if fold_k and not train_ratio:
            # only for cross validation
            print "[{} folds cross validation]".format(fold_k)
            for i in range(self.class_num):
                np.random.seed(i)
                np.random.shuffle(self.dict_by_class[i])
                self.dict_by_class[i] = np.array_split(self.dict_by_class[i], fold_k)
                np.random.seed(i)
                np.random.shuffle(self.dict_by_class[i])
            self.test_fold_id = 0
        self.get_train_test_id(train_ratio, fold_k, seed)
    
    def shuffle_train(self, seed=None):
        np.random.seed(seed)
        np.random.shuffle(self.train_id)
    
    def get_cur_pos(self, cur_pos, full_num, batch_size):
        get_pos = range(cur_pos, cur_pos + batch_size)
        if cur_pos + batch_size <= full_num:
            cur_pos += batch_size
        else:
            rest = cur_pos + batch_size - full_num
            get_pos = range(cur_pos, full_num) + range(rest)
            cur_pos = rest
        return cur_pos, get_pos

    def __call__(self, batch_size, phase='train', maxlen=None, var_list=[]):
        if phase == 'train':
            self.train_cur_pos, get_pos = self.get_cur_pos(self.train_cur_pos, self.train_num, batch_size)
            cur_id = self.train_id[get_pos]
        elif phase == 'test':
            self.test_cur_pos, get_pos = self.get_cur_pos(self.test_cur_pos, self.test_num, batch_size)
            cur_id = self.test_id[get_pos]

        def func(flag, maxlen=maxlen):
            if flag == 'data':
                res = self.__dict__[flag][cur_id]
                if self.expand_dim is not None:
                    res = np.expand_dims(res, self.expand_dim)
                return res
            elif flag == 'labels':
                return self.labels[cur_id]
        
        res = {}
        for key in var_list:
            if not res.has_key(key):
                res[key] = func(key)
        
        return res


# data = datamanager("data/CharacterTrajectories/CharacterTrajectories.npz", 
#                    train_ratio=0.8, fold_k=None, norm=None, expand_dim=False, seed=0)
# X = data(64, var_list=['data', 'labels'])
# print X['data'].shape 
# print X['labels'].shape
# seq = X['data'][3]
# print np.argmax(X['labels'][3])

# import matplotlib 
# matplotlib.use("Agg")
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot as plt 


# # fig = plt.figure()
# # ax = fig.gca(projection='3d')
# # ax.plot3D(seq[:, 0], seq[:, 1], seq[:, 2])

# plt.plot(seq[:,0], seq[:,1])


# plt.savefig('figs/test/tmp.png')