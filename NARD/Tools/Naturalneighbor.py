
import numpy as np

class NNSearch():
    def __init__(self,A):
        self.A = A
        (self.NaN,self.nn,self.rnn,self.dis_index)=self.natural_search()

## 获得带有索引的排序字典dis_index
    def get_dis_index(self):
        #传入的A应该是欧式距离矩阵
        A = self.A
        n = A.shape[0]
        dis_index = {}
        nn = {}
        rnn = {}
        for i in range(0,n):
             dis = np.sort(A[i,:])
             index = np.argsort(A[i,:])
             dis_index[i]=[dis,index]
             nn[i] = []
             rnn[i] = []
        return dis_index,nn,rnn


## 自动迭代寻找自然邻居，返回迭代次数t,最近邻nn，逆近邻rnn
    def natural_search(self):
        n = self.A.shape[0]
        dis_index,nn,rnn = self.get_dis_index()
        nb = [0]*n
        t = 0
        num_1 = 0
        num_2 = 0
        while t+1 < n:
            for i in range(0,n):
                x = i
                y = dis_index[x][1][t+1]
                nn[x].append(y)
                rnn[y].append(x)
                nb[y] = nb[y]+1
            num_1 = nb.count(0)
            if num_1 != num_2:
                num_2 = num_1
            else:
                break
            t = t+1
        NaN=[]
        for i in range(len(nn)):
           NaN.append(set(nn[i])&set(rnn[i]))
        return NaN,nn,rnn,dis_index

if __name__ == '__main__':
    print('this is a main process')
