
import copy

import numpy as np
from scipy.spatial.distance import squareform, pdist

from Tools.Plot import plot_dot


class NARD:

    def __init__(self, data, NNtool, SDGS):
        self.data_=data
        self.centers_=[]
        self.SDGS_=SDGS.SDGS_
        self.SDGS_class_=SDGS
        self.NNtool_=NNtool
        self.NNES_=self.NNES_initiation()
        self.NARD_=self.NARD_generation()

    ## 实例化对象时自动进行邻域扩张
    def NNES_initiation(self):
        NNES={}
        for i in range(len(self.NNtool_.NaN)):
            neighborRangeTemp = set(self.NNtool_.NaN[i])
            for extraNeighbor in self.NNtool_.NaN[i]:
                neighborRangeTemp |= set(self.NNtool_.NaN[extraNeighbor])
            neighborRangeTemp2 = set(neighborRangeTemp)
            for extraNeighbor in neighborRangeTemp:
                neighborRangeTemp2 |= set(self.NNtool_.NaN[extraNeighbor])
            if i in neighborRangeTemp2:
                neighborRangeTemp2.remove(i)
            NNES[i] = list(neighborRangeTemp2)
        return NNES

    #采用局部相对密度的计算方式，来得到局部的自适应符合分布的密度，密度=自身值/局部最大值
    def NARD_generation(self):
        NARD=np.zeros(len(self.data_))
        for key, value in self.SDGS_.items():
            NNES_mapped_of_fdg=[]

            for i, index in enumerate(value.dataIndex):
                #对下标进行映射，特征分布里面的NAN下标是全局的，计算时需要传局部的
                NNES_mapped = np.where(np.isin(value.dataIndex, self.NNES_[index]))[0]
                # 转换全局的index为局部的index，用于局部相对密度计算
                NNES_mapped_of_fdg.append(NNES_mapped)
            NDD = self.NDD_generation(value.data, NNES_mapped_of_fdg,value.dataIndex)
            localDensityPeak=np.max(NDD)
            for i,index in enumerate(value.dataIndex):
                nard=NDD[i]/localDensityPeak
                if nard==1:
                    self.centers_.append(index)
                NARD[index]=nard
        return NARD


    def NDD_generation(self,fdgData,adaptiveNeighborhood,value_dataIndex):
        # 计算特征分布内数据的距离矩阵
        disMatrix_of_fdg = squareform(pdist(fdgData,metric='euclidean'))

        length=len(fdgData)
        Dist, MGD = self.MGD_generation(disMatrix_of_fdg, length, adaptiveNeighborhood, value_dataIndex)
        NDD = []
        for i in range(len(adaptiveNeighborhood)):
            NDD_i = MGD[i]
            if(len(adaptiveNeighborhood[i])>0):
                for index, j in enumerate(adaptiveNeighborhood[i]):  # for each neighbor
                    if j != -1:
                        wkDenj = MGD[j] * (1 / Dist[i][index])  # wkDenj is an array
                        NDD_i = NDD_i + wkDenj
                NDD.append(NDD_i)
            else:
                NDD.append(NDD_i)
        return NDD

    def MGD_generation(self, disMatrix_of_fdg, length, adaptiveNeighborhood, value_dataIndex):

        Dist = []
        MGD = []
        for i in range(length):
            ithDistances = disMatrix_of_fdg[i]
            Dist.append(ithDistances[adaptiveNeighborhood[i]])
            ##如果邻居数为0，则自适应邻居距离设置为无限小
            if len(adaptiveNeighborhood[i])!=0:
                MGD.append(1 / np.average(Dist[i]))
            else:
                # 防止出现inf
                MGD.append(0.0000001)
        return Dist, MGD