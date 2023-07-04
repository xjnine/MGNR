
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import arff
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import MinMaxScaler

class DensityPeak:
    """
    密度峰值聚类算法
    """

    def __init__(self, distanceMatrix, dcRatio=0.2, clusterNumRatio=0.01, dcType="max", kernel="gaussian"):
        '''
        构造器，初始化相关参数
        :param distanceMatrix: 数据集的距离矩阵
        :param dcRatio: 半径比率 通常是0.2
        :param dcType: 半径计算类型 包括‘max’,'ave','min' Hausdorff距离等
        :param kernel: 密度计算时选取的计算函数 包括'cutoff-kernel' 'gaussian-kernel'
        '''
        # 实例间距离矩阵
        self.distance_m = distanceMatrix
        # 半径比率
        self.dcRatio_f = dcRatio
        # 半径类型
        self.dcType = dcType
        # 密度计算核
        self.kernel = kernel
        # 簇中心数量占比
        self.clusterCenterRatio_f = clusterNumRatio
        # 密度向量，存储密度
        self.densities_l = []
        # 存储master
        self.masters_l = []
        # 存储实例到其master的距离
        self.distanceToMaster_l = []
        # 代表性向量，存储实例的代表性
        self.representativeness_l = []
        # 簇中心
        self.clusterCenter_l = []
        # 实例数量
        self.numSample = 0
        # 半径dc
        self.dc_f = 0
        # 数据集最大实例间距离
        self.maxDistance = 0
        # 聚类标签
        self.label_l = []
        # 簇块 一个字典 簇号:[簇块]
        self.clusters_d = {}

        self.__initDensityPeak()

    def __initDensityPeak(self):
        '''
        初始化
        :return:
        '''
        # 实例数量
        self.numSample = len(self.distance_m)
        # 最大实例间距离
        self.maxDistance = self.getMaxDistance()
        # 计算半径dc
        self.dc_f = self.getDc()
        # 计算密度
        self.densities_l = self.computeDensities()
        # 计算实例到master的距离
        self.computeDistanceToMaster()
        # 计算实例的代表性
        self.computePriority()

    def getDc(self):
        '''
        计算半径dc
        :return:
        '''
        resultDc = 0.0
        if self.dcType=="max":
                '''
                计算最大Hausdorff距离
                '''
                resultDc = self.maxDistance
        elif self.dcType=="ave":
                '''
                平均Hausdorff距离
                '''
                resultDc = np.mean(self.distance_m)
        elif self.dcType =="min":
                '''
                最小Hausdorff距离
                '''
                resultDc = np.min(self.distance_m)

        return resultDc * self.dcRatio_f

    def getMaxDistance(self):
        '''
        计算实例间最大距离
        :return:
        '''
        return np.max(self.distance_m)

    def computeDensities(self):
        '''
        计算密度，按照给定的kernel进行计算
        :return:
        '''
        # 按照高斯核计算
        if self.kernel == 'gaussian':
            # 方法一，使用循环
            # temp_local_density_list = []
            # for i in range(0, self.numSample):
            #     temp_local_density_list.append(self.gaussian_kernel(i))

            # 方法二，使用矩阵运算
            temp_local_density_list = np.sum(1 / (np.exp(np.power(self.distance_m / self.dc_f, 2))), axis=1)
            return temp_local_density_list
        # 按照截断核计算
        elif self.kernel == 'cutoff':
            temp_local_density_list = []
            for i in range(0, self.numSample):
                temp_local_density_list.append(self.cutoff_kernel(i))
            return temp_local_density_list

    def gaussian_kernel(self, i):
        '''
        高斯核计算密度
        :param i: 实例标号
        :return: 密度
        '''
        tempDensity = 0
        for j in range(len(self.distance_m[i])):
            tempDistance = self.distance_m[i][j]
            tempDensity += np.exp(-(tempDistance / self.dc_f) ** 2)
        return tempDensity

    def cutoff_kernel(self, i):
        '''
        截断核计算密度
        :param i: 实例标号
        :return: 密度
        '''
        tempDensity = 0
        for j in range(len(self.distance_m[i])):
            tempDistance = self.distance_m[i][j]
            tempDensity += self.F(tempDistance - self.dc_f)
        return tempDensity

    def F(self, x):
        '''
        截断核计算辅助函数
        :param x: 距离差值
        :return:
        '''
        if x < 0:
            return 1
        else:
            return 0

    def computeDistanceToMaster(self):
        '''
        计算实例到master的距离，同时确定实例的master
        :return:
        '''
        # 将密度降序排序，返回索引
        tempSortDensityIndex = np.argsort(self.densities_l)[::-1]
        # 初始化距离向量
        self.distanceToMaster_l = np.zeros(self.numSample)
        # 密度最高的获得最高优先级
        self.distanceToMaster_l[tempSortDensityIndex[0]] = float('inf')
        # 初始化master向量
        self.masters_l = np.zeros(self.numSample, dtype=int)
        # 密度最高的自己是自己的master
        self.masters_l[tempSortDensityIndex[0]] = 0

        # 计算距离和master
        # 选择密度大于自己且距离最近的作为自己的master
        for i in range(1, self.numSample):
            tempIndex = tempSortDensityIndex[i]
            self.masters_l[tempIndex] = tempSortDensityIndex[
                np.argmin(self.distance_m[tempIndex][tempSortDensityIndex[:i]])]
            self.distanceToMaster_l[tempIndex] = np.min(self.distance_m[tempIndex][tempSortDensityIndex[:i]])
        # print(self.masters_l)

    def computePriority(self):
        '''
        计算代表性（优先级）
        :return:
        '''



        self.representativeness_l = np.multiply(self.densities_l, self.distanceToMaster_l)

    def getLabel(self, i):
        '''
        获取实例的标签
        :param i: 实例标号
        :return: 实例聚类标签
        '''
        if self.label_l[i] < 0:
            return self.label_l[i]
        else:
            # 实例没有标签，则使用其master的标签作为自己的标签 聚类中即为聚类簇号
            return self.getLabel(self.masters_l[i])

    def getClusterCenter(self):
        n = int(self.numSample * self.clusterCenterRatio_f)
        return np.argsort(self.representativeness_l)[-n:][::-1]

    def cluster(self):
        '''
        按照比例计算聚类簇中心个数 进行聚类
        :param clusterRatio: 簇中心占比
        :return:
        '''
        n = int(self.numSample * self.clusterCenterRatio_f)
        # n = 3
        self.cluster2(n=n)

    def cluster2(self, n=3):
        '''
        按照给定的簇中心个数进行聚类
        :param n: 簇中心个数
        :return:
        '''

        # 初始化标签向量
        self.label_l = np.zeros(self.numSample, dtype=int)
        # 初始化聚类中心
        self.clusterCenter_l = np.argsort(self.representativeness_l)[-n:][::-1]
        # 初始化簇号 使用簇号作为聚类标签
        for i in range(n):
            self.label_l[self.clusterCenter_l[i]] = -i - 1

        # 统计聚类标签
        for i in range(self.numSample):
            if self.label_l[i] < 0:
                continue
            self.label_l[i] = self.getLabel(self.masters_l[i])

        # 初始化聚类簇块
        self.clusters_d = {key: [] for key in self.label_l[self.clusterCenter_l]}

        # 按照聚类结果划分簇块
        for i in self.label_l[self.clusterCenter_l]:
            self.clusters_d[i] += [j for j in range(self.numSample) if self.label_l[j] == i]

    @staticmethod
    def getDistanceByEuclid(instance1, instance2):
        '''
        按照欧氏距离计算实例间距离
        :param instance1: 实例1
        :param instance2: 实例2
        :return: 欧氏距离
        '''
        dist = 0
        for key in range(len(instance1)):
            dist += (float(instance1[key]) - float(instance2[key])) ** 2
        return dist ** 0.5
class DensityPeak_Auto_Adaptive:
    """
    密度峰值聚类算法
    """
    def __init__(self, points,distanceMatrix,fdn,NARD,dcRatio=0.2, clusterNumRatio=0.15, dcType="max", kernel="gaussian"):
        '''
        构造器，初始化相关参数
        :param distanceMatrix: 数据集的距离矩阵
        :param dcRatio: 半径比率 通常是0.2
        :param dcType: 半径计算类型 包括‘max’,'ave','min' Hausdorff距离等
        :param kernel: 密度计算时选取的计算函数 包括'cutoff-kernel' 'gaussian-kernel'
        '''
        # 数据集
        self.points=points
        # 特征分布数
        self.fdn=fdn
        self.NARD=NARD
        # 实例间距离矩阵
        self.distance_m = distanceMatrix
        # 半径比率
        self.dcRatio_f = dcRatio
        # 半径类型
        self.dcType = dcType
        # 密度计算核
        self.kernel = kernel
        # 簇中心数量占比
        self.clusterCenterRatio_f = clusterNumRatio
        # 密度向量，存储密度
        self.densities_l = []
        # 存储master
        self.masters_l = []
        # 存储实例到其master的距离
        self.distanceToMaster_l = []
        # 代表性向量，存储实例的代表性
        self.representativeness_l = []
        # 簇中心
        self.clusterCenter_l = []
        # 实例数量
        self.numSample = 0
        # 数据集最大实例间距离
        self.maxDistance = 0
        # 聚类标签
        self.label_l = []
        # 簇块 一个字典 簇号:[簇块]
        self.clusters_d = {}

        self.__initDensityPeak()

    def __initDensityPeak(self):
        '''
        初始化
        :return:
        '''
        # 实例数量
        self.numSample = len(self.distance_m)
        # 最大实例间距离
        self.maxDistance = self.getMaxDistance()
        # 计算半径dc
        # self.dc_f = self.getDc()
        # 计算密度
        self.densities_l = self.computeDensities()
        # 计算实例到master的距离
        self.computeDistanceToMaster()
        # 计算实例的代表性
        self.computePriority()

    def getDc(self):
        '''
        计算半径dc
        :return:
        '''
        resultDc = 0.0
        if self.dcType=="max":
                '''
                计算最大Hausdorff距离
                '''
                resultDc = self.maxDistance
        elif self.dcType=="ave":
                '''
                平均Hausdorff距离
                '''
                resultDc = np.mean(self.distance_m)
        elif self.dcType =="min":
                '''
                最小Hausdorff距离
                '''
                resultDc = np.min(self.distance_m)

        return resultDc * self.dcRatio_f

    def getMaxDistance(self):
        '''
        计算实例间最大距离
        :return:
        '''
        return np.max(self.distance_m)

    def computeDensities(self):
        '''
        计算密度，按照给定的kernel进行计算
        :return:
        '''

        #adaptive_adrelative_density_calculating
        if self.kernel == 'NARD':


            return self.NARD



    def gaussian_kernel(self, i):
        '''
        高斯核计算密度
        :param i: 实例标号
        :return: 密度
        '''
        tempDensity = 0
        for j in range(len(self.distance_m[i])):
            tempDistance = self.distance_m[i][j]
            tempDensity += np.exp(-(tempDistance / self.dc_f) ** 2)
        return tempDensity

    def cutoff_kernel(self, i):
        '''
        截断核计算密度
        :param i: 实例标号
        :return: 密度
        '''
        tempDensity = 0
        for j in range(len(self.distance_m[i])):
            tempDistance = self.distance_m[i][j]
            tempDensity += self.F(tempDistance - self.dc_f)
        return tempDensity

    def F(self, x):
        '''
        截断核计算辅助函数
        :param x: 距离差值
        :return:
        '''
        if x < 0:
            return 1
        else:
            return 0

    def computeDistanceToMaster(self):
        '''
        计算实例到master的距离，同时确定实例的master
        :return:
        '''
        # 将密度降序排序，返回索引
        tempSortDensityIndex = np.argsort(self.densities_l)[::-1]
        # 初始化距离向量
        self.distanceToMaster_l = np.zeros(self.numSample)
        # 密度最高的获得最高优先级
        self.distanceToMaster_l[tempSortDensityIndex[0]] = float('inf')
        # 初始化master向量
        self.masters_l = np.zeros(self.numSample, dtype=int)
        # 密度最高的自己是自己的master
        self.masters_l[tempSortDensityIndex[0]] = 0

        # 计算距离和master
        # 选择密度大于自己且距离最近的作为自己的master
        for i in range(1, self.numSample):
            tempIndex = tempSortDensityIndex[i]
            self.masters_l[tempIndex] = tempSortDensityIndex[
                np.argmin(self.distance_m[tempIndex][tempSortDensityIndex[:i]])]
            self.distanceToMaster_l[tempIndex] = np.min(self.distance_m[tempIndex][tempSortDensityIndex[:i]])
        # print(self.masters_l)

    def computePriority(self):
        '''
        计算代表性（优先级）
        :return:
        '''
        self.representativeness_l = np.multiply(self.densities_l, self.distanceToMaster_l)

    def getLabel(self, i):
        '''
        获取实例的标签
        :param i: 实例标号
        :return: 实例聚类标签
        '''
        if self.label_l[i] < 0:
            return self.label_l[i]
        else:
            # 实例没有标签，则使用其master的标签作为自己的标签 聚类中即为聚类簇号
            return self.getLabel(self.masters_l[i])

    def getClusterCenter(self):
        n = int(self.numSample * self.clusterCenterRatio_f)
        return np.argsort(self.representativeness_l)[-n:][::-1]

    def cluster(self):
        '''
        按照比例计算聚类簇中心个数 进行聚类
        :param clusterRatio: 簇中心占比
        :return:
        '''
        # n = int(self.numSample * self.clusterCenterRatio_f)
        n = self.fdn
        # n = 7
        # print(f"density peak center:{n}")
        self.cluster2(n=n)

    def cluster2(self, n):
        '''
        按照给定的簇中心个数进行聚类
        :param n: 簇中心个数
        :return:
        '''
        # n=len(self.clusterCenter_l)
        # print(f"current centers:{self.clusterCenter_l}")
        # 初始化标签向量
        self.label_l = np.zeros(self.numSample, dtype=int)
        # 初始化聚类中心
        self.clusterCenter_l = np.argsort(self.representativeness_l)[-n:][::-1]
        # print(f"cluster centers:{self.clusterCenter_l}")
        # 初始化簇号 使用簇号作为聚类标签
        for i in range(n):
            self.label_l[self.clusterCenter_l[i]] = -i - 1

        # 统计聚类标签
        for i in range(self.numSample):
            if self.label_l[i] < 0:
                continue
            self.label_l[i] = self.getLabel(self.masters_l[i])

        # 初始化聚类簇块
        self.clusters_d = {key: [] for key in self.label_l[self.clusterCenter_l]}

        # 按照聚类结果划分簇块
        for i in self.label_l[self.clusterCenter_l]:
            self.clusters_d[i] += [j for j in range(self.numSample) if self.label_l[j] == i]

    @staticmethod
    def getDistanceByEuclid(instance1, instance2):
        '''
        按照欧氏距离计算实例间距离
        :param instance1: 实例1
        :param instance2: 实例2
        :return: 欧氏距离
        '''
        dist = 0
        for key in range(len(instance1)):
            dist += (float(instance1[key]) - float(instance2[key])) ** 2
        return dist ** 0.5

if __name__ == '__main__':
    print("What you are running is not the main of Run!!!")
