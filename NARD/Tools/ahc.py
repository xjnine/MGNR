import csv
import math
import operator
import sys
import ast
from random import randint
import numpy as np
class Data():
    __slots__ = ["point","Rep_num"]
    def __init__(self,point=[],Rep_num = 0):
        self.point = point
        self.Rep_num = Rep_num
class Cluster():
    def __init__(self):
        self.clusters = []#每个类里的点
        self.clusters_index = []#每个类的点的下标
        # self.index = []#类号
        self.number = None
        self.point_num = []

def inputPoints(Points,Contain_point):#输入所有数据矩阵
    size = len(Points)
    points = [Data() for _ in np.arange(size)]
    for index , p in enumerate(points):
        p.point = Points[index]
        p.Rep_num = len(Contain_point[index])
    return points

def merge(cluster,GD):
    avg = find_min(cluster.clusters_index[0],cluster.clusters_index[1],GD)
    index_list = cluster.clusters_index[0] + cluster.clusters_index[1]
    rm_index_1 = 1
    rm_index_2 = 0

    for i,clus in enumerate(cluster.clusters_index):
        for j in range(cluster.number):
            dis = find_min(clus,cluster.clusters_index[j],GD)
            if dis < avg and i != j:
                avg = dis
                index_list = cluster.clusters_index[i] + cluster.clusters_index[j]
                if i < j:
                    rm_index_1 = j
                    rm_index_2 = i
                else:
                    rm_index_1 = i
                    rm_index_2 = j
    new_clus = cluster.clusters[rm_index_1] + cluster.clusters[rm_index_2]
    Rep_num_sum = cluster.point_num[rm_index_1][0] + cluster.point_num[rm_index_2][0]
    cluster.clusters_index.append(index_list)
    cluster.clusters.append(new_clus)
    cluster.point_num.append([Rep_num_sum])
    del cluster.clusters[rm_index_1]
    del cluster.clusters[rm_index_2]
    del cluster.clusters_index[rm_index_1]
    del cluster.clusters_index[rm_index_2]
    del cluster.point_num[rm_index_1]
    del cluster.point_num[rm_index_2]
    cluster.number -= 1


def find_min(cluster1,cluster2,GD):
    min = np.inf
    for point1 in cluster1:
        for point2 in cluster2:
            dis = GD[point1,point2]
            if dis < min:
                min = dis
    return min

def find_max(cluster1,cluster2,GD):
    max = 0
    for point1 in cluster1:
        for point2 in cluster2:
            dis = GD[point1,point2]
            if dis > max:
                max = dis
    return max

def distance(point1, point2):
    dis = 0.0
    # eliminate the class attribute
    for i in range(len(point1)-1):
        add = (float(point1[i]) - float(point2[i]))**2
        dis += add
    return dis**0.5
def find_avg(cluster1,cluster2,GD):
    avg = 0
    count = 0
    for point1 in cluster1:
        for point2 in cluster2:
            avg += GD[point1,point2]
            count += 1
    return avg / count

def ahc(P,GD,k,contain_point,size):
    P = P.tolist()
    points = inputPoints(P,contain_point)
    cluster = Cluster()
    cluster.number = len(P)
    for i,d in enumerate(points):#points是一个有很多Data类型的点的列表，而不是一整个Data数据结构
        cluster.clusters.append([d.point])#每个类包含的点的坐标
        cluster.clusters_index.append([i])#每个类包含的点的下标
        # cluster.index.append([i])#每个类的类号
        cluster.point_num.append([d.Rep_num])
    while cluster.number > k:
        merge(cluster,GD)
    i = 0
    # print(cluster.clusters_index)
    # print(cluster.point_num)
    while i < len(cluster.clusters_index):#把点少的当成噪声类
        if cluster.point_num[i][0] < 0.01 * size:
            del cluster.clusters_index[i]
            del cluster.point_num[i]
        else:
            i += 1
    # print(cluster.clusters_index)
    # print(cluster.point_num)
    return cluster
