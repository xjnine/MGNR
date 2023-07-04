
import numpy as np

from sklearn.metrics import adjusted_mutual_info_score,rand_score,fowlkes_mallows_score,adjusted_rand_score,normalized_mutual_info_score


import time
from Tools import ahc

class HCDC:
    def __init__(self,points,cluster_num):
        self.points_=points
        self.cluster_num_=cluster_num
        self.result_=[]

    def RunAlgorithm(self):

        size = self.points_.shape[0]
        dis = self.get_Distance(self.points_)
        dist = np.argsort(dis,axis=1)
        r,KNN,RNN,NN = self.NN_Search(dist)
        for i in range(size):
            KNN[i].append(i)
        Den = self.get_Density(dis,NN,KNN)

        Rep,cores,Contain_point = self.get_Rep_and_core2(KNN, Den,NN,dis)

        NND_childs = self.get_NNDchilds(cores, Rep, NN)
        GD = self.compute_similartiy(cores, Den, NND_childs,dis,Rep)
        cluster = ahc.ahc(self.points_[cores],GD, self.cluster_num_,Contain_point,size)
        CL = np.zeros(size)
        for i in range(len(cluster.clusters_index)):
            for j in cluster.clusters_index[i]:
                CL[cores[j]] = i + 1
        CL = self.assign_points(Rep,CL)
        self.result_=np.array(CL).astype(int)

        # print("用时",end-start)
        # fmi = lambda x,y:fowlkes_mallows_score(x,y)
        # nmi = lambda x,y:normalized_mutual_info_score(x,y)
        # ari = lambda x,y:adjusted_rand_score(x,y)
        # print("%.3f,%.3f,%.3f"%(nmi(label,CL),ari(label,CL),fmi(label,CL)))

        # self.pf.printScatter_Color(points,CL)

    def get_Distance(self,points):
        size = points.shape[0]
        dis = np.zeros((size,size))
        for i in range(size):
            for j in range(i+1,size):
                dd = np.linalg.norm(points[i] - points[j])
                dis[i,j] = dd
                dis[j,i] = dd
        return dis

    def NN_Search(self,dist):
        size = len(dist)
        r = 1
        flag = 0
        nb = np.zeros(size)
        count = 0
        count1 = 0
        count2 = 0
        KNN = []
        RNN = []
        for i in range(size):
            KNN.append([])
            RNN.append([])
        while flag == 0: 
            count2 = 0
            NN = []
            for i in range(size):
                k = dist[i, r]
                KNN[i].append(k)
                RNN[k].append(i)
                nb[k] += 1
            for i in range(size):
                NN.append(list(set(KNN[i])&set(RNN[i])))
                if nb[i] == 0:
                    count2 += 1
            r = r + 1
            if count1 == count2:
                count += 1
            else:
                count = 1
            if count2 == 0 or count >= 2:
                flag = 1
            count1 = count2
        neigh_bor_sum = r - 1
        return (neigh_bor_sum,KNN,RNN,NN)

    def get_Density(self,dis,NN,KNN):
        size = len(NN)
        Den = np.zeros(size)
        for i in range(size):
            for j in NN[i]:
                SNN = list(set(KNN[i]) & set(KNN[j]))
                if len(SNN) != 0:
                    Den[i] += (len(SNN))**2 / ((np.sum(dis[i,SNN]) + np.sum(dis[j,SNN])) * (dis[i,j]+0.01))
        return Den



    def get_Rep_and_core2(self,KNN,Den,NN,dis):
        size = len(Den)
        Rep = np.ones(size).astype(int) * -1
        Den_argindex = np.argsort(Den)
        cores = []
        Rep_ = []
        for i in range(size):
            Rep_.append([])
        #在每个点的K近邻（包括自己）选择一个密度最大的点作为整个KNN的代表点，若是某点已经有了代表点，
        #则计算已有的代表点的密度与本点的密度差和两点之间距离和的乘积当前计算的，小的为新的代表点
        for i in Den_argindex:
            max_index = KNN[i][np.argmax(Den[KNN[i]])]
            max_Den = np.max(Den[KNN[i]])
            max_index = i if Den[i] > max_Den else max_index
            for j in KNN[i]:
                Rep_[j].append(max_index)
        for i in range(size):
            s = list(set(Rep_[i]))
            max_rep = -1
            max_rep_count = 0
            for j in s:
                num = Rep_[i].count(j)
                if num > max_rep_count:
                    max_rep = j
                    max_rep_count = num
                elif num == max_rep_count:
                    if dis[i,j] * np.abs(Den[i] - Den[j])  < dis[i,max_rep] * np.abs(Den[i] - Den[max_rep]):
                        max_rep = j
                        max_rep_count = num
            Rep[i] = max_rep
        visited = np.zeros(size)
        round = 0
        for i in range(size):
            if visited[i] == 0:
                parent = i
                round += 1
                while Rep[parent] != parent:
                    visited[parent] = round
                    parent = Rep[parent]
                Rep[np.where(visited == round)[0]] = parent
        for i in range(size):
            if Rep[i] == i:
                cores.append(i)
        index = np.zeros(size).astype(int)
        k = 0
        for i in cores:
            index[i] = k
            k += 1
        delete_cores = []
        NND_childs = self.get_NNDchilds(cores,Rep,NN)
        d = {}
        k = 0
        for i in range(len(cores)):
            d[cores[i]] = k
            k += 1
        index = np.argsort(Den[cores]).astype(int)
        cores_sort = []
        for i in index:
            cores_sort.append(cores[i])
        for i in cores_sort:
            if i in delete_cores:
                continue
            else:
                for j in cores_sort:
                    if i in delete_cores:
                        break
                    if j in delete_cores:
                        continue
                    else:
                        if j in NND_childs[d[i]] or i in NND_childs[d[j]]:
                                if Den[i] < Den[j]:
                                    Rep[np.where(Rep == i)[0]] = j
                                    cores.remove(i)
                                    delete_cores.append(i)
                                    NND_childs[d[i]] = list(set(NND_childs[d[i]]).union(set(NND_childs[d[j]])))
                                    break
                                if Den[i] > Den[j]:
                                    Rep[np.where(Rep == j)[0]] = i
                                    cores.remove(j)
                                    NND_childs[d[j]] = list(set(NND_childs[d[i]]).union(set(NND_childs[d[j]])))
                                    delete_cores.append(j)
                                    continue

        Contain_point = []
        for i in range(len(cores)):
            Contain_point.append(np.where(Rep==cores[i])[0])
        return Rep, cores,Contain_point


    #将每个核心点的NND_childs看做是以他们自己为代表点的点的NN近邻并集
    def get_NNDchilds(self,cores,Rep,NN):
        NND_childs = []
        for i in range(len(cores)):
            NND_child = list(np.where(Rep==cores[i])[0])
            temp = []
            if len(NND_child) != 1:
                for j in NND_child:
                    temp = set(temp).union(set(NN[j]))
            else:
                temp = set(NN[NND_child[0]]).union(set(NN[NND_child[0]]))
            NND_childs.append(list(set(NND_child).union(temp)))
        return NND_childs



    def compute_similartiy(self,cores,Den,NND_childs,dis,Rep):
        length = len(cores)
        similarity = np.zeros((length,length))
        for i in range(length):
            similarity[i,i] = 0
        maxd = 0
        for i in range(length):
            for j in range(i+1,length):
                if maxd < dis[cores[i],cores[j]]:
                    maxd = dis[cores[i],cores[j]]
        max_p = 1
        for i in range(length):
            for j in range(i+1,length):
                common = list(set(NND_childs[i]) & set(NND_childs[j]))

                rep_i = np.where(Rep==cores[i])[0]
                rep_j = np.where(Rep==cores[j])[0]
                mean_Deni = np.mean(Den[rep_i])
                mean_Denj = np.mean(Den[rep_j])
                if len(common) != 0:
                    dd = dis[cores[i], cores[j]] * max(len(NND_childs[i]),len(NND_childs[j])) * np.abs(mean_Deni + mean_Denj) / ((len(common) ** 2) * (2 * np.sqrt(mean_Deni * mean_Denj)))

                    max_p = max(max_p,len(NND_childs[i]) * len(NND_childs[j]) / (len(common) ** 2))
                    similarity[i, j] = dd
                    similarity[j, i] = dd
        for i in range(length):
            for j in range(i + 1, length):
                common = list(set(NND_childs[i]) & set(NND_childs[j]))
                if len(common) == 0:
                    dd = (max_p) * dis[cores[i],cores[j]] * maxd
                    similarity[i, j] = dd
                    similarity[j, i] = dd
        return similarity


    #把剩余的点分配给他们的代表点
    def assign_points(self,Rep,CL):
        size = CL.shape[0]
        for i in range(size):
            if CL[i] == 0:
                CL[i] = CL[Rep[i]]
        return CL
    def min_max(self,data):
        size,dim = data.shape
        for j in range(dim):
            max = np.max(data[:,j])
            min = np.min(data[:,j])
            for i in range(size):
                data[i,j] = (data[i,j] - min) / (max - min)
        return data


class HCDC_Auto_Adaptive:
    def __init__(self, points, cluster_num):
        self.points_ = points
        self.cluster_num_ = cluster_num
        self.result_ = []

    def RunAlgorithm(self):

        size = self.points_.shape[0]
        dis = self.get_Distance(self.points_)
        dist = np.argsort(dis, axis=1)
        r, KNN, RNN, NN = self.NN_Search(dist)
        for i in range(size):
            KNN[i].append(i)
        Den = self.get_Density(dis, NN, KNN)
        Rep, cores, Contain_point = self.get_Rep_and_core2(KNN, Den, NN, dis)
        NND_childs = self.get_NNDchilds(cores, Rep, NN)
        GD = self.compute_similartiy(cores, Den, NND_childs, dis, Rep)
        cluster = ahc.ahc(self.points_[cores], GD, self.cluster_num_, Contain_point, size)
        CL = np.zeros(size)
        for i in range(len(cluster.clusters_index)):
            for j in cluster.clusters_index[i]:
                CL[cores[j]] = i + 1
        CL = self.assign_points(Rep, CL)
        self.result_=np.array(CL).astype(int)
        # print("用时",end-start)
        # fmi = lambda x,y:fowlkes_mallows_score(x,y)
        # nmi = lambda x,y:normalized_mutual_info_score(x,y)
        # ari = lambda x,y:adjusted_rand_score(x,y)
        # print("%.3f,%.3f,%.3f"%(nmi(label,CL),ari(label,CL),fmi(label,CL)))

        # self.pf.printScatter_Color(points,CL)

    def get_Distance(self, points):
        size = points.shape[0]
        dis = np.zeros((size, size))
        for i in range(size):
            for j in range(i + 1, size):
                dd = np.linalg.norm(points[i] - points[j])
                dis[i, j] = dd
                dis[j, i] = dd
        return dis

    def NN_Search(self, dist):
        size = len(dist)
        r = 1
        flag = 0
        nb = np.zeros(size)
        count = 0
        count1 = 0
        count2 = 0
        KNN = []
        RNN = []
        for i in range(size):
            KNN.append([])
            RNN.append([])
        while flag == 0:
            count2 = 0
            NN = []
            for i in range(size):
                k = dist[i, r]
                KNN[i].append(k)
                RNN[k].append(i)
                nb[k] += 1
            for i in range(size):
                NN.append(list(set(KNN[i]) & set(RNN[i])))
                if nb[i] == 0:
                    count2 += 1
            r = r + 1
            if count1 == count2:
                count += 1
            else:
                count = 1
            if count2 == 0 or count >= 2:
                flag = 1
            count1 = count2
        neigh_bor_sum = r - 1
        return (neigh_bor_sum, KNN, RNN, NN)

    def get_Density(self, dis, NN, KNN):
        size = len(NN)
        Den = np.zeros(size)
        for i in range(size):
            for j in NN[i]:
                SNN = list(set(KNN[i]) & set(KNN[j]))
                if len(SNN) != 0:
                    Den[i] += (len(SNN)) ** 2 / ((np.sum(dis[i, SNN]) + np.sum(dis[j, SNN])) * (dis[i, j] + 0.01))
        return Den

    def get_Rep_and_core2(self, KNN, Den, NN, dis):
        size = len(Den)
        Rep = np.ones(size).astype(int) * -1
        Den_argindex = np.argsort(Den)
        cores = []
        Rep_ = []
        for i in range(size):
            Rep_.append([])
        # 在每个点的K近邻（包括自己）选择一个密度最大的点作为整个KNN的代表点，若是某点已经有了代表点，
        # 则计算已有的代表点的密度与本点的密度差和两点之间距离和的乘积当前计算的，小的为新的代表点
        for i in Den_argindex:
            max_index = KNN[i][np.argmax(Den[KNN[i]])]
            max_Den = np.max(Den[KNN[i]])
            max_index = i if Den[i] > max_Den else max_index
            for j in KNN[i]:
                Rep_[j].append(max_index)
        for i in range(size):
            s = list(set(Rep_[i]))
            max_rep = -1
            max_rep_count = 0
            for j in s:
                num = Rep_[i].count(j)
                if num > max_rep_count:
                    max_rep = j
                    max_rep_count = num
                elif num == max_rep_count:
                    if dis[i, j] * np.abs(Den[i] - Den[j]) < dis[i, max_rep] * np.abs(Den[i] - Den[max_rep]):
                        max_rep = j
                        max_rep_count = num
            Rep[i] = max_rep
        visited = np.zeros(size)
        round = 0
        for i in range(size):
            if visited[i] == 0:
                parent = i
                round += 1
                while Rep[parent] != parent:
                    visited[parent] = round
                    parent = Rep[parent]
                Rep[np.where(visited == round)[0]] = parent
        for i in range(size):
            if Rep[i] == i:
                cores.append(i)
        index = np.zeros(size).astype(int)
        k = 0
        for i in cores:
            index[i] = k
            k += 1
        delete_cores = []
        NND_childs = self.get_NNDchilds(cores, Rep, NN)
        d = {}
        k = 0
        for i in range(len(cores)):
            d[cores[i]] = k
            k += 1
        index = np.argsort(Den[cores]).astype(int)
        cores_sort = []
        for i in index:
            cores_sort.append(cores[i])
        for i in cores_sort:
            if i in delete_cores:
                continue
            else:
                for j in cores_sort:
                    if i in delete_cores:
                        break
                    if j in delete_cores:
                        continue
                    else:
                        if j in NND_childs[d[i]] or i in NND_childs[d[j]]:
                            if Den[i] < Den[j]:
                                Rep[np.where(Rep == i)[0]] = j
                                cores.remove(i)
                                delete_cores.append(i)
                                NND_childs[d[i]] = list(set(NND_childs[d[i]]).union(set(NND_childs[d[j]])))
                                break
                            if Den[i] > Den[j]:
                                Rep[np.where(Rep == j)[0]] = i
                                cores.remove(j)
                                NND_childs[d[j]] = list(set(NND_childs[d[i]]).union(set(NND_childs[d[j]])))
                                delete_cores.append(j)
                                continue

        Contain_point = []
        for i in range(len(cores)):
            Contain_point.append(np.where(Rep == cores[i])[0])
        return Rep, cores, Contain_point

    # 将每个核心点的NND_childs看做是以他们自己为代表点的点的NN近邻并集
    def get_NNDchilds(self, cores, Rep, NN):
        NND_childs = []
        for i in range(len(cores)):
            NND_child = list(np.where(Rep == cores[i])[0])
            temp = []
            if len(NND_child) != 1:
                for j in NND_child:
                    temp = set(temp).union(set(NN[j]))
            else:
                temp = set(NN[NND_child[0]]).union(set(NN[NND_child[0]]))
            NND_childs.append(list(set(NND_child).union(temp)))
        return NND_childs

    def compute_similartiy(self, cores, Den, NND_childs, dis, Rep):
        length = len(cores)
        similarity = np.zeros((length, length))
        for i in range(length):
            similarity[i, i] = 0
        maxd = 0
        for i in range(length):
            for j in range(i + 1, length):
                if maxd < dis[cores[i], cores[j]]:
                    maxd = dis[cores[i], cores[j]]
        max_p = 1
        for i in range(length):
            for j in range(i + 1, length):
                common = list(set(NND_childs[i]) & set(NND_childs[j]))

                rep_i = np.where(Rep == cores[i])[0]
                rep_j = np.where(Rep == cores[j])[0]
                mean_Deni = np.mean(Den[rep_i])
                mean_Denj = np.mean(Den[rep_j])
                if len(common) != 0:
                    dd = dis[cores[i], cores[j]] * max(len(NND_childs[i]), len(NND_childs[j])) * np.abs(
                        mean_Deni + mean_Denj) / ((len(common) ** 2) * (2 * np.sqrt(mean_Deni * mean_Denj)))

                    max_p = max(max_p, len(NND_childs[i]) * len(NND_childs[j]) / (len(common) ** 2))
                    similarity[i, j] = dd
                    similarity[j, i] = dd
        for i in range(length):
            for j in range(i + 1, length):
                common = list(set(NND_childs[i]) & set(NND_childs[j]))
                if len(common) == 0:
                    dd = (max_p) * dis[cores[i], cores[j]] * maxd
                    similarity[i, j] = dd
                    similarity[j, i] = dd
        return similarity

    # 把剩余的点分配给他们的代表点
    def assign_points(self, Rep, CL):
        size = CL.shape[0]
        for i in range(size):
            if CL[i] == 0:
                CL[i] = CL[Rep[i]]
        return CL

    def min_max(self, data):
        size, dim = data.shape
        for j in range(dim):
            max = np.max(data[:, j])
            min = np.min(data[:, j])
            for i in range(size):
                data[i, j] = (data[i, j] - min) / (max - min)
        return data


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    a = HCDC()
    a.RunAlgorithm()