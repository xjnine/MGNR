
import numpy as np
import copy
from Tools.Naturalneighbor import NNSearch

##Sample Distribution
class SampleDistribution:
    def __init__(self, data, label,dataIndex):
        self.data = data
        self.dataIndex = dataIndex
        self.center = self.data.mean(0)
        self.label = label
        self.num = len(data)
        self.baryCenter=self.get_bary_center()
    # 获得样本分布的质心
    def get_bary_center(self):
        return self.dataIndex[np.argsort(((self.data - self.center) ** 2).sum(axis=1) ** 0.5)[0]]

##Sample Distribution Characterizing
class SDGS:
    def __init__(self,data,distanceMatrix):
        self.data_=data
        self.sdn_ = 0  #sample distribution number
        self.distanceMatrix_=distanceMatrix
        self.NNtool_ = NNSearch(distanceMatrix)
        self.MGN_=self.MGN_generation()
        self.SDGS_=self.SDGS_generation()

    def SDGS_label(self):
        labels=np.zeros(len(self.data_),dtype=int)
        for label,fdg in self.SDGS_.items():
            for index in fdg.dataIndex:
                labels[index]=label
        return list(labels)

    def MGN_generation(self):

        # 获取邻居和反邻居的交集列表
        intersectionList = []
        for i,nan in enumerate(self.NNtool_.NaN):
            temp=set()
            temp|=nan|{i}
            ithSet =temp
            intersectionList.append(ithSet)
        MGN=intersectionList
        return MGN

    ##Sample Distribution Group Set
    def SDGS_generation(self):
        mgnTemp = copy.deepcopy(self.MGN_)
        ## inner function for merging
        def merge(MGN):
            sdgs = {}
            iterateList = [False] * len(MGN)
            DistributionCount = 0
            for i in range(len(iterateList)):
                if not iterateList[i]:
                    sdgs.setdefault(DistributionCount, MGN[i])
                    for j in range(i, len(iterateList)):
                        # NNG[i] and NNG[j] are judged. If the number of their intersection is greater than 1, the two NNGs are merged and the iteration labels of the two NNGs are set to True.
                        if len(sdgs[DistributionCount] & MGN[j]) > 1 and not iterateList[j]:
                            iterateList[j] = True
                            sdgs[DistributionCount] |= MGN[j]
                    DistributionCount += 1
            return sdgs

        ## Using merging criterion to merge the NNG
        while True:
            sdgs = merge(mgnTemp)
            if len(sdgs) == len(mgnTemp):
                break
            mgnTemp = sdgs

        # formation of data Distribution Group
        SDGS = {}
        count = 0
        for value in sdgs.values():
            realData = []
            for v in value:
                realData.append(self.data_[v])
            realData = np.array(realData)
            SDGS[count] = SampleDistribution(realData, count, list(value))
            count += 1


        self.sdn_ = len(SDGS)

        return SDGS