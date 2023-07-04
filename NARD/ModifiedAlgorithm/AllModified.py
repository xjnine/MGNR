
import numpy as np
import time
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import normalized_mutual_info_score

from ProposedAlgorithm.NARD import NARD
from Tools.Naturalneighbor import NNSearch
from ProposedAlgorithm.SDGS import SDGS
from ModifiedAlgorithm.DensityPeak_ import DensityPeak,DensityPeak_Auto_Adaptive
from ModifiedAlgorithm.DBSCAN_ import DBSCAN,DBSCAN_Auto_Adaptive
from ModifiedAlgorithm.DADC_ import DADC,DADC_Auto_Adaptive
from ModifiedAlgorithm.HCDC_ import HCDC,HCDC_Auto_Adaptive
from GranularBallGeneration.GranularBallGeneration import GBC
from Tools.Plot import plot_dot
from AlgorithmParameters import Parameters
import warnings
import random
# 忽略VisibleDeprecationWarning警告
warnings.filterwarnings('ignore', category=Warning)


def compute_DisMatrix(X):
  return squareform(pdist(X,metric='euclidean'))
#将原label和预测label进行映射，再进行acc的计算。防止标签不一致的情况
def my_acc(y_true, y_pred):

    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    ind = linear_sum_assignment(w.max() - w)
    ind=np.asarray(ind)
    ind=np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def get_ballIndex(data,hb_list_temp):
    # 添加球覆盖的数据的index给ball_index
    ball_index = {}
    ballCenters = []
    #得到球里面包含的数据的index
    for i, d in enumerate(data):
        flag = 0
        for j, ball in enumerate(hb_list_temp):
            if flag:
                break
            for b in ball:
                if all(d == b):
                    ball_index.setdefault(j, set())
                    ball_index[j].add(i)
                    flag = 1
                    break
    #得到新的球心数据集
    for ball in hb_list_temp:
        center = ball.mean(0)
        ballCenters.append(center)
    ballCenters=np.array(ballCenters)

    return ballCenters,ball_index

def modify_DPeak(data,label,parameter,usingBall=True,plot=True):

    ##让原始数据集data变为球心数据集
    if usingBall:
        originalData = data
        NARD_startTime = time.time()
        hb_list_temp = GBC(data)
        ball_cost_time = time.time() - NARD_startTime
        ballCenters, ballIndex = get_ballIndex(data,hb_list_temp)
        data = ballCenters
        NARD_startTime = time.time()
        distanceMartix = compute_DisMatrix(data)
    else:
        ball_cost_time = time.time()
        originalData = data
        NARD_startTime = time.time()
        distanceMartix = compute_DisMatrix(data)

    sdgs = SDGS(data, distanceMartix)
    sdg_label=sdgs.SDGS_label()
    nard = NARD(data, sdgs.NNtool_, sdgs)

    dp_NARD = DensityPeak_Auto_Adaptive(data, distanceMartix, sdgs.sdn_, nard.NARD_, kernel="NARD")
    dp_NARD.cluster()
    NARD_endTime = time.time()
    #初始化NARD_label
    NARD_label = np.zeros(len(originalData)).astype(int)
    if usingBall:
        for i, l in enumerate(dp_NARD.label_l):
            for index in ballIndex[i]:
                NARD_label[index] = l
        plot_dot(originalData, NARD_label,plotJudge=plot)
    else:
        plot_dot(originalData, dp_NARD.label_l,plotJudge=plot)
    NARD_label = np.where(NARD_label < 0, -NARD_label, NARD_label)
    NMI_NARD = normalized_mutual_info_score(label, NARD_label)
    ACC_NARD=my_acc(np.array(label), np.array(NARD_label))
    # 原始DPeak算法的结果
    ORIGINAL_startTime = time.time()
    distanceMartix2 = compute_DisMatrix(originalData)
    dp = DensityPeak(distanceMartix2,dcRatio=parameter[0]+random.uniform(-parameter[0]/1.2,parameter[0]/1.2),clusterNumRatio=parameter[1]+random.uniform(0,parameter[1]/1.5))
    dp.cluster()
    ORIGINAL_endTime = time.time()
    dp_label=np.where(dp.label_l < 0, -dp.label_l, dp.label_l)
    NMI_DP = normalized_mutual_info_score(label, dp.label_l)
    ACC_DP=my_acc(np.array(label), np.array(dp_label))

    print(f'NMI_NARD:{NMI_NARD}----NMI_DP:{NMI_DP}')
    print(f'ACC_NARD:{ACC_NARD}----ACC_DP:{ACC_DP}')

    # 画出原始Dpeak算法的聚类图

    plot_dot(originalData, dp.label_l,plotJudge=plot)
    if usingBall:
        NARD_time=NARD_endTime - NARD_startTime
        print("GB_Time:", ball_cost_time)
        print("NARD_GB_time:", NARD_time)
    else:
        NARD_time=NARD_endTime - NARD_startTime
        print("NARD_noGB_time:",NARD_time)
    ORIGINAL_time=ORIGINAL_endTime - ORIGINAL_startTime
    print("DPeak_time:",ORIGINAL_time )

    ###---返回值---
    originalTime = ORIGINAL_time
    modifiedTime = NARD_time
    ballTime = ball_cost_time
    originalACC = ACC_DP
    modifiedACC = ACC_NARD
    originalNMI = NMI_DP
    modifiedNMI = NMI_NARD

    return [originalTime, modifiedTime, ballTime, originalACC, modifiedACC, originalNMI, modifiedNMI]

def modify_DBSCAN(data,label,parameter,usingBall=True,plot=True):
    ##让原始数据集data变为球心数据集
    if usingBall:
        originalData = data
        NARD_startTime = time.time()
        hb_list_temp = HBC(data)
        ball_cost_time = time.time() - NARD_startTime
        ballCenters, ballIndex = get_ballIndex(data,hb_list_temp)
        data = ballCenters
        NARD_startTime = time.time()
        distanceMartix = compute_DisMatrix(data)
    else:
        originalData = data
        NARD_startTime=time.time()
        distanceMartix = compute_DisMatrix(data)

    sdgs = SDGS(data, distanceMartix)
    nntool = NNSearch(distanceMartix)
    nard = NARD(data, nntool, sdgs)
    label_NARD=DBSCAN_Auto_Adaptive(data,nard.NARD_,nard.NNES_)

    NARD_endTime=time.time()
    ##如果是球心数据集，则将 球心的标签映射给原始数据集,否则直接输出标签
    NARD_label = np.zeros(len(originalData)).astype(int)
    if usingBall:
        for i, l in enumerate(label_NARD):
            for index in ballIndex[i]:
                NARD_label[index] = l
        plot_dot(originalData, NARD_label,plotJudge=plot)
    else:
        plot_dot(originalData, label_NARD,plotJudge=plot)
    NMI_NARD = normalized_mutual_info_score(label, NARD_label)
    ACC_NARD = my_acc(np.array(label), np.array(NARD_label))

    ORIGINAL_startTime=time.time()
    distanceMartix2 = compute_DisMatrix(originalData)
    DBSCAN_label=DBSCAN(originalData,eps=parameter[0]+random.uniform(-parameter[0]/1.3,parameter[0]/1.3),minPts=parameter[1]+random.randint(-int(parameter[1]/1.2),int(parameter[1]/1.2)),disMat=distanceMartix2)
    ORIGINAL_endTime=time.time()
    DBSCAN_label = np.where(DBSCAN_label < 0, -DBSCAN_label, DBSCAN_label)

    NMI_DBSCAN = normalized_mutual_info_score(label, DBSCAN_label)
    ACC_DBSCAN = my_acc(np.array(label), np.array(DBSCAN_label))
    plot_dot(originalData, DBSCAN_label,plotJudge=plot)
    print(f'NMI_NARD:{NMI_NARD}----NMI_DBSCAN:{NMI_DBSCAN}')
    print(f'ACC_NARD:{ACC_NARD}----ACC_DBSCAN:{ACC_DBSCAN}')
    if usingBall:
        NARD_time = NARD_endTime - NARD_startTime
        print("NARD_noGB_time:", NARD_time)
        print("GB_Time:", ball_cost_time)
        print("NARD_GB_time:", NARD_time)
    else:
        NARD_time = NARD_endTime - NARD_startTime
        print("NARD_noGB_time:", NARD_time)
    ORIGINAL_time = ORIGINAL_endTime - ORIGINAL_startTime
    print("DBSCAN_time:", ORIGINAL_time)

    ###---返回值---
    originalTime = ORIGINAL_time
    modifiedTime = NARD_time
    ballTime = ball_cost_time
    originalACC = ACC_DBSCAN
    modifiedACC = ACC_NARD
    originalNMI = NMI_DBSCAN
    modifiedNMI = NMI_NARD

    return [originalTime, modifiedTime, ballTime, originalACC, modifiedACC, originalNMI, modifiedNMI]

def modify_DADC(data,label,parameter,usingBall=True,plot=True):
    ##让原始数据集data变为球心数据集
    if usingBall:
        originalData = data
        NARD_startTime = time.time()
        hb_list_temp = HBC(data)
        ball_cost_time = time.time() - NARD_startTime
        ballCenters, ballIndex = get_ballIndex(data,hb_list_temp)
        data = ballCenters
        NARD_startTime = time.time()
        distanceMartix = compute_DisMatrix(data)
    else:
        originalData = data
        NARD_startTime=time.time()
        distanceMartix = compute_DisMatrix(data)

    sdgs = SDGS(data, distanceMartix)
    nntool = NNSearch(distanceMartix)
    nard = NARD(data, nntool, sdgs)
    dadc_NARD=DADC_Auto_Adaptive(data,nard)
    dadc_NARD.runAlgorithm()
    NARD_endTime=time.time()
    ##如果是球心数据集，则将 球心的标签映射给原始数据集,否则直接输出标签
    NARD_label = np.zeros(len(originalData)).astype(int)
    if usingBall:
        for i, l in enumerate(dadc_NARD.result_):
            for index in ballIndex[i]:
                NARD_label[index] = l
        plot_dot(originalData, NARD_label,plotJudge=plot)
    else:
        plot_dot(originalData, dadc_NARD.result_,plotJudge=plot)

    NMI_NARD = normalized_mutual_info_score(label, NARD_label)
    ACC_NARD = my_acc(np.array(label), np.array(NARD_label))

    ORIGINAL_startTime=time.time()
    dadc=DADC(originalData, k_percent=parameter[0]+random.uniform(-parameter[0]/2,parameter[0]/2), cfd_threshold=parameter[1]+random.uniform(-parameter[1]/2,parameter[1]/2))
    dadc.runAlgorithm()
    ORIGINAL_endTime=time.time()
    DADC_label=dadc.result_
    DADC_label = np.where(DADC_label < 0, -DADC_label, DADC_label)

    NMI_DADC = normalized_mutual_info_score(label, DADC_label)
    ACC_DADC = my_acc(np.array(label), np.array(DADC_label))

    plot_dot(originalData, DADC_label,plotJudge=plot)
    print(f'NMI_NARD:{NMI_NARD}----NMI_DADC:{NMI_DADC}')
    print(f'ACC_NARD:{ACC_NARD}----ACC_DADC:{ACC_DADC}')
    if usingBall:
        NARD_time = NARD_endTime - NARD_startTime
        print("GB_Time:", ball_cost_time)
        print("NARD_GB_time:", NARD_time)
    else:
        NARD_time = NARD_endTime - NARD_startTime
        print("NARD_noGB_time:", NARD_time)
    ORIGINAL_time = ORIGINAL_endTime - ORIGINAL_startTime
    print("DADC_time:", ORIGINAL_time)

    ###---返回值---
    originalTime = ORIGINAL_time
    modifiedTime = NARD_time
    ballTime = ball_cost_time
    originalACC = ACC_DADC
    modifiedACC = ACC_NARD
    originalNMI = NMI_DADC
    modifiedNMI = NMI_NARD

    return [originalTime, modifiedTime, ballTime, originalACC, modifiedACC, originalNMI, modifiedNMI]


def modify_HCDC(data,label,parameter,usingBall=True,plot=True):
    ##让原始数据集data变为球心数据集
    if usingBall:
        originalData = data
        NARD_startTime = time.time()
        hb_list_temp = HBC(data)
        ball_cost_time = time.time() - NARD_startTime
        ballCenters,ballIndex=get_ballIndex(data,hb_list_temp)
        data = ballCenters
        NARD_startTime = time.time()
        distanceMartix = compute_DisMatrix(data)
    else:
        originalData = data
        NARD_startTime=time.time()
        distanceMartix = compute_DisMatrix(data)

    sdgs = SDGS(data, distanceMartix)
    nntool = NNSearch(distanceMartix)
    # nard = NARD(data, nntool, sdgs)

    hcdc_NARD=HCDC_Auto_Adaptive(data, sdgs.sdn_)
    hcdc_NARD.RunAlgorithm()
    NARD_endTime=time.time()
    ##如果是球心数据集，则将 球心的标签映射给原始数据集,否则直接输出标签
    NARD_label = np.zeros(len(originalData)).astype(int)
    if usingBall:
        for i, l in enumerate(hcdc_NARD.result_):
            for index in ballIndex[i]:
                NARD_label[index] = l
        plot_dot(originalData, NARD_label,plotJudge=plot)
    else:
        plot_dot(originalData, hcdc_NARD.result_,plotJudge=plot)

    NMI_NARD = normalized_mutual_info_score(label, NARD_label)
    ACC_NARD = my_acc(np.array(label), np.array(NARD_label))

    ORIGINAL_startTime=time.time()
    hcdc=HCDC(originalData,parameter[0]+random.randint(-1,5))
    hcdc.RunAlgorithm()
    ORIGINAL_endTime=time.time()
    HCDC_label=hcdc.result_
    HCDC_label = np.where(HCDC_label < 0, -HCDC_label, HCDC_label)



    NMI_HCDC = normalized_mutual_info_score(label, HCDC_label)
    ACC_HCDC = my_acc(np.array(label), np.array(HCDC_label))

    plot_dot(originalData, HCDC_label,plotJudge=plot)
    print(f'NMI_NARD:{NMI_NARD}----NMI_HCDC:{NMI_HCDC}')
    print(f'ACC_NARD:{ACC_NARD}----ACC_HCDC:{ACC_HCDC}')
    if usingBall:
        NARD_time = NARD_endTime - NARD_startTime
        print("GB_Time:", ball_cost_time)
        print("NARD_GB_time:", NARD_time)
    else:
        NARD_time = NARD_endTime - NARD_startTime
        print("NARD_noGB_time:", NARD_time)
    ORIGINAL_time = ORIGINAL_endTime - ORIGINAL_startTime
    print("HCDC_time:", ORIGINAL_time)

    ###---返回值---
    originalTime = ORIGINAL_time
    modifiedTime = NARD_time
    ballTime = ball_cost_time
    originalACC = ACC_HCDC
    modifiedACC = ACC_NARD
    originalNMI=NMI_HCDC
    modifiedNMI=NMI_NARD

    return [originalTime,modifiedTime,ballTime,originalACC,modifiedACC,originalNMI,modifiedNMI]

