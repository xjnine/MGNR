
import copy
import csv
import os
import time

import numpy as np
from sklearn.decomposition import PCA

from ModifiedAlgorithm.AlgorithmParameters import Parameters
from ModifiedAlgorithm.AllModified import modify_DPeak,modify_HCDC,modify_DADC,modify_DBSCAN
from Tools.LoadData import loadRealData,loadSyntheticData
##写csv文件，data格式要为[[],[],[]...[],[],[]],默认覆盖
from Tools.Plot import plot_dot



def write_csv(path,data,state='w'):
    with open(path, state, newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)


def synthetic_test():
    data_dict = ["D1.csv", "D2.csv", "D3.csv", "D4.csv", "D5.csv", "D6.csv"]
    # 结果表头originalTime,modifiedTime,ballTime,originalACC,modifiedACC,originalNMI,modifiedNMI

    experiment = {
        'D1': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype('float64'),
        'D2': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype('float64'),
        'D3': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype('float64'),
        'D4': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype('float64'),
        'D5': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype('float64'),
        'D6': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype('float64'),
        # 'cell': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype('float64'),
    }
    allResult = {
        'DPeak': copy.deepcopy(experiment),
        'DBSCAN': copy.deepcopy(experiment),
        'DADC': copy.deepcopy(experiment),
        'HCDC': copy.deepcopy(experiment),
    }
    #run_time 为实验跑的次数，取结果均值为最终实验结果
    run_time = 1
    for dataSet in data_dict:
        #跑指定数据集
        # if dataSet != "D1.csv":
        #     continue
        data, label = loadSyntheticData("../CompareDatasets", datasetKey=dataSet)
        dataSetName = dataSet.split('.')[0]
        print(f'{dataSet} ')
        # plot_dot(data, label, plotJudge=True)
        for i in range(run_time):
            print("DPeak:")
            Result = modify_DPeak(data, label, Parameters.DPeak[dataSetName], usingBall=True, plot=False)
            allResult['DPeak'][dataSetName] += Result
            print("DBSCAN:")
            Result=modify_DBSCAN(data,label,Parameters.DBSCAN[dataSetName],usingBall=True,plot=False)
            allResult['DBSCAN'][dataSetName] += Result
            print("DADC:")
            Result=modify_DADC(data,label,Parameters.DADC[dataSetName],usingBall=True,plot=False)
            allResult['DADC'][dataSetName] += Result
            print("HCDC:")
            Result=modify_HCDC(data,label,Parameters.HCDC[dataSetName],usingBall=True,plot=False)
            allResult['HCDC'][dataSetName]+=Result

        for algorithmName in allResult.keys():
            value = allResult[algorithmName][dataSetName]
            value /= run_time
            value = np.round(value, 3)
            allResult[algorithmName][dataSetName] = value

    parentPath = r"..\CompareDatasets\experimentResult\synthetic\\"
    for algorithmName in allResult.keys():
        writePath = parentPath + f"{algorithmName}.csv"
        if (os.path.exists(writePath)):
            os.remove(writePath)
        for datasetName in allResult[algorithmName]:
            #实验结果前面加一个数据集名称的头部
            RESULT = [i for i in allResult[algorithmName][datasetName]]
            RESULT.insert(0, datasetName)
            write_csv(writePath, [RESULT], state='a+')

def real_test():

    data_dict = [
        'cell.csv',
        'Automobile.csv',
        'Balance Scale.csv',
        'biodeg.csv',
        'Car Evaluation.csv',
        'chess.csv',
        'Credit Approval.csv',
        'Ionosphere.csv',
        'SpectfHeart.csv',
        'wdbc.csv',
        'WPBC.csv',

    ]
    # 结果表头originalTime,modifiedTime,ballTime,originalACC,modifiedACC,originalNMI,modifiedNMI
    # 初始化实验结果，将结果保存到指定目录
    experiment = {
        'cell':np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype('float64'),
        'Automobile':np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype('float64'),
        'Balance Scale':np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype('float64'),
        'biodeg':np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype('float64'),
        'Car Evaluation':np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype('float64'),
        'chess':np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype('float64'),
        'Credit Approval':np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype('float64'),
        'Ionosphere':np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype('float64'),
        'SpectfHeart':np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype('float64'),
        'wdbc':np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype('float64'),
        'WPBC':np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype('float64'),
        }

    allResult = {
        'DPeak': copy.deepcopy(experiment),
        'DBSCAN': copy.deepcopy(experiment),
        'DADC': copy.deepcopy(experiment),
        'HCDC': copy.deepcopy(experiment),
    }

    run_time = 1
    for dataSet in data_dict:
        if dataSet == "cell.csv":
            continue
        data, label = loadRealData("../CompareDatasets", datasetKey=dataSet)
        #将真实数据降维后聚类
        pca=PCA(n_components=2)
        data=pca.fit_transform(data)

        dataSetName = dataSet.split('.')[0]
        print(f'{dataSet} ')
        for i in range(run_time):
            print("DPeak")
            Result = modify_DPeak(data, label, Parameters.DPeak[dataSetName], usingBall=True, plot=False)
            allResult['DPeak'][dataSetName] += Result
            print("DBSCAN")
            Result=modify_DBSCAN(data,label,Parameters.DBSCAN[dataSetName],usingBall=True,plot=False)
            allResult['DBSCAN'][dataSetName] += Result
            print("DADC")
            Result=modify_DADC(data,label,Parameters.DADC[dataSetName],usingBall=True,plot=False)
            allResult['DADC'][dataSetName] += Result
            print("HCDC")
            Result=modify_HCDC(data,label,Parameters.HCDC[dataSetName],usingBall=True,plot=False)
            allResult['HCDC'][dataSetName]+=Result

        for algorithmName in allResult.keys():
            value = allResult[algorithmName][dataSetName]
            value /= run_time
            value = np.round(value, 3)
            allResult[algorithmName][dataSetName] = value

    parentPath = r"..\CompareDatasets\experimentResult\real\\"
    for algorithmName in allResult.keys():
        writePath = parentPath + f"{algorithmName}.csv"
        if (os.path.exists(writePath)):
            os.remove(writePath)
            #实验结果写上表头
            tableHead=['originalTime','modifiedTime','ballTime','originalACC','modifiedACC','originalNMI','modifiedNMI']
            write_csv(writePath, [tableHead], state='a+')
        for datasetName in allResult[algorithmName]:
            RESULT = [i for i in allResult[algorithmName][datasetName]]
            RESULT.insert(0, datasetName)
            write_csv(writePath, [RESULT], state='a+')
    print(time.ctime())


if __name__ == '__main__':
    synthetic_test()