
import os

import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler
from Tools.Plot import plot_dot


def through(path, keyWord=''):
    file_list = []
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            name = os.path.join(root, file)
            if keyWord in name:
                count += 1
                file_list.append(name)
    return file_list


def loadSyntheticData(datasetPath,datasetKey):
    datasetsPath = through(datasetPath)

    for path in datasetsPath:
        if datasetKey not in path:
            continue
        if ".arff" not in path:
            df = pd.read_csv(path,header=None)
        else:
            try:
                data = arff.loadarff(path)
                print(f"读取{path}...")
                df = pd.DataFrame(data[0])
            except:
                print(f"{path}读取失败")
                continue

        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(df.values[:, 0:2])
        label = df.values[:, -1]
        # plot_dot(data,datasetColor='#314300')
        return data,label

def loadRealData(datasetPath,datasetKey):
    datasetsPath = through(datasetPath)

    for path in datasetsPath:
        if datasetKey not in path:
            continue
        if ".arff" not in path:
            df = pd.read_csv(path,header=None)
        else:
            try:
                data = arff.loadarff(path)
                print(f"读取{path}...")
                df = pd.DataFrame(data[0])
            except:
                print(f"{path}读取失败")
                continue

        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(df.values[:, 1:])
        label = df.values[:, 0]
        # plot_dot(data,datasetColor='#314300')
        return data,label