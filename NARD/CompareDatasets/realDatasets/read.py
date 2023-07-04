"""
@Time : 2023/5/15 19:09
@Author : xiang xuexin
@File : read.py
@Software: PyCharm
"""
from Tools.LoadData import loadRealData,loadSyntheticData
data_dict = ["D1.csv", "D2.csv", "D3.csv", "D4.csv", "D5.csv", "D6.csv","cell.csv"]

for dataSet in data_dict:
    # if dataSet != "haberman.csv":
    #     continue
    data, label = loadSyntheticData("../syntheticDatasets", datasetKey=dataSet)
    print(f"{dataSet}---Observations:{len(data)}---Dimensions:{len(data[0])}---Classes:{len(set(label))}")