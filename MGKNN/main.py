import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from MGKNN import MGKNN
import os
import time
from sklearn.decomposition import PCA


# 获取文件路径
def through(lujing, key_word):
    file_list = []
    count = 0
    for root, dirs, files in os.walk(lujing):
        for file in files:
            name = os.path.join(root, file)
            if key_word in name:
                count += 1
                file_list.append(name)
    return file_list

if __name__ == '__main__':
    pathList = through(r".\datasets", ".csv")

    # 跑某些范围
    pathList = pathList[0:len(pathList)]

    for i, dataPath in enumerate(pathList):
        # 只跑某个文件时使用，跑所有数据将if注掉

        # if "Iris.csv" not in dataPath:
        #     continue

        dataName = dataPath.split("\\")[-1]
        print("------------------------")
        print(f"加载{dataName}....")

        df = pd.read_csv(dataPath, header=None)  # 加载数据集

        #取出数据集的数据和标签部分
        X = df.values[:, 1:]
        y = df.values[:, 0]

        #设置交叉验证kf
        kf = KFold(n_splits=10, random_state=2001, shuffle=True)
        curr_score = 0
        count = 0
        acc_sum = 0
        max_acc = 0
        min_acc = 1
        time_sum = 0

        for train_index, valid_index in kf.split(X):
            beginTime = time.time()
            aknn = MGKNN(X[train_index])
            adaptive_k = aknn.mgknn()
            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(X[train_index], y[train_index])
            results = []
            for predicte_data in X[valid_index]:
                #找到最近的一个邻居
                clf.n_neighbors = 1
                index = clf.kneighbors(X=[predicte_data])[1][0][0]

                #得到使用自然邻居估测的k值
                k = adaptive_k[index]
                #使用k值进行预测
                clf.n_neighbors = k
                result = clf.predict([predicte_data])[0]
                results.append(result)
            acc = sklearn.metrics.accuracy_score(y[valid_index],results)
            endTime = time.time()
            time_sum += endTime - beginTime
            if acc > max_acc:
                max_acc = acc
            if acc < min_acc:
                min_acc=acc
            count += 1
            acc_sum += acc
