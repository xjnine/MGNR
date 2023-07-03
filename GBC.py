import numpy as np
import matplotlib.pyplot as plt


class HB:
    def __init__(self, data, label):
        self.data = data
        self.center = self.data.mean(0)
        self.radius = self.get_radius()
        self.flag = 0
        self.label = label
        self.num = len(data)
        self.out = 0
        self.size = 1
        self.overlap = 0
        self.hardlapcount = 0
        self.softlapcount = 0

    def get_radius(self):
        return max(((self.data - self.center) ** 2).sum(axis=1) ** 0.5)


class UF:
    def __init__(self,len):
        self.parent = [0]*len
        self.size = [0]*len
        self.count = len

        for i in range(0,len):
            self.parent[i] = i
            self.size[i] = 1

    def find(self, x):
        while (self.parent[x] != x):
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if (rootP == rootQ):
            return
        if self.size[rootP] > self.size[rootQ]:
            self.parent[rootQ] = rootP
            self.size[rootP] += self.size[rootQ]
        else:
            self.parent[rootP] = rootQ
            self.size[rootQ] += self.size[rootP]
        self.count = self.count - 1


    def connected(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        return rootP == rootQ

    def count(self):
        return self.count


def division(hb_list, hb_list_not):
    gb_list_new = []
    for hb in hb_list:
        if len(hb) > 16:
            ball_1, ball_2 = spilt_ball(hb)
            dm_parent = get_dm(hb)
            dm_child_1 = get_dm(ball_1)
            dm_child_2 = get_dm(ball_2)
            w = len(ball_1) + len(ball_2)
            w1 = len(ball_1) / w
            w2 = len(ball_2) / w
            w_child = w1 * dm_child_1 + w2 * dm_child_2
            t2 = w_child < dm_parent
            if t2:
                gb_list_new.extend([ball_1, ball_2])
            else:
                hb_list_not.append(hb)
        else:
            hb_list_not.append(hb)
    return gb_list_new, hb_list_not


def spilt_ball(data):
    center = data.mean(0)
    ball1 = []
    ball2 = []
    dis1 = 0
    dis1i = 0
    dis2 = 0
    dis2i = 0
    for i in range(len(data)):
        if dis1 < (sum((data[i] - center) ** 2)):
            dis1 = sum((data[i] - center) ** 2)
            dis1i = i
    for i in range(len(data)):
        if dis2 < (sum((data[i] - data[dis1i]) ** 2)):
            dis2 = sum((data[i] - data[dis1i]) ** 2)
            dis2i = i
    for j in range(0, len(data)):
        if (sum((data[j] - data[dis1i]) ** 2)) < (sum((data[j] - data[dis2i]) ** 2)):
            ball1.extend([data[j]])
        else:
            ball2.extend([data[j]])
    ball1 = np.array(ball1)
    ball2 = np.array(ball2)
    return [ball1, ball2]

def get_dm(hb):
    num = len(hb)
    if num > 2:
        center = hb.mean(0)
        diff_mat = center-hb
        sq_diff_mat = diff_mat ** 2
        sq_distances = sq_diff_mat.sum(axis=1)
        distances = sq_distances ** 0.5
        sum_radius = 0
        for i in distances:
            sum_radius = sum_radius + i
        mean_radius = sum_radius / num
        return mean_radius
    else:
        return 1


def get_radius(hb):
    num = len(hb)
    center = hb.mean(0)
    diff_mat = center-hb
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    radius = max(distances)
    return radius


def plot_dot(data):
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1], s=7, c="#314300", linewidths=5, alpha=0.6, marker='o', label='data point')
    plt.legend(loc=1)


def draw_ball(hb_list):
    is_isolated = False
    for data in hb_list:
        if len(data) > 1:
            center = data.mean(0)
            radius = np.max((((data - center) ** 2).sum(axis=1) ** 0.5))
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            plt.plot(x, y, ls='-', color='black', lw=0.7)
        else:
            plt.plot(data[0][0], data[0][1], marker='*', color='#0000EF', markersize=3)
            is_isolated = True
    plt.plot([], [], ls='-', color='black', lw=1.2, label='hyper-ball boundary')
    plt.legend(loc=1)
    if is_isolated:
        plt.scatter([], [], marker='*', color='#0000EF', label='isolated point')
        plt.legend(loc=1)
    plt.show()


def normalized_ball(hb_list, hb_list_not, radius_detect):
    hb_list_temp = []
    for hb in hb_list:
        if len(hb) < 2:
            hb_list_not.append(hb)
        else:
            if get_radius(hb) <= 2 * radius_detect:
                hb_list_not.append(hb)
            else:
                ball_1, ball_2 = spilt_ball(hb)
                hb_list_temp.extend([ball_1, ball_2])
    
    return hb_list_temp, hb_list_not


def connect_ball(hb_list):
    hb_cluster = {}
    for i in range(0, len(hb_list)):
        hb = HB(hb_list[i], i)
        hb_cluster[i] = hb

    radius_sum = 0
    num_sum = 0
    hb_len = 0
    radius_sum = 0
    num_sum = 0
    for i in range(0, len(hb_cluster)):
        if hb_cluster[i].out == 0:
            hb_len = hb_len + 1
            radius_sum = radius_sum + hb_cluster[i].radius
            num_sum = num_sum + hb_cluster[i].num
    # 重叠统计
    for i in range(0, len(hb_cluster) - 1):
        if hb_cluster[i].out != 1:
            center_i = hb_cluster[i].center
            radius_i = hb_cluster[i].radius
            for j in range(i + 1, len(hb_cluster)):
                if hb_cluster[j].out != 1:
                    center_j = hb_cluster[j].center
                    radius_j = hb_cluster[j].radius
                    dis = ((center_i - center_j) ** 2).sum(axis=0) ** 0.5
                    if (dis <= radius_i + radius_j) & ((hb_cluster[i].hardlapcount == 0) & (
                            hb_cluster[j].hardlapcount == 0)):  # 由于两个的点是噪声球，所以纳入重叠统计
                        hb_cluster[i].overlap = 1
                        hb_cluster[j].overlap = 1
                        hb_cluster[i].hardlapcount = hb_cluster[i].hardlapcount + 1
                        hb_cluster[j].hardlapcount = hb_cluster[j].hardlapcount + 1

    hb_uf = UF(len(hb_list))
    for i in range(0, len(hb_cluster) - 1):
        if hb_cluster[i].out != 1:
            center_i = hb_cluster[i].center
            radius_i = hb_cluster[i].radius
            for j in range(i + 1, len(hb_cluster)):
                if hb_cluster[j].out != 1:
                    center_j = hb_cluster[j].center
                    radius_j = hb_cluster[j].radius
                    max_radius = max(radius_i, radius_j)
                    min_radius = min(radius_i, radius_j)
                    dis = ((center_i - center_j) ** 2).sum(axis=0) ** 0.5
                    dynamic_overlap = dis <= radius_i + radius_j + 0 * (max_radius)  # 重叠条件
                    num_limit = ((hb_cluster[i].num > 2) & (hb_cluster[j].num > 2))
                    if dynamic_overlap & num_limit:
                        hb_cluster[i].flag = 1
                        hb_cluster[j].flag = 1
                        hb_uf.union(i, j)
                    if dis <= radius_i + radius_j + ((max_radius)):
                        hb_cluster[i].softlapcount = 1
                        hb_cluster[j].softlapcount = 1

    for i in range(0, len(hb_cluster)):
        k = i
        if hb_uf.parent[i] != i:
            while (hb_uf.parent[k] != k):
                k = hb_uf.parent[k]
        hb_uf.parent[i] = k

    for i in range(0, len(hb_cluster)):
        hb_cluster[i].label = hb_uf.parent[i]
        hb_cluster[i].size = hb_uf.size[i]

    label_num = set()
    for i in range(0, len(hb_cluster)):
        label_num.add(hb_cluster[i].label)

    list = []
    for i in range(0, len(label_num)):
        list.append(label_num.pop())

    for i in range(0, len(hb_cluster)):
        distance = np.sqrt(2)
        if hb_cluster[i].flag == 0:
            for j in range(0, len(hb_cluster)):
                if hb_cluster[j].flag == 1:
                    center = hb_cluster[i].center
                    center2 = hb_cluster[j].center
                    dis = ((center - center2) ** 2).sum(axis=0) ** 0.5 - (hb_cluster[i].radius + hb_cluster[j].radius)
                    if dis < distance:
                        distance = dis
                        hb_cluster[i].label = hb_cluster[j].label
                        hb_cluster[i].flag = 2

    label_num = set()
    for i in range(0, len(hb_cluster)):
        label_num.add(hb_cluster[i].label)
    return hb_cluster


def GBC(data):

    hb_list_temp = [data]
    hb_list_not_temp = []
    # 按照质量分化
    while 1:
        ball_number_old = len(hb_list_temp) + len(hb_list_not_temp)
        hb_list_temp, hb_list_not_temp = division(hb_list_temp, hb_list_not_temp)
        ball_number_new = len(hb_list_temp) + len(hb_list_not_temp)
        if ball_number_new == ball_number_old:
            hb_list_temp = hb_list_not_temp
            break


    # 全局归一化
    radius = []
    for hb in hb_list_temp:
        if len(hb) >= 2:
            radius.append(get_radius(hb))
    radius_median = np.median(radius)
    radius_mean = np.mean(radius)
    radius_detect = max(radius_median, radius_mean)
    hb_list_not_temp = []
    while 1:
        ball_number_old = len(hb_list_temp) + len(hb_list_not_temp)
        hb_list_temp, hb_list_not_temp = normalized_ball(hb_list_temp, hb_list_not_temp, radius_detect)
        ball_number_new = len(hb_list_temp) + len(hb_list_not_temp)
        if ball_number_new == ball_number_old:
            hb_list_temp = hb_list_not_temp
            break


    gb_list_cluster = connect_ball(hb_list_temp)

    return gb_list_cluster
