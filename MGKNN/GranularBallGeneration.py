import numpy as np
import matplotlib.pyplot as plt


class GB:
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


def division(gb_list, gb_list_not):
    gb_list_new = []
    for gb in gb_list:
        if len(gb) > 16:
            ball_1, ball_2 = spilt_ball(gb)
            dm_parent = get_dm(gb)
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
                gb_list_not.append(gb)
        else:
            gb_list_not.append(gb)
    return gb_list_new, gb_list_not


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

def get_dm(gb):
    num = len(gb)
    if num > 2:
        center = gb.mean(0)
        diff_mat = center-gb
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


def get_radius(gb):
    num = len(gb)
    center = gb.mean(0)
    diff_mat = center-gb
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    radius = max(distances)
    return radius


def plot_dot(data):
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1], s=7, c="#314300", linewidths=5, alpha=0.6, marker='o', label='data point')
    plt.legend(loc=1)


def draw_ball(gb_list):
    is_isolated = False
    for data in gb_list:
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


def normalized_ball(gb_list, gb_list_not, radius_detect):
    gb_list_temp = []
    for gb in gb_list:
        if len(gb) < 2:
            gb_list_not.append(gb)
        else:
            if get_radius(gb) <= 2 * radius_detect:
                gb_list_not.append(gb)
            else:
                ball_1, ball_2 = spilt_ball(gb)
                gb_list_temp.extend([ball_1, ball_2])
    
    return gb_list_temp, gb_list_not


def connect_ball(gb_list):
    gb_cluster = {}
    for i in range(0, len(gb_list)):
        gb = GB(gb_list[i], i)
        gb_cluster[i] = gb

    radius_sum = 0
    num_sum = 0
    gb_len = 0
    radius_sum = 0
    num_sum = 0
    for i in range(0, len(gb_cluster)):
        if gb_cluster[i].out == 0:
            gb_len = gb_len + 1
            radius_sum = radius_sum + gb_cluster[i].radius
            num_sum = num_sum + gb_cluster[i].num
    # 重叠统计
    for i in range(0, len(gb_cluster) - 1):
        if gb_cluster[i].out != 1:
            center_i = gb_cluster[i].center
            radius_i = gb_cluster[i].radius
            for j in range(i + 1, len(gb_cluster)):
                if gb_cluster[j].out != 1:
                    center_j = gb_cluster[j].center
                    radius_j = gb_cluster[j].radius
                    dis = ((center_i - center_j) ** 2).sum(axis=0) ** 0.5
                    if (dis <= radius_i + radius_j) & ((gb_cluster[i].hardlapcount == 0) & (
                            gb_cluster[j].hardlapcount == 0)):  # 由于两个的点是噪声球，所以纳入重叠统计
                        gb_cluster[i].overlap = 1
                        gb_cluster[j].overlap = 1
                        gb_cluster[i].hardlapcount = gb_cluster[i].hardlapcount + 1
                        gb_cluster[j].hardlapcount = gb_cluster[j].hardlapcount + 1

    gb_uf = UF(len(gb_list))
    for i in range(0, len(gb_cluster) - 1):
        if gb_cluster[i].out != 1:
            center_i = gb_cluster[i].center
            radius_i = gb_cluster[i].radius
            for j in range(i + 1, len(gb_cluster)):
                if gb_cluster[j].out != 1:
                    center_j = gb_cluster[j].center
                    radius_j = gb_cluster[j].radius
                    max_radius = max(radius_i, radius_j)
                    min_radius = min(radius_i, radius_j)
                    dis = ((center_i - center_j) ** 2).sum(axis=0) ** 0.5
                    dynamic_overlap = dis <= radius_i + radius_j + 0 * (max_radius)  # 重叠条件
                    num_limit = ((gb_cluster[i].num > 2) & (gb_cluster[j].num > 2))
                    if dynamic_overlap & num_limit:
                        gb_cluster[i].flag = 1
                        gb_cluster[j].flag = 1
                        gb_uf.union(i, j)
                    if dis <= radius_i + radius_j + ((max_radius)):
                        gb_cluster[i].softlapcount = 1
                        gb_cluster[j].softlapcount = 1

    for i in range(0, len(gb_cluster)):
        k = i
        if gb_uf.parent[i] != i:
            while (gb_uf.parent[k] != k):
                k = gb_uf.parent[k]
        gb_uf.parent[i] = k

    for i in range(0, len(gb_cluster)):
        gb_cluster[i].label = gb_uf.parent[i]
        gb_cluster[i].size = gb_uf.size[i]

    label_num = set()
    for i in range(0, len(gb_cluster)):
        label_num.add(gb_cluster[i].label)

    list = []
    for i in range(0, len(label_num)):
        list.append(label_num.pop())

    for i in range(0, len(gb_cluster)):
        distance = np.sqrt(2)
        if gb_cluster[i].flag == 0:
            for j in range(0, len(gb_cluster)):
                if gb_cluster[j].flag == 1:
                    center = gb_cluster[i].center
                    center2 = gb_cluster[j].center
                    dis = ((center - center2) ** 2).sum(axis=0) ** 0.5 - (gb_cluster[i].radius + gb_cluster[j].radius)
                    if dis < distance:
                        distance = dis
                        gb_cluster[i].label = gb_cluster[j].label
                        gb_cluster[i].flag = 2

    label_num = set()
    for i in range(0, len(gb_cluster)):
        label_num.add(gb_cluster[i].label)
    return gb_cluster


def GBC(data):

    gb_list_temp = [data]
    gb_list_not_temp = []
    # 按照质量分化
    while 1:
        ball_number_old = len(gb_list_temp) + len(gb_list_not_temp)
        gb_list_temp, gb_list_not_temp = division(gb_list_temp, gb_list_not_temp)
        ball_number_new = len(gb_list_temp) + len(gb_list_not_temp)
        if ball_number_new == ball_number_old:
            gb_list_temp = gb_list_not_temp
            break


    # 全局归一化
    radius = []
    for gb in gb_list_temp:
        if len(gb) >= 2:
            radius.append(get_radius(gb))
    radius_median = np.median(radius)
    radius_mean = np.mean(radius)
    radius_detect = max(radius_median, radius_mean)
    gb_list_not_temp = []
    while 1:
        ball_number_old = len(gb_list_temp) + len(gb_list_not_temp)
        gb_list_temp, gb_list_not_temp = normalized_ball(gb_list_temp, gb_list_not_temp, radius_detect)
        ball_number_new = len(gb_list_temp) + len(gb_list_not_temp)
        if ball_number_new == ball_number_old:
            gb_list_temp = gb_list_not_temp
            break


    gb_list_cluster = connect_ball(gb_list_temp)

    return gb_list_cluster
