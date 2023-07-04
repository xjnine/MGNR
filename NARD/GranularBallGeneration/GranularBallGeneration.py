
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

from Tools.Plot import plot_dot, draw_ball


def division(hb_list, hb_list_not):
    gb_list_new = []
    for hb in hb_list:
        if len(hb) > 1:
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


def spilt_ball2(data):
    ball1 = []
    ball2 = []
    # n, m = data.shape
    # x_mat = data.T
    # g_mat = np.dot(x_mat.T, x_mat)
    # h_mat = np.tile(np.diag(g_mat), (n, 1))
    # d_mat = np.sqrt(h_mat + h_mat.T - g_mat * 2)

    #  分裂方法1：
    # 调用pdist计算距离矩阵
    A = pdist(data)
    d_mat = squareform(A)
    r, c = np.where(d_mat == np.max(d_mat))
    r1 = r[1]
    c1 = c[1]
    for j in range(0, len(data)):
        if d_mat[j, r1] < d_mat[j, c1]:
            ball1.extend([data[j, :]])
        else:
            ball2.extend([data[j, :]])
    ball1 = np.array(ball1)
    ball2 = np.array(ball2)
    return [ball1, ball2]

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
    if num==0:
        return 0
    center = hb.mean(0)
    diff_mat = center - hb
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sum_radius = 0
    for i in distances:
        sum_radius = sum_radius + i
    mean_radius = sum_radius / num
    if num > 2:
        return mean_radius
    else:
        return 1


def get_radius(hb):
    num = len(hb)
    center = hb.mean(0)
    diff_mat = center - hb
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    radius = max(distances)
    return radius




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
    # 画出粒球覆盖情况
    # plot_dot(data,show=False)
    # draw_ball(hb_list_temp)







    return hb_list_temp
