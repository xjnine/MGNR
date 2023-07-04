
import numpy as np
from matplotlib import pyplot as plt

#plotJudge用于判断是否画图
def plot_dot(data, label=[],text=[],display_text=False,show=True,marker='o',s=10,datasetColor="grey",plotJudge=True):
    if not plotJudge:
        return
    color = {
        -1:'black',
        0: '#707afa',
        1: '#ffe135',
        2: '#16ccd0',
        3: '#ed7231',
        4: '#0081cf',
        5: '#afbed1',
        6: '#bc0227',
        7: '#d4e7bd',
        8: '#f8d7aa',
        9: '#fecf45',
        10: '#f1f1b8',
        11: '#b8f1ed',
        12: '#ef5767',
        13: '#e7bdca',
        14: '#8e7dfa',
        15: '#d9d9fc',
        16: '#2cfa41',
        17: '#e96d29',
        18: '#7f722f',
        19: '#bd57fa',
        20: '#e4f788',
        21: '#fb8e94',
        22: '#b8d38f',
        23: '#e3a04f',
        24: '#edc02f',
        100: 'black' }
    if len(label)==0:
        color_list = [datasetColor] * len(data)
    else:
        color_list=[-1]*len(label)
        for i,l in enumerate(set(label)):
            for j,index in enumerate(label):
                if index==l and i<25:
                    color_list[j]=color[i]
                elif index==l and i>24:
                    color_list[j]='red'


    plt.figure(figsize=(10, 10))
    #将绘图控制在某个区间内
    plt.axis([-0.1,1.1,-0.1,1.1])
    plt.scatter(data[:,0], data[:,1],s = s, c = color_list, linewidths=5, alpha=0.7, marker=marker,label=' ')
    if len(text) >0 and display_text:
        for index,(i,j) in enumerate(zip(data[:,0],data[:,1])):
            plt.text(i, j+j*0.02,text[index],ha="center")
    plt.legend()
    if show:
        plt.show()

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

