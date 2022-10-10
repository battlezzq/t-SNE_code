# coding utf-8
"""t-SNE对手写数字进行可视化"""
from sklearn import manifold
import numpy as np
import matplotlib.pyplot as plt
import time
tsne=manifold.TSNE
def getDistance(x):
    sum_x =np.sum(np.square(x),1)
    distance = np.add(np.add(-2 * np.dot(x,x.T),sum_x).T,sum_x)
    return distance

def getPerplexity(distance,index=0,beta=1.0):
    prob = np.exp(-distance *beta)
    prob[index]=0
    sum_prob = np.sum(prob)
    perp = np.log(sum_prob) +beta * np.sum(distance * prob) / sum_prob
    prob = prob / sum_prob
    return perp, prob

def seach_prob(x, tol =1e-5, perplexity = 30.0):
    #先计算高维分布 由于需要用到sigema，因此用困惑度求sigema
    (n,d) = x.shape#n为x点个数,d为特征个数
    dist = getDistance(x)
    pair_prob = np.zeros((n,n))
    beta = np.ones((n,1))
    base_perp = np.log(perplexity)

    for i in range(n):#对每个点
        if i % 500 == 0:
            print('计算第 %s 个点的概率，共 %s 个点'%(i,n))
            Note.write('计算第 %s 个点的概率，共 %s 个点 \n'%(i,n))
        # 初始化beta 高斯分布的参数
        betamin = -np.inf
        betamax = np.inf
        #计算当前点的复杂度与对应的概率
        perp, this_prob =getPerplexity(dist[i],i,beta[i])

        #
        perp_diff = perp - perplexity#当前点的复杂度与设定复杂度之间的差值
        tries = 0
        while np.abs(perp_diff) > tol and tries < 50:#寻找次数小于50 并且复杂度差值大于ξ（当复杂度差值小于ξ时，就不用再进行查找）
            if perp_diff > 0:
                betamin =beta[i].copy()
                if betamax ==np.inf or betamax ==-np.inf:
                    beta[i] =beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax =beta[i].copy()
                if betamin ==np.inf or betamin ==-np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (betamin + beta[i]) / 2
            #更新当前i对应复杂度与概率分布
            perp,this_prob = getPerplexity(dist[i],i,beta[i])
            perp_diff = perp - base_perp
            tries = tries + 1
        #记录该复杂度下经过二分法找到的prob值 pair_prob为n*n矩阵 即pij代表第j个点附近复杂度个点对应的概率分布
        pair_prob[i,] = this_prob
    print("方差平均值为:",np.mean(np.sqrt(1 / beta)))
    Note.write("方差平均值为: %s \n"%np.mean(np.sqrt(1 / beta)))
    return pair_prob#得到pij概率分布矩阵

def pca(x,no_dims= 50):#先利用PCA进行降维
    print("利用PCA对数据进行预处理中......")
    Note.write("利用PCA对数据进行预处理中......")
    (n,d) = x.shape
    x = x - np.tile(np.mean(x,0),(n,1))#先计算每个样本点的均值即np.mean(x,0)得到1*n的均值矩阵，再利用tile进行拓展平铺得到每一行都相等的n*n的均值矩阵
    l, M = np.linalg.eig(np.dot(x.T,x))
    y = np.dot(x,M[:,0:no_dims])#M[:,0:no_dims]为特征向量矩阵
    return y

def tsne(x, no_dims=2, initial_dims=50, perplexity=50.0, max_iter=10000):
    #确认目标维度数据格式
    if isinstance(no_dims,float):
        print("错误：目标维度为浮点数,不符合要求")
        return -1
    if round(no_dims) != no_dims:
        print("错误：目标维度应该为整数")
        return -1

    # 初始化参数
    x = pca(x,initial_dims).real
    (n,d) = x.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    y = np.random.randn(n,no_dims)#随机生成低维数据
    dy = np.zeros((n,no_dims))
    iy = np.zeros((n,no_dims))
    gains = np.ones((n,no_dims))

    #对称化-高维概率分布矩阵
    P = seach_prob(x,1e-5,perplexity)# P是一个n*n的概率矩阵
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4
    P = np.maximum(P, 1e-12)

    for iter in range(max_iter):
        #Q为低维空间概率分布
        sum_y = np.sum(np.square(y),1) #按行平方相加
        num = 1 / (1+np.add(np.add(-2 * np.dot(y,y.T),sum_y).T,sum_y))#t分布相似度
        num[range(n),range(n)] = 0#定义pii的相似度为0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        #梯度下降法 迭代
        PQ = P-Q#概率分布做差
        for i in range(n):
            #对每个数据Xi
            #dy 梯度 dy为n*50的零矩阵
            dy[i,:] = np.sum(np.tile(PQ[:,i] * num[:,i],(no_dims,1)).T * (y[i,:]-y),0)
            #PQ-n*n PQ[:,i]-n*1 y-n*50   num-n*n  num[:,i]-n*1
            # np.tile(PQ[:,i] * num[:,i],(no_dims,1)) 是(n*50,1)矩阵 它的转置为(1,n*50)矩阵 再与y-(n,50)矩阵相乘 得到n*50的梯度矩阵

        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        #gain是一个n*50全1矩阵
        gains = (gains + 0.2) * ((dy > 0) != (iy > 0)) + (gains * 0.8) * ((dy > 0) == (iy > 0))
        gains[gains < min_gain] = min_gain
        #若gains每个值都比min_gain0.01小，那么gains全都变为0.01 否则不变，继续按原来的增益进行梯度下降
        iy = momentum * iy - eta * (gains * dy)#n*50 梯度矩阵会变为原来的0.2或0.8倍
        y = y + iy#低维数值修正/更新
        y = y - np.tile(np.mean(y, 0), (n, 1))# 中心化，使y的均值为0
        # 计算当前的损失函数的值
        if (iter + 1)%100 == 0:
            if iter > 100:
                C = np.sum(P * np.log(P / Q))
            else:
                C = np.sum(P/4 * np.log( P/4 / Q))

            print("第 ",(iter+1), "次迭代的损失为:",C)
            Note.write("第%s"%(iter+1))
            Note.write("次迭代的损失为:%s\n"%C)
        if iter == 100:
            P = P/4
    print("完成训练!")
    Note.write('完成训练！')
    return y

if __name__ == "__main__":
    Note=open('record.txt',mode='w')
    start_time = time.time()
    X = np.loadtxt("mnist2500_X.txt")
    labels = np.loadtxt("mnist2500_labels.txt")
    Y = tsne(X, 2, 50)
    end_time = time.time()
    running_time = end_time - start_time
    print(running_time)
    from matplotlib import pyplot as plt
    plt.scatter(Y[:,0], Y[:,1], 20, labels)
    plt.savefig("t-SNE_img")
    plt.show()
