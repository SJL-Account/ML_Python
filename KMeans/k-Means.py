#coding:utf-8
import numpy as np
import io


def autoNorm(dataSet):
    '''
    用于
    标准化
    '''
    m = dataSet.shape[0]
    # 1*n
    min_values = dataSet.min(0)
    # 1*n
    max_values = dataSet.max(0)

    # 1*n
    differ = max_values - min_values

    # m*n
    new_dataSet = np.zeros((dataSet.shape[0], dataSet.shape[1]))

    # new_value=(old_value-min)/(max-min)
    new_dataSet = (dataSet - np.tile(min_values, (m, 1))) / np.tile(differ, (m, 1))

    return new_dataSet

def load_dataset(file_name):
    f=open(file_name)
    lines=f.readlines()
    data_set=[]
    for currentline  in lines:
        line=currentline.strip().split('\t')
        line=map(float,line)
        data_set.append(line)
    return  data_set

def distants(vecA,vecB):
    return np.sqrt(np.sum(np.power(vecA-vecB,2)))

def SSE (centeroids,cluster_assment):
    cluster_assment=np.array(cluster_assment)

    k=centeroids.shape[0]

    m=cluster_assment.shape[0]

    cluster_sum=np.zeros((k,1))


    for i in range(k):
        for j in range(m):
            if i==cluster_assment[j][0]:
                cluster_sum[i]+=cluster_assment[j][1]

    return cluster_sum

def rand_cent(data_set, k):
    '''
    生成随机质心点
    :param data_set:
    :param k: k个质心点
    :return:
    '''
    n=data_set.shape[1] #n列
    centroids=np.mat(np.zeros((k,n)))# k行n列
    for j in range(n):
        minj= np.min(data_set[:, j])#从第j列中选出最小值
        maxj= np.max(data_set[:, j])#从第j列中选出最小值
        rangej= float(maxj-minj)#范围
        centroids[:, j]=minj+rangej*np.random.rand(k, 1)  #k行1列
    return  centroids


def kMeans(data_set, k):

    m = data_set.shape[0]

    #m*2 m行 0列代表距离该行数据最近的质心索引值，1列代表质心到该行数据的距离
    cluster_assment= np.mat(np.zeros((m,2)))

    cluster_changed=True

    #k*n   k个质心点，n维数据
    centeroids=rand_cent(data_set,k);
    while cluster_changed:
        cluster_changed = False

        #对每一行数据
        for i in range(m):
            minDist=np.inf
            minIndex=-1
            #对每个质心点
            for j in range(k):

                # 计算每一行数据对每个质心的距离
                Dist=distants(data_set[i],centeroids[j])

                #如果i行数据到j质点的距离均小于其他质点的距离
                if Dist<minDist:
                    #把最小距离更新为i行数据到质点j的距离
                    minDist=Dist
                    #把最小索引更新为j
                    minIndex=j
            #如果该行所属于的簇在计算后没变
            if cluster_assment[i,0]!=minIndex :cluster_changed=True
            #将该行的所属质心更新
            cluster_assment[i,:]=minIndex,minDist**2
        print centeroids
        #用mean更新质心
        for i in range(k):
            #读出质心索引为i的簇
            i_assments= data_set[np.nonzero(cluster_assment[:,0].A==i)[0]]

            #用簇的平均值来更新簇的质心

            centeroids[i,:]=np.mean(i_assments,axis=0)

    return centeroids,cluster_assment

data_set= np.mat(load_dataset('1200.txt'))
data_set=data_set[:,[4,5]]
data_set = autoNorm(data_set)

centerids,cluster_assment= kMeans(data_set,k=2)

cluster_sum=SSE(centerids,cluster_assment)

print cluster_sum

centerids=np.array(centerids)

import matplotlib.pyplot as plt


fig=plt.figure()

ax=fig.add_subplot(111)

data_set=np.array(data_set)


ax.scatter(data_set.T[0], data_set.T[1])
ax.scatter(centerids[0],centerids[1],color='r')
plt.xlabel('X1');plt.ylabel('X2')

plt.show()







