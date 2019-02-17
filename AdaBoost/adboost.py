# coding:utf-8

import numpy as np

'''
adaboost算法步骤
输入：数据集、label、迭代次数
输出：整个弱分类的步骤

'''

'''
弱分类器算法
'''


def loadSimpData():
    datMat = np.matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

def stumpClassify(data, dim, value, symbol):
    '''
    二值化，如果大于的话
    '''
    reArray = np.ones((data.shape[0], 1))
    if symbol == 'lt':
        reArray[data[:, dim] <= value] = -1.0
    else:
        reArray[data[:, dim] >= value] = -1.0
    return reArray


def buildStump(data, label, D):
    bestStump = {}

    min_errRate = float("+inf")  # np.inf

    m, n = data.shape
    # 对每个特征
    for i in range(n):
        # 计算最小最大距离求步长
        col_min = data[:, i].min()
        col_max = data[:, i].max()
        step_num = 10
        step_size = (col_max - col_min )/ step_num
        # 对每个步长
        for j in range(-1, int(step_num) + 1):

            col_value = col_min + j * step_size

            for symbol in ['lt', 'ml']:

                predict_values = stumpClassify(data, i, col_value, symbol)

                # 定义错误率
                errArr = np.ones((m, 1))

                errArr[predict_values == label.T] = 0

                errRate = np.mat(D.T) * errArr

                if errRate < min_errRate:
                    min_errRate = errRate
                    bestClassEst = predict_values.copy()
                    bestStump['dim'] = i
                    bestStump['value'] = col_value
                    bestStump['symbol'] = symbol

    return bestStump, min_errRate, bestClassEst


def adaboost(data, label, numIter=10):
    label=np.mat(label)
    m, n = data.shape
    D = np.ones((m, 1)) / m
    weakClassArr = []
    aggErrorRates=[]

    aggClassEst=np.zeros((m,1))
    for i in range(numIter):
        # 通过弱决策算法选出最小错误率的决策边界 ,并返回决策边界和错误率，以及最小错误率的最好预测分类
        bestStump, min_errRate, bestClassEst = buildStump(data, label, D)

        # 计算alpha的值

        # epsilon=未正确分类的样本数目/总样本数目
        epsilon = min_errRate
        # alpha=0.5(ln(1-epsilon)/epsilon)
        alpha = float(0.5 * (np.log((1 - epsilon) / epsilon)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)

        # 如果样本i被正确分类，样本i的权重更新：D*exp(-alpha)/sum(D)

        # 如果样本i被错分类，样本i的权重更新为D*(exp(alpha))/sum(D)
        # 如果结果相同，则为+1，如果结果不同则为-1
        a=-1 * alpha * label.T
        expen = np.multiply(a, bestClassEst)

        D = np.multiply(D, np.exp(expen))

        D = D / D.sum()

        # 对alpha与预测分类进行累计加和

        aggClassEst+=alpha*bestClassEst
        # 根据加和后的值重新计算整体错误率，如果错误率为0，则退出循环优化结束

        aggErrors = np.multiply(np.sign(aggClassEst)!=label.T,np.ones((m,1)))

        aggErrorRate = aggErrors.sum() / m
        aggErrorRates.append(aggErrorRate)
        if aggErrorRate == 0:
            break;


        # 如果超出自定义迭代次数，则退出循环优化结束


        # 返回弱分类器的决策边界组合


    return  weakClassArr, aggErrorRates

def adaClassifier(test_data,classifierArr):

    test_mat=np.mat(test_data)
    aggClassEst=np.ones((test_mat.shape[0],1))
    for i in range(len(classifierArr)):
        bestClassEst=stumpClassify(test_mat,classifierArr[i]['dim'],classifierArr[i]['value'],classifierArr[i]['symbol'])
        aggClassEst+=classifierArr[i]['alpha']*bestClassEst
        #print aggClassEst
    return np.sign(aggClassEst)



