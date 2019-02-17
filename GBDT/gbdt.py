import numpy as np
from sklearn.tree import DecisionTreeRegressor

class loss:

    def mse(self,y,y_pre):

        return 0.5*(((y-y_pre)**2).sum())/len(y)


    def d_mse(self,y,y_pre): pass

class GBDT:


    def __init__(self,n_estimate,
                  learning_rate,
                  loss_function,
                  regression,
                  criterion,
                  splitter,
                  max_depth,
                  min_samples_split,
                  min_samples_leaf,
                  min_weight_fraction_leaf,
                  max_features,
                  random_state,
                  max_leaf_nodes,
                  presort):
        self.n_estimate=n_estimate
        self.learning_rate=learning_rate
        self.trees=[DecisionTreeRegressor(criterion,
                                          splitter,
                                          max_depth,
                                          min_samples_split,
                                          min_samples_leaf,
                                          min_weight_fraction_leaf,
                                          max_features,
                                          random_state,
                                          max_leaf_nodes,
                                          presort)]*n_estimate
        self.loss=loss()

        if loss_function is None:
            self.loss_function=self.loss.mse()

        self.regression=regression



    def fit(self,x,y):
        '''

        :return:
        '''
        #1.设置起始决策树

        tree_0=self.trees[0]

        y_pre=tree_0.fit(x)

        for i in range(1,self.n_estimate):
            # 1.计算梯度

            gradient=self.loss.d_mse(y,y_pre)

            #2.学习第n棵树,让第n棵树学习到下降的路径

            self.trees[i].fit(x,gradient)

            #3.寻找步长

            #4.更新模型 进行下一次学习路径

            y_pre-=np.multiply(self.learning_rate,self.trees[i].predict(x))



    def predict(self,x):

        tree_0 = self.trees[0]

        y_pre = tree_0.fit(x)

        for i in range(self.n_estimate):

            y_pre -= np.multiply(self.learning_rate, self.trees[i].predict(x))

        return y_pre

class GBDTRegressor(GBDT):

    def __init__(self,n_estimate,
                  learning_rate,
                  loss_function,
                 criterion='mse',
                 splitter='best',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 presort=False):
        super(GBDTRegressor,self).__init__(n_estimate,
                  learning_rate,
                  loss_function,
                  True,
                  criterion,
                  splitter,
                  max_depth,
                  min_samples_split,
                  min_samples_leaf,
                  min_weight_fraction_leaf,
                  max_features,
                  random_state,
                  max_leaf_nodes,
                  presort)



if __name__=='__main__':

    gbdt=GBDTRegressor(n_estimate=10,learning_rate=0.1,loss_function=None)
    gbdt.fit()
    gbdt.predict()









