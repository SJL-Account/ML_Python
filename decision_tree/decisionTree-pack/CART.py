
#coding:utf-8
'''
���㷨Ϊ�������㷨��ID3�㷨
'''
def calcEnt(dataset):
    '''
    �������ݼ��ϵ���
    ���룺���ݼ���
    ��������ϵ���
    '''
    # H=-( n/m*log(n/m,2))

    #np
    #m=dataset������
    m=len(dataset)

    #dic
    #����label,ֵ��label���ֵĴ���
    label_counts={}

    for line_vector in dataset:
        label=line_vector[-1]
        if label not in label_counts.keys():
            label_counts[label]=0
        label_counts[label]+=1
    h=0.0
    for key in label_counts:
        #��label���ֵĸ���
        prob=float(label_counts[key])/m
        h-=prob*log(prob,2)
    return h

def splitDataSet(dataset,axis,split_point,value):

    '''
    ��Ҫ�����ǽ�ά�ȡ���������Ҫ��ά��

    dataset:���ݼ�
    axis:������
    value:����ֵ
    return ��np.array ����ӵ��������value����ȥ��axis�е��������
    '''
    leftDataSet=[]
    rightDataSet=[]

    if value==1:
        for vector in dataset.tolist():
            if vector[axis] <= split_point:
                # ��ά
                 reduceDataset=vector[:axis]
                 reduceDataset.extend(vector[axis + 1:])
                 leftDataSet.append(reduceDataset)
        return  np.array(leftDataSet)
    elif value==0:
        for vector in dataset.tolist():
            if vector[axis] > split_point:
                # ��ά
                 reduceDataset=vector[:axis]
                 reduceDataset.extend(vector[axis + 1:])
                 rightDataSet.append(reduceDataset)
        return  np.array(rightDataSet)

def chooseBestPointToSplit(dataset):
    '''
    �����������������
    ������Ϣ������ѡȡ��������к���ѷָ��
    :param dataset: ���б�ǩ�����ݼ���
    :return: ��õ�����ֵ��(�Ծ���Ӱ������),��ѷָ��
    '''

    # �㷨���裺
    # �����ͱ����ҳ���ѷָ��
    # ѭ�������У�ѭ���е�����ֵ���������ݼ�����ֵ
    # ȡ������Ϣ���������Ǹ�
    # ȡ������������Ϣ�����Ǹ�����ֵ
    # ���ظ�ֵ


    #��û�а�������������ֵʱ���صĴ�СInfo(D)
    Info=calcEnt(dataset)
    #����������ݷָ����Ϣ��Info_left(D)
    Info_A_left=0.0
    #����������ݷָ����Ϣ��Info_right(D)
    Info_A_right=0.0

    #ÿ�����ݵĳ��ȣ��еĳ��ȣ�
    n=len(dataset[0])-1

    #�������
    best_feature = -1
    #��������е���ѷָ��
    split_points=0.0
    #�������
    col_max_gain = 0.0

    #ѭ����������������е�����ֵ

    #����ÿ��
    for i in range(n):
        
        #ȡ��ÿ�е�����ֵ
        feature_values= [example[i] for example in dataset]

        #��ÿ������ֵ����
        feature_values.sort()

        #ÿ������ֵ�ĳ���
        f_m=len(feature_values)


        # ѭ����������ֵ
        for fi in range(f_m):

            if fi+1<f_m :
                #ѡȡ a1+a2/2 ��ΪԤѡ�ָ��
                split_point=(feature_values[fi]+feature_values[fi+1])/2



            #���ݷָ��ֱ���������������ֵ
            leftdataset=[]
            rightdataset=[]

            for vector in dataset.tolist():
                if vector[i] <= split_point:
                    leftdataset.append(vector)
                else:
                    rightdataset.append(vector)

            # ѭ�������

            #������
            prob_left=len(leftdataset)/float(len(dataset))

            #�����Ϣ�� InfoA
            Info_left=prob_left*calcEnt(leftdataset)

            #�Ҳ����
            prob_right = len(rightdataset) / float(len(dataset))

            #����Ϣ��
            Info_right= prob_right * calcEnt(rightdataset)

            #����split_point �������Ϣ��
            col_ent = (Info_left + Info_right)

            #��Ϣ����
            gain=Info-col_ent

            #��Ϣ������

            #gain_rate= gain/col_ent



            if gain >col_max_gain:
                col_max_gain=gain
                col_split_point=split_point
                best_feature=i


    return  best_feature,col_split_point

def majorityCnt(class_list):

    '''
    ������Ѿ�ȫ������ʱ�����������ִ������ȽϷ�����

    class_list:��ǩ����
    return :������
    '''

    class_counts={}

    for vote in class_list:
        if vote not in class_counts.keys():
            class_counts[vote]=0
        class_counts[vote]+=1
    sorted_class_list=sorted(class_counts.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sorted_class_list[0][0]

def create_tree(dataset,features):

    '''
    ����������
    :param dataset:���ݼ���
    :param labels: ��������
    :return: ��������ֵ�
    '''
    #ѡ��dataset����𼯺�
    class_labels=[example[-1] for example in dataset]

    #��� ����
    features_copy=copy.deepcopy(features)

    #��ֹ����
    # 1.������ֵ�����е�ֵ��ͬ
    if class_labels.count(class_labels[0])==len(class_labels):
        #��ʱ������

        #class_count=len(class_labels)
        #if class_count>5:
            #print dataset
        #print  class_key[class_labels[0]],str(class_count)
        return class_labels[0]
    # 2.���������
    if len(dataset[0])==1:
        return majorityCnt(class_labels)

    n=len(dataset[0])

    continuity_dataset=[]
    discrete_dataset=[]

    #----------------------������-------------------------------------------
    #�����ѷָ������к͸��е���ѷָ��
    col_i,split_point=chooseBestPointToSplit(dataset)
    #ѡ���������
    best_feature=features_copy[col_i]
    best_feature=best_feature+'<='+str(split_point)

    # ����һ���ֵ�tree
    tree ={best_feature:{}}

    #ɾ����������������
    del(features_copy[col_i])

    #��ȡ�������������������ֵ
    best_values=[example[col_i] for example in dataset]

    #�ݹ� ��������ࡢ�Ҳ������ֵ�ֱ�ָ����ݣ������е��Ѿ��������������У���Ϊ��һ����Ҷ�����ݼ��ϡ�������
    for value in [1,0]:
        sub_labels=features_copy[:]
        tree[best_feature][value] =create_tree(splitDataSet(dataset,col_i,split_point,value),sub_labels)
    #--------------------------------������-----------------------------------
    return tree

# {'AT60<=100': {'1': {'Flipers': {'1': 'yes', '0': 'no'}}, '0': 'no'}}

#��������

def classify(tree,features,testVector):
    #��ȡ���ڵ�
    firstKey = tree.keys()[0]
    #��ȡ�νڵ�
    secondDic=tree[firstKey]
    #ȡ�������������׽ڵ�ĵ�λ�ã�ͨ�������ַ����������޷��Ӳ��������л�ȡ����ֵ�ģ�
    index= features.index(firstKey)
    class_label=''
    for key in secondDic.keys():
        if str(testVector[index])==key:
            #���������ֵ�������ֵ�
            if type(secondDic[key]).__name__=='dict':
                class_label=classify(secondDic[key],features,testVector)
            else:
                class_label=secondDic[key]
    return class_label

def load_tree(file_name):
    '''
    ������
    '''
    import pickle
    fr=open(file_name)
    return pickle.load(fr)

def store_tree(tree,file_name):
    '''
    ������
    '''
    import pickle
    fw=open(file_name,'w')
    pickle.dump(tree,fw)
    fw.close()