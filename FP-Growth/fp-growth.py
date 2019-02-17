#coding:utf-8
class treeNode:

    def __init__(self,nameValue,numOccur,parentNode):
        self.children={}
        self.parent=parentNode
        self.count=numOccur
        self.name=nameValue
        self.nodeLink=None
    def inc (self,numOccur):
        self.count+=numOccur

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):

    dataDict={}
    for tranSet in dataSet:
        key=frozenset(tranSet)
        if not key  in dataDict.keys():
            dataDict[key]=1
        else:
            dataDict[key]+=1
    return dataDict


def creatTree(dataSet,minSupport=1):
    '''
    构建FP-Growth
    :param dataSet:数据集
    :param minSupport:最小支持度
    :return:
    '''
    HeaderTable={}
    reTree=treeNode('TopNode',1,None)

    #算法步骤
    #求出C1放进HeaderTable
    for tranSet in dataSet:
        for item in tranSet:
            HeaderTable[item]=HeaderTable.get(item,0)+dataSet[tranSet]
    #删除掉于支持度的项
    for key in  HeaderTable.keys():
        if HeaderTable[key]<minSupport:
            del(HeaderTable[key])

    if len(HeaderTable) ==0: return None,None
    #构造新的HeaderTable
    for key in HeaderTable.keys():
        HeaderTable[key]=[HeaderTable[key],None]
    #构造频繁项集

    freSet=[v for v in HeaderTable.keys()]

    for tranSet,count in dataSet.items():

        localD={}
        #通过freset 和
        for item in  tranSet:

            if item in freSet:

                localD[item]= HeaderTable[item][0]
        if localD>0:

            sorteditems = [v[0] for v in sorted(localD.items(),key=lambda p:p[1],reverse=True)]

            updateTree(sorteditems,reTree,HeaderTable,count)
    return reTree,HeaderTable

def updateTree(item,inTree,HeaderTable,count):

        #是否在我的子节点里面
        if item[0] in inTree.children:
            inTree.children[item[0]].inc(count)
        #不在的话添加一个树节点
        else:
            #inTree是个指针
            inTree.children[item[0]]=treeNode(item[0],count,inTree)

            #进行链表连接
            #如果头表格中还没有存储指向的对象的话
            if HeaderTable[item[0]][1]==None:
                #就把这个对象给他
                HeaderTable[item[0]][1]=inTree.children[item[0]]
            #更新链表
            else:
                updateHeader(HeaderTable[item[0]][1],inTree.children[item[0]])
        #如果排完序列的集合没用完
        if len(item)>1:
            updateTree(item[1::],inTree.children[item[0]],HeaderTable,count)

def updateHeader(testNode,targetNode):
    #如果结束节点有东西，往下找找到为空的上节点
    while testNode.nodeLink!=None:
        testNode=testNode.nodeLink
    #把你想要的节点赋给上个节点
    testNode.nodeLink=targetNode

def ascendTree(preFix,treeNode):

    if treeNode.parent!=None:
        preFix.append(treeNode.name)
        treeNode=treeNode.parent
        ascendTree(preFix,treeNode)


def findPreFix(bastPat,treeNode):
    condPats={}
    #如果该节点上方有节点
    while treeNode!=None:
        preFix=[]
        #递归索引出路径
        ascendTree(preFix,treeNode)
        #找下一个基地所在的对象
        if len(preFix)>1:
            condPats[frozenset(preFix[1:])]=treeNode.count
        treeNode=treeNode.nodeLink
    return condPats


def minTree(inTree,HeaderTable,pre,freSet,minSup):

    '''

    :return:
    '''
    #递减排列HeaderTable中的序列形成一个列表BigList
    bigList={v[0] for v in sorted(HeaderTable.items(),key=lambda p:p[1])}

    #循环bigList
    for basePat in bigList:
        #把下一层基加入和上一层基组合成一个新的频繁组合 NewFreSet
        newFreSet=pre.copy()
        newFreSet.add(basePat)
        #把newFreSet加入到总的频繁项集中
        freSet.append(newFreSet)

        #找出该层基的前缀
        condPattBases=findPreFix(basePat,HeaderTable[basePat][1])

        #以该层为基础创建一颗FP-tree
        myTree,myHead=creatTree(condPattBases,minSup)

        if myHead!=None:
            minTree(myTree,myHead,newFreSet,freSet,minSup)


dataset=loadSimpDat()

dataset=createInitSet(dataset)

print (dataset)

tree,table=  creatTree(dataset,3)

freSet=[]

minTree(tree,table,set([]),freSet,3)

print (freSet)