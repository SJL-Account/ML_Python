import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mlb
from wordcloud import WordCloud
import pandas as pd
from Apriori import *
import numpy as np

papers=pd.read_csv('Caj_张金亮/papers_all.csv',encoding='gbk')

#关键词提取
#print (papers[papers.refer_key!=''].title.value_counts()[papers[papers.refer_key!=''].title.value_counts()>5].index.tolist())

def generate_paper_double_tuple(top_refer):
    '''
    生成文献二元组关系
    :return:
    '''
    papers_relation=[]
    papers_set=[]
    for refer in top_refer:
        #找到top_refer的key
        master_keys=papers[(papers.title==refer)&(papers.refer_key is not None)].refer_key
        papers_set.append(refer)
        for master_key in master_keys:
            try:
                master_title=papers[papers.key==master_key].title
                papers_set.append(master_title.values[0])
                papers_relation.append((master_title.values[0],refer))
            except:
                continue
    return papers_set,papers_relation

def generate_col_double_tuple(papers_df,col):
    '''
    生成多元素二元组关系
    :param papers_df:
    :param col:
    :return:
    '''
    # 通过papers_df读取maseter文章
    master_papers = papers_df[papers_df.refer_key.isnull()]
    master_papers_key = master_papers.key.values.tolist()

    # 节点关系集合
    nodes_relation = set()
    # 节点集合
    nodes_set = set()

    node_array = []  # [a,b,c]
    refer_node_array = []  # [d,e,f]

    # [a,d] [a,e]....
    # 循环遍历所有的主节点
    for key in master_papers_key:
        # 获取主节点keyword集合
        master_nodes = papers_df[papers_df.key == key][col].values.tolist()
        # 获取refer keyword集合
        refer_nodes = papers_df[papers_df.refer_key == key][col].values.tolist()

        # 遍历每个节点的族谱这字符串
        for node in master_nodes:
            # 去掉尾巴的|，用|分割并添加到集合中
            node_array.append(str(node).strip('|').split('|')[0])
        for refer_node in refer_nodes:
            # 去掉尾巴的|，用|分割并添加到集合中
            refer_node_array.append(str(refer_node).strip('|').split('|')[0])

        for i in node_array:
            for j in refer_node_array:
                if i != None or j != None:
                    nodes_relation.add((i, j))
                    nodes_set.add(i)
                    nodes_set.add(j)

    return nodes_relation, nodes_set

def generate_col_dataset(col):
    '''
    产生多元素数据集（一般用来频繁模式挖掘）
    :param col:
    :return:
    '''
    node_dataset = []
    for node in papers[col].values:
        node_dataset.append(str(node).split('|'))
    return node_dataset

def generate_frequency(papers_df,col,deadline):
    '''
    生成字段频率图
    :param papers_df:
    :param col:
    :param deadline:
    :return:
    '''
    field_frequency_list = []
    #分割和组合
    for field_values in papers_df[col].values:

        for field in str(field_values).split('|'):

            if field is not None:
                field_frequency_list.append(field)

    #生成dataFrame进行组合计数
    value_count=pd.DataFrame(field_frequency_list)[0].value_counts()
    value_count=value_count[value_count>deadline]
    #value_count.plot(kind='bar')
    plt.barh(len(value_count)-np.arange(len(value_count)),value_count.values)
    plt.yticks(len(value_count)-np.arange(len(value_count)),value_count.index)
    plt.show()

    return  value_count.index.tolist()

def plot_know_graph(node_set,node_relation):

    '''
    画出知识图谱
    :param node_set:
    :param node_relation:
    :return:
    '''
    nodes=list(node_set)
    relation=list(node_relation)
    gh=nx.DiGraph()
    gh.add_nodes_from(nodes)
    for row in relation:
        gh.add_edge(row[0],row[1])

    nx.draw(gh, pos=None, ax=None, hold=None,with_labels=True,node_size=1000,node_color=['w'],font_color='r')
    plt.show()

def generate_word_cloud(papers_df,col):
    '''
    生成词云
    :param papers_df:
    :param col:
    :return:
    '''
    words=""
    for i in papers_df[col].values:
        words+="|"+str(i)

    wc=WordCloud(font_path='C:/Windows/Fonts/simsun.ttc',
                 width=800,
                 height=400,
                 margin=2,
                 ranks_only=None,
                 prefer_horizontal=0.9,
                 mask=None,
                 scale=1,
                 color_func=None,
                 max_words=200,
                 min_font_size=4,
                 stopwords=None,
                 random_state=None,
                 background_color='white',
                 max_font_size=None,
                 font_step=1,
                 mode='RGB',
                 relative_scaling=0.5,
                 regexp=None,
                 collocations=True,
                 colormap=None,
                 normalize_plurals=True)
    wc_im=wc.generate(words)
    plt.imshow(wc_im)
    plt.show()

def plot_high_frequency_keyword():

    kewyword_list=generate_frequency(papers,'keywords',40)
    #删除nan
    if len(kewyword_list)==1:
        del kewyword_list[1]
    index_= papers.keywords.apply(lambda x: len(set(kewyword_list)&set(str(x).split('|')))!=0)
    keywords_relation,keywords_set=generate_col_double_tuple(papers[index_],'keywords')
    plot_know_graph(keywords_set,keywords_relation)

def plot_high_frequency_author():
    '''
    画出高频率的作者关系图
    :return:
    '''
    #找出高频率的作者集合
    kewyword_list=generate_frequency(papers,'authors',20)
    #在dataframe中找到包含这些作者的集合
    index_= papers.authors.apply(lambda x: len(set(kewyword_list)&set(str(x).split('|')))!=0)
    #产生二元组
    keywords_relation,keywords_set=generate_col_double_tuple(papers[index_],'authors')
    #画出知识图谱
    plot_know_graph(keywords_set,keywords_relation)

def plot_author_feature(who):
    # 获取含有该作者所在的数据
    index_=papers['authors'].apply(lambda x :who in str(x))
    who_papers=papers[index_]

    #生成该作者的关键词图
    generate_word_cloud(who_papers,'keywords')

    #画出该作者的出版年限频图
    who_papers['year'].value_counts().plot(kind='bar')
    plt.show()

    #画出出版报的分布

    generate_frequency(who_papers, 'journal', 0)


if __name__=='__main__':

    print('-------------------------程序开始-------------------------------------')
    print('-------------------------高频关键词知识图谱----------------------------')
    #plot_high_frequency_keyword()
    print('-------------------------高频作者知识图谱----------------------------')
    #plot_high_frequency_author()
    print('-------------------------人物分析------------------------------------')
    #plot_author_feature('刘宝珺')

    print ('-------------------------作者顺序分析---------------------------------')


    dataSet = generate_col_dataset('authors')

    C1 = createC1(dataSet)

    L, supportData = Apiroir(dataSet, 0.01)

    # 筛选频率比较高的二元组以上的
    high_frequency_author_tuple = []
    for support in supportData:
        if len(support) > 1:
            if supportData[support] > 0.02:
                high_frequency_author_tuple.append(support)
                print(support,'频繁度为：',supportData[support] )
    new_papers = pd.DataFrame()
    for tuple_ in high_frequency_author_tuple:
        index_ = papers.authors.apply(lambda x: len(set(tuple_) & set(str(x).split('|'))) == len(tuple_))
        new_papers = new_papers.append(papers[index_])

    '''
        names=['张金亮','毛凤鸣','常象春','刘宝珺','沈凤']
    
        first_author_prob=[]
        for name in names:
            first_author_prob.append(new_papers[new_papers.authors.apply(lambda x: str(x).split('|')[1] == name)].shape[0] / len(new_papers))
        pd.DataFrame(first_author_prob,index=names).plot(kind='bar',title='频繁项集中第二作者的概率')
        plt.show()
    
    '''

    relation,set=generate_col_double_tuple(new_papers,'authors')

    plot_know_graph(set,relation)




    '''     
    ('张金亮', '毛凤鸣')
    ('常象春', '刘宝珺', '张金亮')
    ('常象春', '毛凤鸣')
    ('张金亮', '常象春', '刘宝珺', '毛凤鸣')
    ('张金亮', '刘宝珺')
    ('常象春', '刘宝珺')
    ('张金亮', '常象春', '毛凤鸣')
    ('刘宝珺', '张金亮', '毛凤鸣')
    ('常象春', '张金亮')
    ('常象春', '刘宝珺', '毛凤鸣')
    ('沈凤', '张金亮')
    ('刘宝珺', '毛凤鸣')   
    
    print (papers.columns)
    plot_author_feature('张金亮')
    dataSet=generate_col_dataset('authors')
        
            C1=createC1(dataSet)
        
        
            L,supportData=Apiroir(dataSet,0.01)
        
            #筛选频率比较高的二元组以上的
            high_frequency_author_tuple=[]
            for support in supportData:
                if len(support)>1:
                    if supportData[support]>0.02:
                        high_frequency_author_tuple.append (support)
                        print(support)
            new_papers=pd.DataFrame()
            for tuple_ in  high_frequency_author_tuple:
                index_=papers.authors.apply(lambda x:len(set(tuple_)&set(str(x).split('|')))==len(tuple_))
                new_papers=new_papers.append(papers[index_])
            print (new_papers.shape)
            print(new_papers[new_papers.authors.apply(lambda  x:str(x).split('|')[1]=='张金亮')].shape[0]/len(new_papers))
    '''




    
