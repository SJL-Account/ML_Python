import pandas as pd
import jieba
papers=pd.read_csv('Caj_张金亮/papers_all.csv',encoding='gbk')

print (papers.abstract_text[0],papers.keywords[0])

print(jieba.cut(papers.abstract_text[0]))