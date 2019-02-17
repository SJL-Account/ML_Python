import re
import xml
import os
import requests
from selenium import webdriver
import time
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mlb
import uuid
from wordcloud import WordCloud


class cnki_craw:

    def __init__(self):
        os.chdir('Caj_张金亮')
        self.driver = webdriver.Chrome()
        self.cnki_url = 'http://rss.cnki.net/rss/Getinfobydoi.aspx?'
        self.url_list = []
        self.paper_dict = {'key': [], 'title': [], 'authors': [], 'authors_orgn': [], 'keywords': [], 'year': [], 'journal': [],'url': [], 'refer_key': [], 'abstract_text': [], 'download': [], 'page': []}

    def craw_caj_doi(self):

        for i, file_name in enumerate(os.listdir()):

            file = open(file_name, 'rb')

            try:#有的可能没有doi

                # 在文件的最后一行
                final_line = file.readlines()[-1].decode()
                #正则表达式匹配
                pattern = re.compile('<DOI>(.+)</DOI>')
                #找到doi字符串
                DOI = pattern.findall(final_line)[0]
            except:
                continue

            print('正在获取doi...', i, file_name, DOI, self.cnki_url + 'doi=' + DOI)

            self.url_list.append(self.cnki_url + 'doi=' + DOI)

    def scrapy_web_info(self,url):
        '''
        给一个链接，返回所有信息
        '''
        self.driver.get(url)
        time.sleep(2)
        # 文章标题
        title_text = self.driver.find_element_by_class_name('title').text
        # 作者列表
        author_list = self.driver.find_element_by_class_name('author').find_elements_by_css_selector('a')
        # 研究机构表
        author_orgn_list = self.driver.find_element_by_class_name('orgn').find_elements_by_css_selector('a')
        # 关键词表
        key_words_list = self.driver.find_elements_by_xpath("//label[@id='catalog_KEYWORD']/following-sibling::a")
        # 年限表
        year_text = self.driver.find_element_by_xpath('//*[@id="mainArea"]/div[3]/div[3]/div[2]/div[2]/p[3]/a').text.split('年')[0]
        # 期刊名称
        journal_text = self.driver.find_element_by_xpath('//*[@id="mainArea"]/div[3]/div[3]/div[2]/div[2]/p[1]/a').text
        #摘要正文
        abstract_text = self.driver.find_element_by_id('ChDivSummary').text
        #下载量
        download = self.driver.find_element_by_xpath('//*[@id="mainArea"]/div[3]/div[3]/div[1]/div[4]/div[1]/div[1]/span[1]/b').text
        #页数
        page = self.driver.find_element_by_xpath('//*[@id="mainArea"]/div[3]/div[3]/div[1]/div[4]/div[1]/div[1]/span[3]/b').text

        authors = ''
        authors_orgn = ''
        keywords = ''
        for author in author_list:
            authors += author.text + '|'
        for orgn in author_orgn_list:
            authors_orgn += orgn.text + '|'
        for keyword in key_words_list:
            keywords += keyword.text.strip(';') + '|'

        return title_text, authors.strip('|'), authors_orgn.strip('|'), keywords.strip('|'), year_text, journal_text, abstract_text, download, page

    def craw_info_by_iod(self):
        #paper数据结构
        paper_dict = {'key': [], 'title': [], 'authors': [], 'authors_orgn': [], 'keywords': [], 'year': [], 'journal': [],'url': [], 'refer_key': [], 'abstract_text': [], 'download': [], 'page': []}
        for i,url in enumerate(self.url_list):
            print('-----------------------------------------------------华丽的分割线--------------------------------------------------------------------')

            try: #有的网页有问题会停止爬取
                title_text, authors, authors_orgn, keywords, year, journal, abstract_text, download, page = self.scrapy_web_info(url)
                #主文章
                print(i,title_text, authors, authors_orgn, keywords, year, journal, download, page)
                #生成idkey
                maseter_key = str(uuid.uuid1())
                #字段填充
                self.paper_dict['key'].append(maseter_key)
                self.paper_dict['title'].append(title_text)
                self.paper_dict['authors'].append(authors)
                self.paper_dict['authors_orgn'].append(authors_orgn)
                self.paper_dict['keywords'].append(keywords)
                self.paper_dict['year'].append(year)
                self.paper_dict['journal'].append(journal)
                self.paper_dict['url'].append(url)
                self.paper_dict['refer_key'].append('')
                self.paper_dict['abstract_text'].append(abstract_text)
                self.paper_dict['download'].append(download)
                self.paper_dict['page'].append(page)
                #请求源代码，获取参考文献链接
                resp = requests.get(url=url)
                # 定义正则表达式匹配参考文献路径
                pattern = re.compile("LoadFile\('framecatalog_CkFiles','(.+)'\)")

                # 参考文献可能好几页
                print('开始爬取参考文献...')
                for i in range(5):
                    # 定义参考文献链接
                    reference_url = 'http://kns.cnki.net' + pattern.findall(resp.text)[0] + '&page=' + str(i)
                    # 发起浏览
                    self.driver.get(reference_url)
                    time.sleep(2)
                    # 找到所有作者标签
                    refer_paper_urls = [element.get_attribute('href') for element in self.driver.find_elements_by_xpath('/html/body/div[1]/ul/li/a[@target="kcmstarget"]')]

                    if len(refer_paper_urls) == 0:
                        print('没有参考文献，退出...')
                        break;
                    for refer_paper_url in refer_paper_urls:
                        print(refer_paper_url)
                        #参考文献信息
                        title_text, authors, authors_orgn, keywords, year, journal, abstract_text, download, page = self.scrapy_web_info(refer_paper_url)
                        print(title_text, authors, authors_orgn, keywords, year, journal, download, page)
                        #填充参考文献
                        self.paper_dict['key'].append(str(uuid.uuid1()))
                        self.paper_dict['title'].append(title_text)
                        self.paper_dict['authors'].append(authors)
                        self.paper_dict['authors_orgn'].append(authors_orgn)
                        self.paper_dict['keywords'].append(keywords)
                        self.paper_dict['year'].append(year)
                        self.paper_dict['journal'].append(journal)
                        self.paper_dict['url'].append(url)
                        self.paper_dict['refer_key'].append(maseter_key)
                        self.paper_dict['abstract_text'].append(abstract_text)
                        self.paper_dict['download'].append(download)
                        self.paper_dict['page'].append(page)

            except:
                continue

    def save_csv(self):
        papers=pd.DataFrame(self.paper_dict)
        papers.to_csv('papers_.csv',index=None)

if  __name__ =='__main__':
    c_craw=cnki_craw()
    c_craw.craw_caj_doi()
    c_craw.craw_info_by_iod()
    c_craw.save_csv()