#_*_coding:gbk_*_

import urllib2
import re


def url_load(url):
    '''
    @brief:ͨ��url��ȡ��ҳԴ����
    @param:urlͳһ��Դ��λ��
    @return ��̬ҳԴ��
    '''
    user_agent='Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0'

    headers={'User-Agent':user_agent}

    req=urllib2.Request(url,headers=headers)
    

    response=urllib2.urlopen(req)

    html=response.read()

    return html.decode('utf-8','ignore').encode('gbk','ignore')

def save_page(file_name,result):
    '''
    ����Դ�����ļ�
    '''
    print '���ڱ���'+file_name+'....'
    
    #���ļ�

    f=open(file_name,'w+')
    
    #��д�ļ�

    f.write(result)
    
    #�ر��ļ�

    f.close()    


def tieba_spider(tieba_name):
    '''
    ��ȡ��ҳ
    '''
    basic_url=str(raw_input('���������ɵ�ַ'))

    page_start=int(raw_input('��������ʼҳ��:'))

    page_end=int(raw_input('���������ҳ��:'))

    for i in range(page_start,page_end):

        page_No=(i-1)*50
        
        url=basic_url+str(page_No)
        
        html=url_load(url)

        pattern=re.compile(r'http://pan.baidu.com/share/link\?shareid=.{1,100} ��ȡ��:.{4}',re.S)

        item_list=pattern.findall(html)

        for item in item_list:
            
               print item


if __name__=='__main__':

    tieba_spider('�ٶ����̰�')

    
