#coding:utf-8

from bs4 import BeautifulSoup
import requests
import io
import re
request_url='https://movie.douban.com/chart'

response=requests.get(request_url)

#xpath= response.xpath('//*[@id="content"]/div/div[1]/div/div/table[1]/tbody/tr/td[2]/div/a')

bs= BeautifulSoup (response.text,"lxml")
'''
print bs.head.contents


for i in  bs.head.children:
    print i
    
'''
'''
with open('body.json', 'w+') as f:
    for i in bs.body.descendants:
       print i

'''

'''
for i in bs.body.strings:
       print i.strip().strip('\n')
'''
'''
for i in bs.body.stripped_strings:
       print i

'''
'''
for i in bs.find_all(['a','span']):
    print i.string
'''

print type(bs.find_all(['a','span']))
print type(bs.select('a'))
print type(bs.body.strings)
print type(bs.body.descendants)
#print  bs.finda_all(id='collect_form_11600078')
#print bs.find_all(href=re.compile('https://movie.douban.com/subject/\d+'))
'''
for i in bs.find_all(href=re.compile('https://movie.douban.com/subject/\d+')):
    if i['title'] :
        print i['title']
'''
'''
for i in bs.select('div .pl2'):
    print i
'''

'''
for i in bs.select('div .item'):
    print  i.a['href']
'''





#print bs.title

#print bs.title.string




'''

item= bs.find_all('a')

for i in item:
    print i.string


item= bs.select('.pl2')



for i in item:
    print i.a.text.strip()
'''
