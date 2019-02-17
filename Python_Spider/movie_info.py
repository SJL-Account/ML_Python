#coding:utf-8

from bs4 import BeautifulSoup
import requests
import io
import re
request_url='https://movie.douban.com/chart'


response=requests.get(request_url)

bs= BeautifulSoup (response.text,"lxml")

links=[]

for i in bs.select('div .item'):
    links.append(i.a['href'])

for link in links:
    response=requests.get(link)
    bs = BeautifulSoup(response.text, "lxml")
    for info in bs.select('#info'):
        for span in info.find_all(class_="pl"):
            print span

