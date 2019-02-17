#coding:utf-8

import requests

request_url='https://movie.douban.com/chart'

response=requests.get(request_url)

a=response.xpath('//*[@id="link-download"]/li/text()')

print a