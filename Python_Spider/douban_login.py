#coding:utf-8
import requests
from bs4 import BeautifulSoup
import re
#import html5lib
headers = {'user-agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36'}
login_url='https://accounts.douban.com/login'

form_data={
'source':'index_nav',
'redir':'https://www.douban.com/',
'form_email':'17645013991',
'form_password':'db68351560',
#'captcha-solution':'filed',
#'captcha-id':'HP6v7b4bI4nA7KHx0UZ5CxpL:en',
'login':u'登录'
}

reponse=requests.post(login_url,data=form_data,headers=headers)

content=reponse.content


bs= BeautifulSoup(content,'html5lib')

captcha= bs.find('img',id='captcha_image')

if captcha:

    re_captcha_id=r'<input type="hidden" name="captcha-id" value=".*?"/>'
    captcha_id =re.findall(re_captcha_id,content)
    print captcha['src']
    print captcha_id

    captcha_id_text=input('请输入id')
    captcha=input('请输入验证码')

    form_data['captcha-solution']=captcha_id_text

    form_data['captcha-id']=captcha

reponse=requests.post(login_url,data=form_data)

print reponse.text





