import requests
from PIL import Image
import json
from io import BytesIO

'''
request_url='http://www.baidu.com'

response=requests.get(request_url)

print response.text

print response.encoding

print response

print dir(requests)

'''

'''
request_url='http://www.baidu.com'

params={'k1':'v1','k2':'v2'}

response=requests.get(request_url,params)

print response.url
'''

'''
request_url='https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1503243285233&di=522d2ccd31421eeeb28d725c3c7978d4&imgtype=0&src=http%3A%2F%2Fimg.tupianzj.com%2Fuploads%2Fallimg%2F160810%2F9-160Q0161036-50.jpg'

response=requests.get(request_url)

image= Image.open(BytesIO(response.content))

image.save('meinv.jpg')
'''

'''
request_url='https://github.com/timeline.json'

response= requests.get(request_url)

print response.text
'''

'''
request_url='https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1503243285233&di=522d2ccd31421eeeb28d725c3c7978d4&imgtype=0&src=http%3A%2F%2Fimg.tupianzj.com%2Fuploads%2Fallimg%2F160810%2F9-160Q0161036-50.jpg'

response=requests.get(request_url,stream=True)

with open('meinv3.jpg','wb+') as f:
    for chunck in response.iter_content(1024):
        f.write(chunck)

'''

'''
request_url='http://httpbin.org/post'

form={'username':'sunjinlong','password':'123456'}

data= json.dumps(form)

response=requests.post(request_url,data=form)

print response.text

response= requests.post(request_url,data=data)

print response.text
'''

'''
request_url='http://www.baidu.com'

response=requests.get(request_url)

cookies= response.cookies

for c,v in cookies.get_dict().items():
    print c,v

'''

'''
request_url='http://httpbin.org/get'

cookies={'c':'v','c1':'v1'}

response=requests.get(request_url,cookies=cookies)

print response.text
'''

'''
request_url='http://github.com'

response=requests.get(request_url,allow_redirects=True)

print response.url

print response.status_code

print response.history
'''

























