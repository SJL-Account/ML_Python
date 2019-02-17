# -*- coding:utf-8 -*-
from PIL import Image
import io
ascii_char = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")

#转字符串
def get_char(r,g,b,alpha=256):
    if(alpha==0):
        return  ' '

    #字符集合长度
    length=len(ascii_char)

    #灰度值
    grey =int(0.2126 * r + 0.7152 * g + 0.0722 * b)

    #rgb值在字符串上的大致位置
    unit =(256.0+1.0)/length

    return ascii_char[int(grey/unit)]

if __name__=='__main__':

    img= Image.open('xiaoyu.jpg')
    txt=''
    #自定义位置，等比例放大缩小
    img=img.resize((80,73),Image.NEAREST)

    height=img.height
    width=img.width

    for j in range(height):
        for i in range(width):
            txt+=get_char(*img.getpixel((i,j)))


        txt+='\n'
    print txt

    f=open('1.txt','w')

    f.write(txt)

    f.close()


