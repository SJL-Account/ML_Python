#coding:utf-8
from numpy import *

def import_txt(file_name,future_nums):

    f=open(file_name)
    lines=f.readlines()

    m=len(lines)
    #m*1
    labels=[]
    #m*6
    future_values=zeros((m,future_nums))

    index=0

    for line in lines:
        line=line.strip()
        values=line.split('\t')
        future_values[index,:]=values[:future_nums]
        labels.append(int(values[-2]))
        index+=1
    return future_values,labels

if __name__=='__main__':
   inX,x_lable=import_txt('test.txt',6)
   print inX
    

    


