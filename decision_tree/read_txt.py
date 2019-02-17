#coding:utf-8
from numpy import *

def import_txt(file_name,future_nums):

    f=open(file_name)
    lines=f.readlines()

    m=len(lines)
    #m*1
    labels=[]
    #m*6
    future_values=[]

    index=0

    for line in lines:
        line=line.strip()
        values=line.split('\t')
        future_values[index,:]=values[:future_nums]
        labels.append(int(values[-1]))
        index+=1
    return np.array(future_values),labels


    

    


