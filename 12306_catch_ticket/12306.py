#coding:utf-8

import re
import requests
from pprint import pprint
import datetime
import thread
import time
from Tkinter import *
root = Tk()
root.title("catch 12306")
root.geometry('1000x200')
k=0

config={'start_station':'BJP','end_station':'HBB','date':['2018-02-06','2018-02-07','2018-02-08','2018-02-09'],'timing_hours':14,'start_time_hour':9,'train_type':u''}


#def monitor(delay):
while True:
    try :
        for date in config['date']:

            response=requests.get(url='https://kyfw.12306.cn/otn/leftTicket/queryZ?leftTicketDTO.train_date='+date+'&leftTicketDTO.from_station='+config['start_station']+'&leftTicketDTO.to_station='+config['end_station']+'&purpose_codes=0X00')
            data=response.json()['data'][u'result']
            data_len=len(data)
            for i in range(data_len):
                Info= data[i].split(u'|')
                train= Info[3]
                start_time= Info[8]
                end_time= Info[9]
                timing= Info[10]
                train_type=train[0]
                YW= Info[28]
                YZ= Info[29]
                timing_hours=int(timing.split(u':')[0])
                start_time_hour=int(start_time.split(u':')[0])
                #print "----------------------------------------------------------------------------------------------------------"
                #print '正在进行抓取第'+str(k)+'次'
                k+=1
                Msg= "data:"+date+"     train:"+train+"    start_time:"+start_time+"    end_time:"+end_time+"    timing:"+timing+"    YW:"+YW+"    YZ:"+YZ+"catch time:"+str(datetime.datetime.now())
                if timing_hours<14 and (start_time_hour>9 or start_time_hour<=2) and train_type!=u'G' and train_type!=u'D' :
			if YW !=u'' and YZ !=u'' and (YW!=u'无' or YZ!=u'无' ) :
                    		l = Label(root, text=Msg, bg="red", font=("Arial", 12))
                    		l.pack(side=TOP)
                    		root.mainloop()

                	print Msg
    except:
        print
    time.sleep(10)








