#coding:utf-8

import sqlite3

conn=sqlite3.connect('test.db') #没有自动创建一个

create_sql='create table company( id int primary key not null,enp_name text not null)'

#conn.execute(create_sql)

insert_sql = 'insert into company values(?,?)'

conn.execute(insert_sql,(100,'SJL'))

cursors=conn.execute('select * from company')

for i in cursors:
    print i[1]
conn.close()

