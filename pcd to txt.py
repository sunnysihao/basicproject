# _*_ coding=: utf-8 _*_
import os
#定义一个三维点类
class Point(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
points = []
filename = '/home/gcc/***'
#读取pcd文件,从pcd的第12行开始是三维点
with open(filename+'.pcd') as f:
    for line in f.readlines()[11:len(f.readlines())-1]:
        strs = line.split(' ')
        points.append(Point(strs[0], strs[1], strs[2].strip()))
##strip()是用来去除换行符
##把三维点写入txt文件
fw = open(filename+'.txt','w')
for i in range(len(points)):
     linev = points[i].x+" "+points[i].y+" "+points[i].z+"\n"
     fw.writelines(linev)
fw.close()