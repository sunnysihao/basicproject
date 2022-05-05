# _*_ coding=: utf-8 _*_
import numpy as np

bin_url = "D:\标注\px支撑\脚本分项目\\3D点云\\bin\\1634862979725228000.bin"
pcd_url = "D:\标注\px支撑\脚本分项目\\3D点云\pcd\\1634862979725228000.pcd"

# 读取点云
points = np.fromfile(bin_url, dtype="float32").reshape((-1, 4))

# 写文件句柄
handle = open(pcd_url, 'a')

# 得到点云点数
point_num = points.shape[0]

# pcd头部（重要）
handle.write(
    '# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')
string = '\nWIDTH ' + str(point_num)
handle.write(string)
handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
string = '\nPOINTS ' + str(point_num)
handle.write(string)
handle.write('\nDATA binary_compressed')

# 依次写入点
for i in range(point_num):
    string = '\n' + str(points[i, 0]) + ' ' + str(points[i, 1]) + ' ' + str(points[i, 2])
    handle.write(string)
handle.close()