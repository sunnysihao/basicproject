# _*_ coding=: utf-8 _*_
import os
import json

dirses = []
for root, dirs, files in os.walk("C:/Users/EDY/Desktop/3548_2048/3548_2048"):
    if dirs not in dirses:
        dirses.append(dirs)
print(dirses)
if os.path.exists("C:/Users/EDY/Desktop/3548_2048"):
    print("cunzai")
else:
    print("没有")
print(os.path.split(r"C:\Users\EDY\Desktop\3548_2048\3548_2048\3d_url\ladybug_000009.pcd"))
data = {'a':123, 'b':000, 'c':'hellow'}
json_data = json.dumps(data, sort_keys=True, indent=4, separators=(',', ':'))
print(json_data)
pydata = json.loads(json_data)
print(pydata)
print(dirses.index(['3d_img0', '3d_img1', '3d_img2', '3d_img3', '3d_url', 'camera_config']))
