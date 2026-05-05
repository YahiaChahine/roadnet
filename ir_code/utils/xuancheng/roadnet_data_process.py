#<location netOffset="-666136.78,-3424078.10" convBoundary="-3440.06,-2995.04,7387.50,5502.03" origBoundary="118.739160,30.938231,118.759965,30.967737" projParameter="+proj=utm +zone=50 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"/>

import sumolib
import pyproj
from pyproj import Proj, transform
import geopandas as gpd
import pandas as pd
import pickle

"""from gis to sumo"""
## 1.获取sumo中交叉口的坐标
net = sumolib.net.readNet('./utils/xuancheng/road-m/xuancheng1116_2.net.xml')
junctions = net.getNodes()
junctions_dict ={j.getID():(j.getCoord()) for j in junctions}

## 2.转换sumo坐标至经纬度
def convert_coordinates(junctions_dict):
    proj_utm = Proj("+proj=utm +zone=50 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    proj_latlon = Proj(proj='latlong', datum='WGS84')
    net_offset_x = -666136.78
    net_offset_y = -3424078.10

    converted_dict = {}
    for key, value in junctions_dict.items():
        known_x, known_y = value
        real_x = known_x - net_offset_x
        real_y = known_y - net_offset_y
        longitude, latitude = transform(proj_utm, proj_latlon, real_x, real_y)
        converted_dict[key] = (longitude, latitude)

    return converted_dict
converted_junctions = convert_coordinates(junctions_dict)
print(converted_junctions)

# 保存字典到文件
with open('./utils/xuancheng/road-m/junctions_dict.pkl', 'wb') as f:
    pickle.dump(converted_junctions, f)
# # 读取字典文件
# with open('./utils/xuancheng/road-m/junctions_dict.pkl', 'rb') as f:
#     converted_junctions = pickle.load(f)
# print(converted_junctions)

# 3.与gis文件中交叉口进行匹配
data = gpd.read_file('./utils/xuancheng/road-m/Export_Output_3.shp')
field_names = data.columns.tolist()
filter_data = data[['fnode_', 'fnodex', 'fnodey', 'tnode_','tnodex', 'tnodey', 'length','roadid']]
filter_data_df = pd.DataFrame(filter_data)
# print(filter_data_df)
junctions_dict_2 ={}
for index, row in filter_data_df.iterrows():
    junctions_dict_2[row['fnode_']] = (row['fnodex'], row['fnodey'])
    junctions_dict_2[row['tnode_']] = (row['tnodex'], row['tnodey'])
# print(junctions_dict_2)

#匹配converted_junctions和junctions_dict_2
def match_junctions(converted_junctions, junctions_dict_2):
    matched_dict = {i:[] for i in converted_junctions.keys()}
    for key, value in converted_junctions.items():
        for key2, value2 in junctions_dict_2.items():
            if abs(value[0] - value2[0]) < 0.0001 and abs(value[1] - value2[1]) < 0.0001:
                matched_dict[key].append(key2)
                continue
    return matched_dict
matched_junctions = match_junctions(converted_junctions, junctions_dict_2)
#检测matched_junctions列表中的元素个数>2的元素
for key, value in matched_junctions.items():
    if len(value) >1:
        ## gis文件存在错误,['34180200007346', '-34180200007346']是同一个交叉口
        if value[0] in value[1]:
            matched_junctions[key]= [value[0]]
            print('存在错误的',key, value)
        elif  value[1] in value[0]:
            matched_junctions[key]= [value[1]]
            print('存在错误的',key, value)
print('---------------------------------------------------------------')
matched_junctions_copy = matched_junctions.copy()
for key, value in matched_junctions_copy.items():
    if len(value) > 1:
        print('仍旧一对多的:', key, value)
    elif len(value) == 0:
        matched_junctions.pop(key)
# print(len(converted_junctions))
# print(len(junctions_dict_2))
# print(matched_junctions)
print(f'gis中匹配上的是{len(matched_junctions)}%{len(junctions_dict_2)}')


"""from gis to cityflow"""
data = gpd.read_file('./road-m/Export_Output_3.shp')
field_names = data.columns.tolist()
filter_data = data[['fnode_', 'fnodex', 'fnodey', 'tnode_','tnodex', 'tnodey', 'length','roadid']]
filter_data_df = pd.DataFrame(filter_data)
print(filter_data_df)

for index, row in filter_data_df.iterrows():
    if '-' in row['fnode_']:
        row['fnode_'] = row['fnode_'].replace('-', '')
        filter_data_df.loc[index, 'fnode_'] = row['fnode_']
    if '-' in row['tnode_']:
        row['tnode_'] = row['tnode_'].replace('-', '')
        filter_data_df.loc[index, 'tnode_'] = row['tnode_']
    if '-' in row['roadid']:
        row['roadid'] = row['roadid'].replace('-', '')
        filter_data_df.loc[index, 'roadid'] = row['roadid']
print(filter_data_df)
