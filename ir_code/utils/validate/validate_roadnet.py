import geopandas as gpd
from pyproj import Proj, transform,Transformer
import pandas as pd
import json
from math import  sqrt,isnan
from shapely.geometry import LineString,Point
import matplotlib.pyplot as plt
import os
import networkx as nx
import numpy as np
import fiona
# pd.set_option('display.max_rows', None)


### shp文件经纬度转坐标系
proj_utm = Proj("+proj=utm +zone=50 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
proj_latlon = Proj(proj='latlong', datum='WGS84')
net_offset_x = -666136.78
net_offset_y = -3424078.10

def convert_coordinates_reverse(junctions_dict):
    transformer = Transformer.from_proj(proj_latlon, proj_utm)
    converted_dict = {}
    for key, value in junctions_dict.items():
        longitude, latitude = value
        real_x, real_y = transformer.transform(longitude,latitude)
        known_x = real_x + net_offset_x
        known_y = real_y + net_offset_y
        converted_dict[key] = (known_x, known_y)
    return converted_dict



def cal_length(road):
    length = 0
    for i in range(len(road['points'])-1):
        length += sqrt((road['points'][i]['x']-road['points'][i+1]['x'])**2+(road['points'][i]['y']-road['points'][i+1]['y'])**2)
    return length



if __name__ =='__main__':
    ### 读取Shapefile文件
    shapefile_path = "Export_Output_3.shp" 
    shapefile_path = os.path.join(os.path.dirname(__file__), shapefile_path)
    shp_gdf = gpd.read_file(shapefile_path)
    field_names = shp_gdf.columns.tolist()

    ### 保留属性,提取坐标
    filter_data = shp_gdf[['fnode_', 'fnodex', 'fnodey', 'tnode_', 'tnodex', 'tnodey', 'length', 'roadid']]
    filter_data_df = pd.DataFrame(filter_data)
    filter_data_df = filter_data_df[filter_data_df['roadid'] != 'NULL']
    junctions_dict_2 ={}
    for index, row in filter_data_df.iterrows():
        junctions_dict_2[row['fnode_']] = (row['fnodex'], row['fnodey'])
        junctions_dict_2[row['tnode_']] = (row['tnodex'], row['tnodey'])
    junctions_dict_2 = dict(sorted(junctions_dict_2.items(), key=lambda x: x[0]))
    converted_junctions = convert_coordinates_reverse(junctions_dict_2 )


    ### 读取CityFlow文件
    cityflow_path = "./roadnet_xuancheng241030.json"
    cityflow_path = os.path.join(os.path.dirname(__file__), cityflow_path)
    with open(cityflow_path, 'r') as f:
        cityflow_data = json.load(f)
    intersection_data = cityflow_data['intersections']
    intersections_dict = {}
    for intersection in intersection_data:
        intersections_dict[intersection['id']] = (intersection['point']['x'], intersection['point']['y'])

    key1 = list(converted_junctions.keys())
    ### key1-key2
    key12 = list(set(key1).difference(set(intersections_dict.keys())))
    key2 = list(intersections_dict.keys())
    ### key2-key1
    key21 = list(set(key2).difference(set(converted_junctions.keys())))
    key = list(set(key1).intersection(set(key2)))
    key =list(sorted(key,key=lambda x:x))


    #### 计算两路网间平均节点距离
    diff = {}
    for k in key:
        diff[k]=(converted_junctions[k][0]-intersections_dict[k][0],converted_junctions[k][1]-intersections_dict[k][1])
    mean_diff = []
    num = 0
    for k in key:
        dis = sqrt(diff[k][0]**2+diff[k][1]**2)
        #判断是否为nan,如果是nan则不计入计算
        if not isnan(dis):
            mean_diff.append(dis)
            num+=1
    print("两路网间平均节点距离：",sum(mean_diff)/num)



    ### 空间一致性检查
    # Create geometries for both networks
    cityflow_geometries = []
    for road in cityflow_data['roads']:
        cityflow_geometries.append(LineString([(point['x'], point['y']) for point in road['points']]))
    cityflow_gdf = gpd.GeoDataFrame(geometry=cityflow_geometries)
    cityflow_gdf.set_crs(proj_utm.srs, inplace=True)
    
    # Transform shapefile to the same coordinate system and apply offset
    shp_gdf = shp_gdf.to_crs(proj_utm.srs)
    shp_gdf['geometry'] = shp_gdf['geometry'].translate(xoff=net_offset_x, yoff=net_offset_y)

    # Create a high-quality figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=500)

    # Plot with better styling
    shp_gdf.geometry.plot(ax=ax, color='blue', linewidth=0.8, alpha=0.7, label='OpenStreetMap')
    cityflow_gdf.plot(ax=ax, color='red', linewidth=0.8, alpha=0.7, label='CityFlow')

    # Enhance the plot appearance
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 保持自然的坐标系，不用invert_yaxis
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))

    ax.set_xlabel('X Coordinate (m)', fontsize=14)
    ax.set_ylabel('Y Coordinate (m)', fontsize=14)
    ax.set_title('Comparison of Road Networks', fontsize=18, fontweight='bold')

    # Add grid for better reference
    ax.grid(True, linestyle='--', alpha=0.4)

    # Adjust limits to focus on the area with data
    bounds = shp_gdf.total_bounds
    cityflow_bounds = cityflow_gdf.total_bounds
    xmin = min(bounds[0], cityflow_bounds[0])
    ymin = min(bounds[1], cityflow_bounds[1])
    xmax = max(bounds[2], cityflow_bounds[2])
    ymax = max(bounds[3], cityflow_bounds[3])

    # Add some padding to the bounds
    padding = 0.05 * max((xmax - xmin), (ymax - ymin))
    ax.set_xlim(xmin - padding, xmax + padding)
    ax.set_ylim(ymin - padding, ymax + padding)
    ax.set_aspect('equal')
    ax.legend(loc='best', frameon=True, framealpha=0.9, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.dirname(__file__)+'/map.png', bbox_inches='tight')



    #### 节点匹配度指标
    # G_shapefile = nx.Graph()
    # G_cityflow = nx.Graph()
    # for index, row in filter_data_df.iterrows():
    #     G_shapefile.add_edge(converted_junctions[row['fnode_']], converted_junctions[row['tnode_']], weight=row['length'])
    # for road in cityflow_data['roads']:
    #     G_cityflow.add_edge((road['points'][0]['x'], road['points'][0]['y']),(road['points'][-1]['x'],road['points'][-1]['y']),weight=cal_length(road))
    # pos_shapefile = {node: node for node in G_shapefile.nodes}
    # plt.figure(figsize=(10, 10))
    # nx.draw(G_shapefile, pos=pos_shapefile, node_size=10, edge_color='blue', node_color='blue', label="Shapefile")
    #
    # pos_cityflow = {node: node for node in G_cityflow.nodes}
    # nx.draw(G_cityflow, pos=pos_cityflow, node_size=10, edge_color='red', node_color='red', label="CityFlow")
    # # 添加图例
    # plt.legend(["Shapefile", "CityFlow"])
    # plt.title("Comparison of Shapefile and CityFlow Road Networks")
    # plt.show()

    # print("节点匹配度指标：",nx.algorithms.similarity.graph_edit_distance(G_shapefile, G_cityflow))
    # distance = nx.graph_edit_distance(G_shapefile, G_cityflow, timeout=10)
    # print("节点匹配度指标：",distance)

    # adj_matrix_shapefile = nx.adjacency_matrix(G_shapefile).todense()
    # adj_matrix_cityflow = nx.adjacency_matrix(G_cityflow).todense()

    # # 计算矩阵差异
    # matrix_difference = np.linalg.norm(adj_matrix_shapefile - adj_matrix_cityflow)
    # print(f"邻接矩阵差异：{matrix_difference}")

    # common_nodes = list(set(G_shapefile.nodes).intersection(set(G_cityflow.nodes)))
    # subgraph_shapefile = G_shapefile.subgraph(common_nodes)
    # subgraph_cityflow = G_cityflow.subgraph(common_nodes)
    # # 获取公共子图的邻接矩阵
    # adj_matrix_shapefile = nx.adjacency_matrix(subgraph_shapefile).todense()
    # adj_matrix_cityflow = nx.adjacency_matrix(subgraph_cityflow).todense()

    # # 计算矩阵差异
    # matrix_difference = np.linalg.norm(adj_matrix_shapefile - adj_matrix_cityflow)
    # print(f"邻接矩阵差异：{matrix_difference}")


