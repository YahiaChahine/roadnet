from asyncore import read
from operator import le
import sys, os
import json
from time import time
import copy
from multiprocessing import Process, cpu_count
from collections import defaultdict
import ast
from tracemalloc import start
from webbrowser import get
import numpy as np
import matplotlib.pyplot as plt
import inspect
import shutil
import orjson 
import pandas as pd


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))

# 添加项目根目录到 sys.path
if project_root not in sys.path:
    sys.path.append(project_root)
from cityflow_env import CityFlowEnv, datetime_utc8
from process import get_veh_trajectory,random_prob


def data_statistics(flow_file):
    with open(flow_file, "r", encoding="utf-8") as f:
        data = json.load(f)  # 解析 JSON 看看是否报错
        origins = defaultdict(int)
        passby =  defaultdict(int)
        for trip in data:
            origin = trip['route'][0]
            origins[origin] += 1
            if len(trip['route']) > 1:
                for r in trip['route'][1:]:
                    passby[r] += 1
        print(origins)
        # Sort by value in descending order
        sorted_origins = dict(sorted(origins.items(), key=lambda x: x[1]))
        sorted_passby = dict(sorted(passby.items(), key=lambda x: x[1]))
        common_roads = set(sorted_origins.keys()) & set(sorted_passby.keys())
        differences = {}
        for road in common_roads:
            diff = abs(sorted_origins[road] + sorted_passby[road])
            differences[road] = (diff, sorted_origins[road], sorted_passby[road])
        sorted_differences = dict(sorted(differences.items(), key=lambda x: x[1][0]))
    return sorted_differences

def filter_deduplicate(flow_file, output_json):
    with open(flow_file, "r", encoding="utf-8") as f:
        data = json.load(f)  # 解析 JSON 看看是否报错
        ## 选择route包含’34180201769‘的trip
        validated_data = []
        for trip in data:
            if '34180202844' in trip['route']:
                seen = set()
                filtered_route = []
                for r in trip['route']:
                    if r in roads and r not in seen:
                        filtered_route.append(r)
                        seen.add(r)
                    else:
                        route_num = len(trip['route'])
                        duplicate_index = trip['route'].index(r)
                        print(f"Invalid trip: {data.index(trip)} valid:{duplicate_index }/{route_num }")
                        break
                trip['route'] = filtered_route
                if  '34180202844' in trip['route']:
                    validated_data.append(trip)
    with open(output_json, "w", encoding="utf-8") as f:
        f.write(orjson.dumps(validated_data).decode("utf-8"))

def filter_route_and_time(route, time):
    """同步route和time"""
    filtered_route = []
    filtered_time = []
    seen = set()  # 用于记录已经处理过的 road
    for r, t in zip(route, time):
        if r not in seen:  
            filtered_route.append(r)
            filtered_time.append(t)
            seen.add(r)
    return filtered_route, filtered_time if filtered_route else ([], [])


def sort_route_and_time(route, time):
    combined = list(zip(route, time))
    combined.sort(key=lambda x: x[1][0])
    # 解压缩回 route 和 time
    sorted_route, sorted_time = zip(*combined)
    return list(sorted_route), list(sorted_time)

def start_with_road(intermediate_json,valideted_csv,validated_json,road = '34180202844'):
    with open(intermediate_json, "r", encoding="utf-8") as f:
        validated_data = json.load(f)
    df = pd.read_csv(valideted_csv)
    df.loc[:, 'route'] = df['route'].apply(ast.literal_eval)
    df.loc[:,'time'] = df['time'].apply(ast.literal_eval)

    filtered_values = df.apply(lambda row: filter_route_and_time(row['route'], row['time']), axis=1, result_type='expand')
    df.loc[:,'route'] = filtered_values[0]
    df.loc[:,'time'] = filtered_values[1]

    # 对每行数据内的time进行排序
    result = df.apply(lambda row: sort_route_and_time(row['route'], row['time']), axis=1, result_type='expand')
    df.loc[:, 'route'] = result[0]
    df.loc[:, 'time'] = result[1]

    df['sort_key'] = df['time'].apply(lambda x: x[0][0] if x and len(x) > 0 and len(x[0]) > 0 else 0).copy()
    df = df.sort_values(by='sort_key', ascending=True)


    ###   df和validated_data是可以对应的
    map_cityflow_csv = {}
    validated_data1 = []
    j = 0
    for i in range(len(df)):
        starttime = df['time'][i][0][0]

        for trip in validated_data:
            if trip['startTime'] == starttime and trip['route'][0] == str(df['route'][i][0]) and starttime<36000:
                start_index = trip['route'].index(road)
                start_index1 = df['route'][i].index(int(road))
                if start_index != start_index1:
                    continue
                trip_copy = copy.deepcopy(trip)
                trip_copy['route'] = trip_copy['route'][start_index:]
                num = 6
                if len(trip_copy['route'])>=num:
                    trip_copy['route'] = trip_copy['route'][0:num]
                    trip_copy['startTime'] = int(df['time'][i][start_index][0])
                    trip_copy['endTime'] = int(df['time'][i][start_index][0])
                    df.at[i, 'time'] = list(df.at[i, 'time'])[start_index:start_index+num]
                    df.at[i, 'route'] = list(df.at[i, 'route'])[start_index:start_index+num]
                    validated_data1.append(trip_copy)
                    map_cityflow_csv[j] = i 
                    j+=1
                    break
                else:
                    break
    validated_data1 = sorted(validated_data1, key=lambda x: x['startTime'])
    with open(validated_json, "w", encoding="utf-8") as f:
        f.write(orjson.dumps(validated_data1).decode("utf-8"))
    map_json = './map_cityflow_csv.json'
    with open(map_json, "w", encoding="utf-8") as f:
        f.write(json.dumps(map_cityflow_csv))
    df.to_csv('./validated.csv', index=False)


def get_traveltime(cfg_file,travel_time_json):

    # gen_flow= {'duration': 3600, 'volume': 11000, 'prob': random_prob(3600)}
    # env = CityFlowEnv(cfg_file3,gen_flow = gen_flow)
    env = CityFlowEnv(cfg_file)
    env.env_phase_map("34180200005443")
    duration = int(3600)
    env.reset()
    from agent import FTAgent,WebsterAgent
    with open(cfg_file, 'r') as file:
        config = json.load(file)
    flow_file = config['flowFile']
    agent = WebsterAgent(env,2,flow_file,3)
    travel_time,veh_tra = get_veh_trajectory(agent, env, duration, agent.name+str(),['34180202844','34180202935','34180202909'],2,path = current_dir+'/xuancheng_0403_veh_tra.png')
    df = pd.read_csv('./validated.csv')
    df.loc[:,'time'] = df['time'].apply(ast.literal_eval)
    map_json = './map_cityflow_csv.json'
    with open(map_json, "r", encoding="utf-8") as f:
        map_cityflow_csv = json.load(f)
    vehicle_time = {}
    vehicle_diff = {}
    for v in veh_tra:
        id = v.split('_')[1]
        index = map_cityflow_csv[id]
        time = df['time'][index]
        travel_time = time[1][1] - time[0][0]
        simu_travel_time = veh_tra[v]['t'][-1] - veh_tra[v]['t'][0]
        vehicle_time[v] = [travel_time, simu_travel_time]
        vehicle_diff[v] = abs(simu_travel_time - travel_time)

    with open(travel_time_json, "w", encoding="utf-8") as f:
        f.write(orjson.dumps(vehicle_time).decode("utf-8"))

def read_traveltime(travel_time_json):
    with open(travel_time_json, 'r') as file:
        vehicle_time  = json.load(file)
    print(vehicle_time)
    diff_sum = []
    for _,v in vehicle_time.items():
        real,sim = v
        if real <150 and sim >40:
            diff_sum.append(abs(real -sim))
    print(sum(diff_sum)/len(diff_sum))
    return vehicle_time

    

def data_plot(travel_time_json):
    with open(travel_time_json, "r", encoding="utf-8") as f:
        vehicle_time = json.load(f)

        # Filter out entries where real travel time > 180 or simulated travel time < 40
        vehicle_time_filtered = {v: vehicle_time[v] for v in vehicle_time 
                                if 44<=vehicle_time[v][0] <= 150 and vehicle_time[v][1] >= 40}
        valid_vehicle = {v: vehicle_time[v] for v in vehicle_time_filtered
                                if abs(vehicle_time[v][0]-vehicle_time[v][1])<50}
        print('真实平均值', sum([v[0] for v in vehicle_time_filtered.values() ])/len(vehicle_time_filtered))  
        print('均值', sum([abs(v[0]-v[1]) for v in vehicle_time_filtered.values() ])/len(vehicle_time_filtered))
        print('有效比例',len(valid_vehicle)/len(vehicle_time_filtered))
        print("有效均值", sum([abs(v[0]-v[1]) for v in valid_vehicle.values() ])/len(valid_vehicle))
        vehicle_time = valid_vehicle
    modes = list(vehicle_time.keys())
    real = [vehicle_time[mode][0] for mode in modes]
    simu = [vehicle_time[mode][1] for mode in modes]

    y = np.arange(len(modes))  # Y轴位置


    fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.6

    ax.barh(y, real, height=bar_width, color='darkseagreen', label='real')
    ax.barh(y, [-val for val in simu], height=bar_width, color='forestgreen', label='simulation')

    total_modes = len(modes)
    tick_step = 10  # 每隔10个显示一次
    tick_positions = np.arange(0, total_modes+10, tick_step)  # 从0开始，每隔10个取一个位置

    ax.set_yticks(tick_positions)
    ax.set_yticklabels([str(i) for i in tick_positions])

    xticks = np.arange(-150,151, 50)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(abs(x)) for x in xticks])

    ax.set_xlabel('Travel Time')
    ax.set_ylabel('Vehicle ID')
    ax.set_title('Comparison of Travel Time ')
    ax.legend()
    ax.axvline(0, color='gray', linewidth=1)

    plt.tight_layout()
    plt.savefig(current_dir+'/travel_time_comparison.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    """configration"""
    cfg_file1 = './cfg/xuancheng/config_xuancheng_validate.json'
    cfg_file2 = './cfg/xuancheng/config_xuancheng_type_validate.json'

    

    flow_file = current_dir+"/data_2023_04_03_filtered.json"
    intermediate_json = current_dir+'/xuancheng_0403_intermediate.json'
    validated_json = current_dir+'/xuancheng_0403_validated.json'
    valideted_csv = current_dir+'/data_2023_04_03_validated.csv'
    travel_time_json = current_dir+'/vehicle_travel_time.json'

    flow_type_file = current_dir+"/data_2023_04_03_type_filtered.json"
    intermediate_type_json = current_dir+'/xuancheng_0403_type_intermediate.json'
    validated_type_json = current_dir+'/xuancheng_0403_type_validated.json'
    valideted_type_csv = current_dir+'/data_2023_04_03_type_validated.csv'
    travel_time_type_json = current_dir+'/vehicle_type_travel_time.json'

    env = CityFlowEnv(cfg_file1)
    roads = env.roads

    # data_statistics(flow_file)

    # filter_deduplicate(flow_file, intermediate_json)
    # start_with_road(intermediate_json,valideted_csv,validated_json,road = '34180202844')
    # get_traveltime(cfg_file1,travel_time_json)
    # res = read_traveltime(travel_time_json)

    # filter_deduplicate(flow_type_file, intermediate_type_json)
    # start_with_road(intermediate_type_json,valideted_type_csv,validated_type_json,road = '34180202844')
    # get_traveltime(cfg_file2,travel_time_type_json)
    # type_res =read_traveltime(travel_time_type_json)
    #比较两个字典
    # diff = {}
    # for key in res:
    #     if key in type_res:
    #         if int(abs(res[key][0] - type_res[key][0]))== 0 and int(abs(res[key][1] - type_res[key][1]))==0:
    #             continue
    #         else:
    #             diff[key] = [abs(res[key][0] - type_res[key][0]), abs(res[key][1] - type_res[key][1])]
    # print(diff)
    data_plot(travel_time_json)






