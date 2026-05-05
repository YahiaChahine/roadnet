from asyncore import read
from operator import le
import sys, os
from time import time
from multiprocessing import Process, cpu_count
from collections import defaultdict
from tracemalloc import start
from webbrowser import get
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import trange
import json


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))

# 添加项目根目录到 sys.path
if project_root not in sys.path:
    sys.path.append(project_root)
from cityflow_env import CityFlowEnv, datetime_utc8
# from process import get_veh_trajectory,random_prob,get_travel_time
from agent import MPAgent,WebsterAgent
from process import get_veh_trajectory,traj_plot,get_route_light


def hierarchical(sc ,pc, env, duration, desc=None, route=None):
    dyna = {}
    throughput = []
    times = []
    density = {k:[] for k in env.roads}
    phase_list = {agent_no: [] for agent_no in env.agents}
    light_list = {}
    
    for i in trange(0, duration, sc.decision_interval, desc=desc):
        state = sc.observe(env,dyna=dyna)
        actions = sc.act(state)
        cur_phase = actions['34180200000057']
        ##  lane-level评估 34180200000173
        if pc:
            state = pc.observe(env,dyna=dyna)
            actions.update(pc.act(state))
        if (i>=1600) and route:
            selected_route = route['flow_10777_0']
            veh_traj = env.get_veh_trajectory(selected_route)
            phase_list,light_list = get_route_light(env,phase_list,light_list,actions,selected_route)   
        if route:
            dyna  = env.step(actions, sc.decision_interval,route)
        else:
            dyna = env.step(actions, sc.decision_interval)
        ###  throughput
        lane_out = dyna.get('lane_out', {})
        agent_lane = env.get_inter_sort_lane('34180200000001')
        inlane= [lane for lane in agent_lane if lane and (env.lane_road_id(lane) in ['34180200024'])]
        throughput.append(2*sum([len(lane_out.get(l, set())) for l in inlane]))
        ###  density
        net_density = env.get_road_density()
        for k in density:
            density[k].append(net_density[k])
        del state
    throughput = [sum(throughput[i:i+6]) for i in range(0, len(throughput), 6)]
    times = [i * sc.decision_interval*6 for i in range(len(throughput))]
    throughput_data = {
        'throughput': throughput,
        'times': times
    }
    avg_density = {k: np.mean(v) for k, v in density.items()}
    ## sort
    avg_density = dict(sorted(avg_density.items(), key=lambda item: item[0]))


    if pc:
        with open('./utils/validate/throughput_data_pc.json', 'w') as f:
            json.dump(throughput_data, f)
    else:
        with open('./utils/validate/throughput_data_sc.json', 'w') as f:
            json.dump(throughput_data, f)
    if pc:
        with open('./utils/validate/density_pc.json', 'w') as f:
            json.dump(avg_density , f)
    else:
        with open('./utils/validate/density_sc.json', 'w') as f:
            json.dump(avg_density , f)

    if route:
        traj_plot(env,veh_traj,light_list,selected_route,1600,2000,sc.decision_interval,'./utils/validate/veh_traj.png') 
    return env.get_avg_travel_time()

def get_travel_time(agent, env, duration, MFD_cfg=None, desc=None, path=None):
    dyna = {}
    for i in trange(0, duration, agent.decision_interval, desc=desc):
        state = agent.observe(env,dyna=dyna)
        actions = agent.act(state)
        dyna = env.step(actions, agent.decision_interval)
        del state
    return env.get_avg_travel_time()


def plot_MFD( x, y, width=1, desc='MP', save=True, fig_dir='./utils/validate'):
    def cluster(arr, width):
        return np.reshape(arr[:len(arr) - len(arr) % width], (-1, width)).mean(axis=1)
    json_path_sc= "./utils/validate/SC_simu_log.json"
    json_path_pc = "./utils/validate/SC_PC_simu_log.json"
    
    simu_log_sc = json.load(open(json_path_sc, 'r', encoding='utf-8'))
    simu_log_pc = json.load(open(json_path_pc, 'r', encoding='utf-8'))

    

    # plt.scatter(cluster(simu_log_pc[x][600:], width), cluster(simu_log_sc[y][600:], width), label='Signal Control + Perimeter Control', alpha=0.5)
    # plt.scatter(cluster(simu_log_sc[x][600:], width), cluster(simu_log_sc[y][600:], width), label= 'Signal Control ', alpha=0.5)
    plt.scatter(cluster(simu_log_pc[x], width), cluster(simu_log_sc[y], width), label='Signal Control + Perimeter Control', alpha=0.5)
    plt.scatter(cluster(simu_log_sc[x], width), cluster(simu_log_sc[y], width), label= 'Signal Control ', alpha=0.5)
    if save:
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.xlabel('Accumulations (veh)')
        plt.ylabel('Trip completion rate (veh/s)')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(f'{fig_dir}/MFD_compare_{width}.png',dpi = 500)
        plt.close()

def plot_throughput(x, y, width=1, desc='MP', save=True, fig_dir='./utils/validate', log_dir='./utils/validate'):
    throughput_pc = json.load(open(f'{log_dir}/throughput_data_pc.json', 'r', encoding='utf-8'))
    throughput_sc = json.load(open(f'{log_dir}/throughput_data_sc.json', 'r', encoding='utf-8'))
    times = throughput_sc[x]
    throughput_sc = throughput_sc[y]
    throughput_pc = throughput_pc[y]
    plt.figure(figsize=(10, 6))
    plt.plot(times, throughput_sc, label='Without Perimeter Control', color='#FFB347')
    plt.plot(times, throughput_pc, label='With Perimeter Control', color='#4169E1')
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Throughput (veh)', fontsize=14)
    plt.title(f'Throughput over Time', fontsize=16)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.savefig(f'{fig_dir}/throughput.png', dpi=500, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    """configration"""
    config_file = "./cfg/xuancheng/config_xuancheng_test_save.json"
    env = CityFlowEnv(config_file,simu_log=['net_accum', 'trip_completion'])
    agent_list = env.agents
    signal_control = MPAgent(env,10)
    perimeter_control = WebsterAgent(env,10,None)

    duration = 3600
    route = {'flow_10777_0':[ "34180200081","34180200078","34180200219",'34180200024_1','34180200024','34180200020','34180201883','34180201879','34180203795','34180203793','34180200209','34180200084']}
    # route = {'flow_10777_0':[ '34180200081', '34180200078', '34180200219', '34180200024_1', '34180200024', '34180200019', '34180200182', '34180200180','34180200084']}
    """Signal-only"""
    # pc = None
    # travel_time= hierarchical(signal_control, pc,env, duration, desc="Running simulation")
    # env.plot_MFD(x='net_accum', y='trip_completion', width=10, desc=f'SC', save=True, fig_dir=current_dir, log_dir=current_dir)
    """Signal + Vehicle Control"""
    # travel_time =hierarchical(signal_control, None,env, duration, desc="Running simulation", route=route)
    """Signal + Perimeter Control"""
    # pc = perimeter_control
    # travel_time= hierarchical(signal_control, perimeter_control,env, duration, desc="Running simulation")
    # env.plot_MFD(x='net_accum', y='trip_completion', width=10, desc=f'SC_PC', save=True, fig_dir=current_dir, log_dir=current_dir)
    # print(f"travel time: {travel_time}")
    plot_MFD(x='net_accum', y='trip_completion', width=30 )
    plot_throughput(x='times', y='throughput', width=6, desc=f'Maxpressure', save=True, fig_dir=current_dir, log_dir=current_dir)



