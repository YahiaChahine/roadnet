from datetime import datetime, timedelta
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cityflow_env_0 import CityFlowEnv
from agent import FTAgent, MPAgent, OAMAgent,GRUAgent # DQNAgent


def cal_co2(agent):
    b0=0.1569;b1=2.45*10**(-2);b2=-7.415*10**(-4);b3=5.975*10**(-5)
    c0=0.07224;c1=9.681*10**(-2);c2=1.075*10**(-3)
    co2 = 0
    co2_list =[]
    cruise_sum=0
    accel_sum=0
    for i in trange(0, duration, agent.decision_interval, desc=agent.name):
        actions = agent.act(agent.observe(env))
        veh_1=env.get_veh_speed()
        # step
        dyna = env.step(actions, agent.decision_interval)
        veh_2=env.get_veh_speed()
        co2_i=0
        exist_veh=(veh_1.keys())&(veh_2.keys())
        leave_veh=(veh_1.keys())-(veh_2.keys())
        for veh in leave_veh:
            v = veh_1.get(veh)
            f_cruise = b0 + b1 * v + b2 * (v ** 2) + b3 * (v ** 3)
            cruise_sum += f_cruise
            co2_i +=f_cruise
            co2+=f_cruise
        for veh in exist_veh:
            v=veh_1.get(veh)
            v2=veh_2.get(veh)
            u = (veh_2.get(veh)-veh_1.get(veh))/agent.decision_interval
            if u<0:
                f = (b0 + b1 * v + b2 * (v ** 2) + b3 * (v ** 3) + b0 + b1 * v2 + b2 * (v2 ** 2) + b3 * (v2 ** 3)) / 2
                co2_i += f
            elif u==0:
                f = b0 + b1 * v + b2 * (v ** 2) + b3 * (v ** 3)
                co2_i += f
            elif u>0:
                a_hat=-1/(2*1200)*0.32*1.184*2.5*v*v-0.015*9.8+u
                f_accel=a_hat*(c0+c1*v+c2*v*v)
                f_cruise = b0 + b1 * v + b2 * (v ** 2) + b3 * (v ** 3)
                cruise_sum += f_cruise
                accel_sum += f_accel
                f = f_cruise + f_accel
                co2_i += f
            co2 += f
        co2_list.append(co2_i*agent.decision_interval*10)
        co2_10=list(avg(co2_list))
    with open('./CO2.txt',mode='a')as file:   
        file.write(agent.name + '\n')
        file.write(str(co2_10)+ '\n')
    print('碳排放量1是:',sum(co2_10))
    print('碳排放量2是:',co2*agent.decision_interval)
    print('cruise排放量:',cruise_sum)
    print('accel排放量:', accel_sum)

    # with open('./CO2.txt',mode='r')as file:   
    #     travel_time_list = file.readlines()
    #     for idx in range(3):
    #         travel_time_list[idx] = [float(i.rstrip()) for i in travel_time_list[idx].split(',')]
    # print(travel_time_list)
  
  
def avg(data):
    LEN=10
    datasum = cnt = 0 
    for num in data:
        datasum += num
        cnt += 1
        if cnt == LEN: 
            yield datasum / LEN
            datasum = cnt = 0 
    if cnt: 
        yield datasum / cnt


if __name__ == '__main__':

    # ## env_settings
    cfg_file = './cfg/config_hangzhou_4x4.json'
    # cfg_file = './cfg/config_jinan_3_4_save.json'
    # cfg_file = './cfg/config_nanchang_warm_up_save.json'
    duration = 3600
    gen_flow_MFD = None # {'duration': duration, 'max_vol': 10, 'inc_time': 1200}
    MFD_cfg = {'x': 'net_accum', 'y': 'net_flow', 'width': 100}
    MFD_file_suffix = '_' + '_'.join(map(str, gen_flow_MFD.values())) if gen_flow_MFD else ''
    env = CityFlowEnv(cfg_file, simu_log=[MFD_cfg['x'], MFD_cfg['y']], gen_flow=gen_flow_MFD)


    ## get co2
    agents = []
    agents.append(FTAgent(env, 10))
    agents.append(MPAgent(env, 10))
    agent = GRUAgent(env, 10)
    agent.load_model(dir='./output/GRU_0.9_n_64/model', e=100)
    agents.append(agent)
    for agent_idx, agent in enumerate(agents):
        cal_co2(agent)
        env.reset()
    


