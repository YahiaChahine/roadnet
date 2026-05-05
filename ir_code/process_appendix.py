from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import json
import json, os, gif

def get_phase(phase_list, actions):
    agent_list = phase_list.keys()
    for agent in agent_list:
        phase_list[agent].append(actions[agent]) 
    return phase_list

def phase_analysis(phase_list,agent_list):
    def count_phase_duration(phase_list):
        result = {i : [] for i in range(8)}
        start_idx = 0
        for idx in range(1, len(phase_list)):
            if phase_list[idx] != phase_list[idx - 1]:
                result[phase_list[idx - 1]].append(f'{start_idx}-{idx - 1}')
                start_idx = idx
        result[phase_list[-1]].append(f'{start_idx}-{len(phase_list) - 1}')

        return result 
    
    phase_keep_time ={}
    phase_switch_time ={}
    for agent in agent_list:
        phase_keep_time[agent] = {i : [] for i in range(8)}
        phase_switch_time[agent] = {i : [] for i in range(8)}
    for agent in agent_list:
        phase_keep_time[agent] = count_phase_duration(phase_list[agent])

        # 统计持续时间不超过1的相位
        for phase in phase_keep_time[agent]:
            for index in range(len(phase_keep_time[agent][phase])):
                start, end = map(int, phase_keep_time[agent][phase][index].split('-'))
                if end - start < 1:
                    phase_switch_time[agent][phase].append(f'{start}-{end}')
    return phase_switch_time


def action_analysis(agent, env, duration, MFD_cfg=None, desc=None, path=None):
    dyna = {}
    selected_agents = ['intersection_2_1','intersection_2_2','intersection_2_3']
    phase_list = {agent_no: [] for agent_no in selected_agents}
    lanemove_list = {agent_no: [] for agent_no in selected_agents}
    lanemove_in_phase = {key:np.where(np.array(value)>0)[0] for key, value in env.env_config.phase_expansion.items()} 
    link_lane_map ={ 0:[0,6],1:[1,7],2:[3,9],3:[4,10],
                    4:[0,1], 5:[3,4],6:[6,7],7:[9,10],
            }
    queue_list = {agent_no: {i: [] for i in [0,3,6,9]} for agent_no in agent.agent_list}
    vehicles = []
    vehicles_info = {}
    ## 添加各智能体相位决策
    for i in trange(0, duration, agent.decision_interval, desc=desc):
        state = agent.observe(env,dyna=dyna)
        actions = agent.act(state)
        vehicles.extend(list(set(env.get_network_veh_id())-set(vehicles)))
        for veh_id in env.get_network_veh_id():
            vehicles_info[veh_id] = env.get_veh_info(veh_id)
        
        phase_list = get_phase(phase_list, actions)
        # for agent_no in agent.agent_list:
        #     for link in queue_list[agent_no]:
        #         queue_list[agent_no][link].append(state[agent_no]['inlane_queue'][link])    
        dyna = env.step(actions, agent.decision_interval)
        del state
    ## 分析各相位频率
    phase_freq = {agent_no: {} for agent_no in selected_agents}
    for agent_no in selected_agents:
        for i in range(8):
            phase_freq[agent_no][i] = phase_list[agent_no].count(i) / len(phase_list[agent_no])
    
    ## 分析相位是否突变
    phase_switch = phase_analysis(phase_list,selected_agents)
    agent_num = len(selected_agents)
    fig, axs = plt.subplots(agent_num, 1, figsize=(12, 8))
    for i, agent_no in enumerate(selected_agents):
        row = i // 3
        col = i % 3
        axs[i].plot(range(len(phase_list[agent_no])),phase_list[agent_no])
        axs[i].set_title(f'Intersection {agent_no}')
        axs[i].set_ylim(0,8)
        # 根据phase_switch_time作图，纵轴为key，横轴为value，横轴范围（0,420）
        for phase in phase_switch[agent_no]:
            for index in range(len(phase_switch[agent_no][phase])):
                start, end = map(int, phase_switch[agent_no][phase][index].split('-'))
                # 绘制散点图，散点小一点，这样似乎会持续增加，每个agent需更新
                axs[i].scatter(start, phase, color='r', s=2)
    plt.tight_layout()
    plt.show()
    plt.savefig('./phase.png')
    plt.close() 

    # link_list = {agent_no:{i:[] for i in range(8)} for agent_no in agent.agent_list}
    # # 分析各link的出现时差
    # for i, agent_no in enumerate(agent.agent_list):
    #     # 将phase_list[agent]中的值根据phase_action转换
    #     lanemove_list[agent_no] = [lanemove_in_phase[phase] for phase in phase_list[agent_no]]
    #     # 将phase_action_list[agent]中的值转换为对应的时差
    #     for idx, phase in enumerate(lanemove_list[agent_no]):
    #         for link in phase:
    #             link_list[agent_no][link].append(idx)
    # # 绘制排队长度图
    # for i, agent_no in enumerate(agent.agent_list):
    #     row = i // 3
    #     col = i % 3
    #     for link in queue_list[agent_no]:
    #         plt.plot(range(len(queue_list[agent_no][link])),queue_list[agent_no][link],label=f'link_{link}')
    #     plt.title(f'Intersection {agent_no}')
    #     plt.ylabel('queue length')
    #     plt.xlabel('decision step')
    #     plt.tight_layout()
    #     plt.legend()
    #     plt.savefig(f'./Intersection {agent_no}')
    #     plt.close() 

    # plt.plot(range(len(phase_list[agent.agent_list[0]])),phase_list[agent.agent_list[0]])
    # vehicles_route_length = {veh_id:sum(env.get_road_length(r) for r in vehicles_info[veh_id]['route'].split(' ')[:-1]) for veh_id in vehicles}
    # print(np.mean(list(vehicles_route_length.values())))
    return env.get_avg_travel_time()
    
def flow_analysis(agent, env, duration, MFD_cfg=None, desc=None, path=None):
    dyna = {}
    selected_agents = ['intersection_2_1','intersection_2_2','intersection_2_3']

    queue_list = {agent_no: {i: [] for i in [0,3,6,9]} for agent_no in agent.agent_list}
    ###     agent_no,time,dir,lane
    depart ={agent_no:{t:{dir:[0,0,0] for dir in range(4)}for t in range(0,duration,300)} for agent_no in selected_agents}
    for i in trange(0, duration, agent.decision_interval, desc=desc):
        state = agent.observe(env,dyna=dyna)
        actions = agent.act(state)
        dyna = env.step(actions, agent.decision_interval)
    #     env.get_network_veh_id()
    #     time = i //300*300
    #     for agent_no in agent.agent_list:
    #         if agent_no in selected_agents:
    #             for dir,road in enumerate(env.intersections[agent_no]['sort_roads'][:4]):
    #                 for lane_idx,lane in enumerate(env.roads[road]['lanes']):
    #                     depart[agent_no][time][dir][lane_idx]+=len(dyna['lane_in'][lane.split(":")[0]])   
    # ## depart转成csv
    # shp_gdf = pd.DataFrame(columns=['agent_no','time','dir','left','through','right','dir_total'])
    # for agent_no in depart:
    #     for time in depart[agent_no]:
    #         for dir in depart[agent_no][time]:
    #             shp_gdf = shp_gdf.append({'agent_no':agent_no,'time':time,'dir':dir,'left':depart[agent_no][time][dir][0],'through':depart[agent_no][time][dir][1],'right':depart[agent_no][time][dir][2],\
    #                 'dir_total':sum([int(depart[agent_no][time][dir][j]) for j in range(3)])},ignore_index=True)
    # shp_gdf.to_csv('./depart_1.csv')
    # shp_gdf = pd.read_csv('./depart_1.csv')
    # shp_gdf = shp_gdf[['agent_no','time','dir','left','through','right','dir_total']]


    # inter_1= shp_gdf[shp_gdf['agent_no']=='intersection_2_1']
    # inter_2= shp_gdf[shp_gdf['agent_no']=='intersection_2_2']
    # inter_3= shp_gdf[shp_gdf['agent_no']=='intersection_2_3']

    # colors = {'E':'r','W':'b','N':'g','S':'y'}
    # dir_map={0:'E',1:'N',2:'W',3:'S'}
    # ###  一个agent_no具有四个方向，使用线性图
    # for idx,group in inter_1.groupby('dir'):
    #     dir = dir_map[group['dir'].iloc[0]]
    #     # plt.plot(group['time'],group['left'],label=f'{dir}_left',color=colors[dir],linestyle='--')
    #     # plt.plot(group['time'],group['through'],label=f'{dir}_through',color=colors[dir])
    #     plt.plot(group['time'],group['left']/group['dir_total'],label=f'{dir}_left_ratio',color=colors[dir])
    #     plt.xticks(np.arange(0, duration, 600))
    #     plt.xlabel('time(s)')
    #     # plt.ylabel('flow(veh)')
    #     plt.ylabel('rate')
    #     plt.legend()
    #     plt.show()
    # plt.savefig('./inter1',dpi=500)
    # plt.close()


    # #### 干线流量及左转比例
    # inter = pd.DataFrame(columns=['agent_no','time','flow','left_ratio'])
    # for idx,group in shp_gdf.groupby(['agent_no','time']):
    #     inter = inter.append({'agent_no':idx[0],'time':idx[1],'flow':group['dir_total'].sum(),'left_ratio':group['left'].sum()/group['dir_total'].sum()},ignore_index=True)
    # inter.to_csv('./inter.csv')
    # for idx,group in inter.groupby('agent_no'):
    #     plt.plot(group['time'],group['flow'],label=idx)
    # plt.xticks(np.arange(0, duration, 600))
    # plt.xlabel('time(s)')
    # plt.ylabel('flow(veh)')
    # plt.legend()
    # plt.show()
    # plt.savefig('./flow.png',dpi=500)
    # plt.close()
    # for idx,group in inter.groupby('agent_no'):
    #     plt.plot(group['time'],group['left_ratio'],label=idx)
    # plt.xticks(np.arange(0, duration, 600))
    # plt.xlabel('time(s)')
    # plt.ylabel('left_ratio')
    # plt.legend()
    # plt.show()
    # plt.savefig('./left_ratio.png',dpi=500)

    # return 0
    return env.get_avg_travel_time()     

def demand_analysis(env,cfg_file,end_time =3600):
    demand_interval = 300
    demand = {t:0 for t in range(300,end_time+1,demand_interval)}
    road_demand = { t: {road:0 for road in env.roads} for t in range(300,end_time+1,demand_interval)}
    with open(cfg_file, 'r') as file:
        cfg = json.load(file)
    flow_file = cfg['flowFile']
    with open(flow_file, 'r') as file:
        flow_data = json.load(file)
    for flow in flow_data:
        route = flow['route']
        road = route[0]
        time = flow['startTime']
        t = time//demand_interval*demand_interval + demand_interval
        demand[t] +=1
        road_demand[t][road] +=1
    # 绘制路网需求
    time = list(demand.keys())
    plt.bar(time, list(demand.values()), width=-300,align='edge', color='skyblue', edgecolor='black')
    trend = np.poly1d(np.polyfit(time, list(demand.values()), 8))  # 三次多项式拟合
    plt.plot(np.array(time)-150,trend(time) , color='red')
    plt.xlabel('时间(s)')
    plt.ylabel('需求数')
    plt.xticks(np.arange(0, end_time+1, 300))
    plt.savefig('./demand.png')
    frames = []
    max_value = 0
    for t in range(300,end_time+1,demand_interval):
        max_value_t = max(road_demand[t].values())
        max_value = max(max_value_t, max_value)
    for t in range(300,end_time+1,demand_interval):
        frames.append(plot(env, t,road_demand[t], max_value=max_value))
    gif.save(frames, './roadnet_perform.gif', duration=3600 //60, unit="s", between="startend")
    return demand

def offeset_analysis(phase_list):
    phase_start={agent:[] for agent in phase_list.keys()}
    phase_end={agent:[] for agent in phase_list.keys()}
    for agent in phase_list.keys():
        for i in range(1,len(phase_list[agent])-1):
            if (phase_list[agent][i]==1)&(phase_list[agent][i-1]!=1):
                phase_start[agent].append(i)
        for i in range(0,len(phase_list[agent])-2):
            if (phase_list[agent][i]==1)&(phase_list[agent][i+1]!=1):
                phase_end[agent].append(i)
    offset = pd.DataFrame(columns=['agent','start','end'])
    for agent in phase_start:
        for i in range(len(phase_start[agent])):
            offset = offset.append({'agent':agent,'start':phase_start[agent][i],'end':phase_end[agent][i]},ignore_index=True)
    offset.to_csv('./offset.csv')
    return phase_start,phase_end