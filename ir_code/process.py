from calendar import c
from cmath import phase
from datetime import datetime, timedelta
import re
from tkinter import W, font
from turtle import color
from tqdm import trange
import numpy as npv
import matplotlib.pyplot as plt
from cityflow_env_0 import CityFlowEnv,datetime_utc8
from multiprocessing import Process, cpu_count
from time import sleep
import json, os, gif
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline,interp1d
from process_appendix import flow_analysis,demand_analysis,action_analysis, phase_analysis, offeset_analysis
gif.options.matplotlib["dpi"] = 350

@gif.frame
def plot(env, t,value, max_value=1):
    x_coord = [i['coord'][0] for i in env.intersections.values()]
    y_coord = [i['coord'][1] for i in env.intersections.values()]
    fig_x = (max(x_coord) - min(x_coord)) / 1000
    fig_y = (max(y_coord) - min(y_coord)) / 1000
    
    plt.figure(figsize=(fig_x * 1.4, fig_y * 1.2))
    plt.rcParams['font.size'] = fig_y * 3
    
    cmap_color = plt.cm.get_cmap('RdYlGn_r')
    for road_id, road_attr in env.roads.items():
        coord = np.array(road_attr['coord'])
        color_index = cmap_color(value[road_id] / max_value)

        for (x1, y1), (x2, y2) in zip(coord[:-1], coord[1:]):
            dx = y2 - y1
            dy = x1 - x2
            ratio = 15 / np.sqrt(dx**2 + dy**2)
            dx *= ratio
            dy *= ratio
            plt.plot([x1 + dx, x2 + dx], [y1 + dy, y2 + dy], c=color_index, alpha=0.99, linewidth=1.6)

    plt.title(f't = {t}')
    plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(0, max_value), cmap_color))
    plt.gca().axis('off')
    plt.savefig(f'./roadnet_{t}.png')


def get_travel_time(agent, env, duration, MFD_cfg=None, desc=None, path=None):
    dyna = {}
    for i in trange(0, duration, agent.decision_interval, desc=desc):
        state = agent.observe(env,dyna=dyna)
        actions = agent.act(state)
        dyna = env.step(actions, agent.decision_interval)
        del state
    return env.get_avg_travel_time()

def get_veh_trajectory(agent,env,duration,desc=None,route=None,interval=10,path='./traj_4.png'):
    dyna = {}
    selected_agents = env.agents
    phase_list = {agent_no: [] for agent_no in selected_agents}
    light_list = {}
    queue =0; delay = 0
    for i in trange(0, duration, interval, desc=desc):
        state = agent.observe(env,dyna=dyna)

        start =0;end = duration
        if (i>=start)&(i<=end):
            # route = ['road_2_0_1','road_2_1_1','road_2_2_1','road_2_3_2']
            # route = ['road_2_4_3','road_2_3_3','road_2_2_3','road_2_1_0']
            
            veh_traj = env.get_veh_trajectory(route)
            t = env.eng.get_current_time()
        if i%agent.decision_interval==0:
            actions = agent.act(state)
            phase_list,light_list = get_route_light(phase_list,light_list,actions,route)
            dyna = env.step(actions, agent.decision_interval)
        for r in route:
            if r == '34180202935':
                l =  env.roads[r]['lanes']
                for lane in l:
                    queue += dyna['lane_queue'][lane]
                    delay += dyna['lane_delay'][lane]
        del state
    queue = queue / duration/len(env.roads['34180202935']['lanes'])
    delay = delay / duration/len(env.roads['34180202935']['lanes'])
    travel_time = sum([veh_traj[veh_id]['t'][-1] - veh_traj[veh_id]['t'][0] for veh_id in veh_traj])/ len(veh_traj) if veh_traj else 0
    traj_plot(env,veh_traj,light_list,route,start,end,agent.decision_interval,path) 
    offeset_analysis(light_list)
    return env.get_avg_travel_time(),veh_traj

def get_pahse_for_route(env,route1,route2):
    dir_phase_map = {
        ('0', '0'): [1, 6],
        ('0', '1'): [0, 6],
        ('1', '1'): [3, 7],
        ('1', '2'): [2, 7],
        ('2', '2'): [1, 4],
        ('2', '3'): [0, 4],
        ('3', '3'): [3, 5],
        ('3', '0'): [2, 5]
    }
    dir_xy_map = {
        '0': (1, 0),
        '1': (0, 1),
        '2': (-1, 0),
        '3': (0, -1)
    }
    try:
        x, y, dir1 = route1.split('_')[1:]
        dir2 = route2.split('_')[3]
        dx, dy = dir_xy_map[dir1]
        x,y = int(x),int(y)
        x += dx
        y += dy
        phase = dir_phase_map.get((dir1, dir2))
        agent = f'intersection_{x}_{y}'
    except:
        if env.roads[route1]['end_inter']== env.roads[route2]['start_inter']:
            agent = env.roads[route1]['end_inter']
            if agent not in env.agents: 
                phase = []
            else:
                dir1 = (env.intersections[agent]['sort_roads'].index([route1])+2)%4
                dir2 = env.intersections[agent]['sort_roads'].index([route2])-4
                phase = dir_phase_map.get((str(dir1), str(dir2)))
                if phase is None:
                    phase = []
        else:
            agent = ''
            phase = []
    return agent, phase

def get_agent_for_route(env,route):
    dir_xy_map = {
        '0': (1, 0),
        '1': (0, 1),
        '2': (-1, 0),
        '3': (0, -1)
    }
    try:
        x, y, dir1 = route.split('_')[1:4]
        dx, dy = dir_xy_map[dir1]
        x,y = int(x),int(y)
        x += dx
        y += dy
        agent = f'intersection_{x}_{y}'
    except:
        agent = env.roads[route]['end_inter']
    return agent

def get_route_light(env,phase_list,light_list,actions,route):
    agent_list = actions.keys()
    for agent in agent_list:
        phase_list[agent].append(actions[agent]) 

    for i in range(len(route)-1):
        agent,phase = get_pahse_for_route(env,route[i],route[i+1])
        if agent in actions and len(phase)>0:
            if agent not in light_list:
                light_list[agent]=[]
            if actions[agent] in phase:
                light_list[agent].append('forestgreen')
            else:
                light_list[agent].append('crimson')
    return phase_list,light_list

def get_agent_disatnce(env,agent,route):
    agent_distance = {}
    distance = 0
    for i in range(len(route)-1):
        distance+=env.get_road_length(route[i])
        agent = get_agent_for_route(env,route[i])
        if agent not in agent_distance:
            agent_distance[agent] = distance
    return agent_distance

def get_route_distance(env,route):
    distance = 0
    for i in range(len(route)):
        distance += env.get_road_length(route[i])
    return distance

def get_direction_phase(phase_list,actions):
    agent_list = actions.keys()
    for agent in agent_list:
        phase_list[agent].append(actions[agent]) 
    direction_light={'0':{agent:[] for agent in agent_list },'2':{agent:[] for agent in agent_list}}
    for direction in direction_light:
        for agent in agent_list:
            for idx,phase in enumerate(phase_list[agent]):
                if (phase in [1,6]) and (direction=='0'):
                    direction_light[direction][agent].append('forestgreen')
                elif (phase in [4])&(direction=='2'):
                    direction_light[direction][agent].append('limegreen')
                else:
                    direction_light[direction][agent].append('crimson')
    return phase_list,direction_light

def smooth(x,y):
    x_smooth = np.linspace(min(x), max(x), int((max(x)-min(x))/5))
    y_smooth = interp1d(x, y,kind='linear')(x_smooth)
    return [x_smooth, y_smooth]

def traj_plot(env,veh_traj,light_list,route,start,end,interval,figname='./traj_1.png'):
    plt.figure()
    # 遍历每辆车的轨迹数据
    if interval == 10:
        t = np.arange(len(light_list[list(light_list.keys())[0]])*10,step=10)+start
        # [start,end]   
        t = np.arange(start, end, step=interval) if start < end else np.arange(end, start, step=interval)
    else:
        t = np.arange(len(light_list[list(light_list.keys())[0]])*interval,step =interval )+start
    veh_traj={k:v for k, v in veh_traj.items() if len(v['route']) >= 2}

    for veh_id, veh_traj_info in veh_traj.items():
        if (len(veh_traj_info['route'])>=2):
            color='darkgrey';label='->'.join(route)
            x,y = smooth(veh_traj_info['t'],veh_traj_info['distance'])
            if veh_id=='flow_10777_0':
                plt.plot(x,y,'burlywood',label="Target Vehicle")
            else:
                plt.plot(x,y,color)
        else:
            continue
    plt.grid(ls = "--", lw = 0.5, color = "#4E616C")
    ### 绘制红绿灯
    for agent in light_list:
        distance = get_agent_disatnce(env,agent,route)
        ###   ditance是每个智能体所在位置
        for i in range(len(t) - 1):
            plt.fill_between([t[i], t[i+1]],distance[agent]-10,distance[agent]+10, color=light_list[agent][i])
    ###  终点水平线
    end_distance = get_route_distance(env,route)
    plt.axhline(y=end_distance, color='gray', linestyle='--', label='Destination')
    plt.legend(loc='upper right', fontsize=8)
    plt.xlabel("Time(/s)")
    plt.ylabel("Distance(/m)")
    plt.yticks(np.arange(0, 3100,500))
    plt.savefig(figname,dpi=500)


def get_travel_time_list(result_dir):
    if not os.path.exists(result_dir):
        print('result_dir doesn\'t exist')
        return
    if (not os.path.exists(result_dir+'/travel_time.txt'))&(not os.path.exists(result_dir+'/travel_time.npy')):
        travel_time_list=[]
        print('travel_time doesn\'t exist')
    elif os.path.exists(result_dir+'/travel_time.txt'):
        with open(result_dir+'/travel_time.txt',mode='r')as file:   
            travel_time_list=file.readlines()
        travel_time_list=[float(i.rstrip()) for i in  travel_time_list]
    elif os.path.exists(result_dir+'/travel_time.npy'):
        travel_time_list=np.load(result_dir+'/travel_time.npy', allow_pickle=True).tolist() 
    return travel_time_list



def get_result_dir(output_dir,gen_flow=None,cfg_file=None,change=True):
    if (gen_flow!=None)&(change):
        volume = gen_flow['volume']
        roadenet = ((cfg_file.split('/')[-1]).split('_')[-1]).split('.')[0]
        result_dir = output_dir +f'/result_{roadenet}_{volume}'
    elif (gen_flow!=None)&(not change):
        result_dir = '/'.join(cfg_file.split('/')[:-1])
    return result_dir

def run_episodes(output_dir,cfg_file,gen_flow=None,duration=3600,result_dir=None):
    model_dir= output_dir+'/model'
    if  not os.path.exists(result_dir):
        os.mkdir(result_dir)
    log_dir = result_dir + "/log"
    if  not os.path.exists(log_dir):
        os.mkdir(log_dir)

    ### env
    # duration = gen_flow['duration'] if gen_flow else 7200
    gen_flow_MFD = gen_flow
    MFD_cfg = {'x': 'net_accum', 'y': 'net_flow', 'width': 100}
    MFD_file_suffix = '_' + '_'.join(map(str, gen_flow_MFD.values())) if gen_flow_MFD else ''
    

    ## get time_list
    travel_time_list=get_travel_time_list(result_dir=result_dir)
    print(f'model_dir:{model_dir}')
    for e in range(max(len(travel_time_list),1), 100):
    # for e in range(90,101):
        simulator_cfg_file = cfg_file
        simulator_cfg_file = simulator_cfg_file.split('/')[-1]
        simulator_cfg_file = simulator_cfg_file.replace('.json', '_new.json')
        simulator_cfg_file = '/'.join((log_dir, simulator_cfg_file))
        with open(cfg_file, 'r') as file:
            cfg = json.load(file)
        if "replayLogFile" in cfg.keys():
            if e % 1 == 0:
                cfg["replayLogFile"] =  result_dir+'/' +cfg["replayLogFile"].split('.')[0]+'_'+str(e)+'.log'
                cfg["roadnetLogFile"] = result_dir +'/' +cfg["roadnetLogFile"]
            else:
                del cfg["replayLogFile"]
                del cfg["roadnetLogFile"]
                cfg['saveReplay'] = False
        with open(simulator_cfg_file, 'w') as file:
            json.dump(cfg, file,indent=4)
        
        env = CityFlowEnv(simulator_cfg_file, simu_log=[MFD_cfg['x'], MFD_cfg['y']], gen_flow=gen_flow_MFD)
        demand_analysis(env,cfg_file)
        
        #########       agent  ###############
        # agent = MPLight(env, 10)
        # agent = CDQAgent(env,10)
        # agent = AttendLight(env, 10)
        # agent = FRAP(env, 10)
        # agent = OAMAgent(env,10)
        agent = OAMOAgent(env,10)
        # agent = WebsterAgent(env,1,cfg['flowFile'])
        # agent = FTAgent(env,10)
        # agent = OAMAgentPlus(env,10)
        # agent = MPAgent(env, 10)



        # ## 检查agent是否一致
        if  (os.path.exists(model_dir))&(agent.name in ['CDQ','OAM','OAMPlus','MPLight','AttendLight','FRAP']):
            agent_name= os.listdir(model_dir)[0].split('_')[0]
            if agent_name in agent.name:
                pass
            elif (agent_name=='OAMO')&(agent.name=='CDQ'):
                agent.name=='OAMO'
            else:
                print("model_dir conflict with agent ")
                return
            try:
                agent.load_model(dir=model_dir, e=e)
            except OSError:
                print(f"Model file {e} not exists. Retry...", end='\r')
                break
        else:
            print("model_dir doesn't exist")
        env.reset()
        travel_time=get_travel_time(agent, env, duration, MFD_cfg, agent.name+str())
        # travel_time = action_analysis(agent, env, duration, MFD_cfg, agent.name+str())
        # travel_time = flow_analysis(agent, env, duration, MFD_cfg, agent.name+str())
        # travel_time,_ = get_veh_trajectory(agent, env, duration, agent.name+str())
        travel_time_list.append(travel_time)
        print(f'{e}_travel_time:',travel_time)
        with open(result_dir+'/travel_time.txt',mode='a')as file:   
            file.write(str(travel_time)+'\n')    
        del env
        plt.plot(range(1,len(travel_time_list)+1),travel_time_list)
        plt.savefig(f'{result_dir}/travel_time.png')
        plt.close()
    plt.plot(range(1,len(travel_time_list)+1),travel_time_list,label='agent')
    plt.ylabel('travel_time(s)')
    plt.xlabel('episodes')
    plt.legend() # show the label
    plt.savefig(f'{result_dir}/travel_time.png')
    plt.close()
    
def generate_flow_file(output_dir,gen_flow,cfg_file):
    result_dir=get_result_dir(output_dir,gen_flow,cfg_file)
    if  not os.path.exists(result_dir):
        os.mkdir(result_dir)
    new_cfg_file = cfg_file.split('/')[-1]
    new_cfg_file = new_cfg_file.replace('.json', f'_random.json')
    new_cfg_file = '/'.join((result_dir, new_cfg_file))
    with open(cfg_file, 'r') as file:
        cfg = json.load(file)

    cfg['flowFile'] = cfg['roadnetFile'].replace('.json', f'_flow_random.json')
    cfg['flowFile'] = cfg['flowFile'].split('/')[-1]
    cfg['flowFile'] = '/'.join((result_dir, cfg['flowFile']))

    with open(new_cfg_file, 'w') as file:
        json.dump(cfg, file,indent=4)

    return new_cfg_file

def compare_baseline(agents_output,result='/result',env_name ='hangzhou',png_name='./compare.png'):
    # colors=[i for i in 'bgrcmykw' ]
    colors=['firebrick','steelblue','limegreen'] 
    colors=['mediumpurple','goldenrod']
    colors=['mediumpurple','firebrick','dimgray','limegreen'] 
    len_=100
    agents_travel_time_lists ={}
    for agent in agents_output:
        travel_time_lists=[]
        for idx,output_dir in enumerate(agents_output[agent]):
            result_dir= output_dir + result
            travel_time_list = get_travel_time_list(result_dir=result_dir)
            travel_time_list = np.ravel(travel_time_list)
            len_ = len(travel_time_list) if len_ > len(travel_time_list) else len_
    for agent in agents_output:
        travel_time_lists=[]
        for idx,output_dir in enumerate(agents_output[agent]):
            result_dir= output_dir + result
            try:
                travel_time_list = get_travel_time_list(result_dir=result_dir)[:len_]
            except:
                print(f'{result_dir} doesn\'t exist')
            travel_time_list = np.ravel(travel_time_list)
            travel_time_lists.append(travel_time_list)
        agents_travel_time_lists[agent] = travel_time_lists

    ## plot
    time = np.array([i for i in range(1, len_+1)])
    Algo = agents_output.keys()
    all = pd.DataFrame([])
    for seed in range(3):  ## 一个算法test的次数
        for algo in Algo:
            data = pd.DataFrame(np.ones((len(time), 4)))
            data.columns = [
                'Training episodes', 'agent', 'Average travel time (s)', 'seed'
            ]
            data['Training episodes'] = time
            data['agent'] = algo.replace('_', ' ')

            ### 只改这行 放入相应的reward
            data['Average travel time (s)'] = agents_travel_time_lists[algo][seed]

            data['seed'] = seed
            all = pd.concat([all, data], 0, ignore_index=True)    
    plt.figure(figsize=[15, 10])
    # 在某张图上继续画线
    sns.set(style="white", font_scale=2.5)
    fig = sns.lineplot(x='Training episodes',
                    y='Average travel time (s)',
                    data=all,
                    hue='agent',
                    # style='agent',
                    palette = colors,
                    linewidth=2.5)
    # plt.title(env_name+'_'+result_dir.split('/')[-1])
    # plt.axhline(360,color="limegreen",linewidth=2.5,label ='OAMO',linestyle='-.')
    plt.show()
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.ylim(320,900)
    plt.xlim(-1,101)
    plt.savefig('./'+env_name+'_'+result_dir.split('/')[-1]+'.png',dpi=500)


def read_model_weight(model_dir='./output/2022-10-13_10-25-35'+'/model',e=500):
    result_dir = f"./output/{model_dir.split('/')[2]}"+'/result'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    import h5py
    f=h5py.File(model_dir+f"/MS_{e}.h5","r+")
    for key in f.keys():
        for key_next in f[key]:
            for key_next2 in f[key][key_next]:
                with open(result_dir+'/weight.txt',mode='a')as file: 
                    print('3级:',f[key][key_next][key_next2].name)
                    file.write('3级:'+f[key][key_next][key_next2].name+'\n')
                    print(f[key][key_next][key_next2][()])
                    file.write(str(f[key][key_next][key_next2][()].tolist())+'\n')
            print('\n')

def linear_plat(tot_time, inc_time):
    prob = list(range(inc_time))
    prob += [prob[-1]] * (tot_time - inc_time)
    prob = np.divide(prob, sum(prob))
    return prob.tolist()


def random_prob(num):
    #随机分布
    np.random.seed(0)
    prob = np.random.random(num)
    prob /= prob.sum()
    return prob

def random_to_empty_prob(time1,time2):
    #一半时间生成，另一半时间放空to_empty_prob
    prob = np.random.random(time1)
    prob /= prob.sum()
    prob = np.concatenate(prob,np.zeros(time2-time1),axis=0)
    return prob


if __name__ == '__main__':
    import os, sys
     # env_settings
    # cfg_file = './cfg/hangzhou/config_hangzhou_4x4.json'
    # cfg_file = './cfg/hangzhou/config_hangzhou_4x4_save.json'
    # cfg_file = './cfg/hangzhou/config_hangzhou_9000.json'
    # cfg_file = './cfg/hangzhou/config_hangzhou_14000.json'
    # cfg_file = './cfg/jinan/config_jinan_3x4.json'
    # cfg_file = './cfg/jinan/config_jinan_3x4_save.json'
    # cfg_file = './cfg/jinan/config_jinan_3000.json'
    cfg_file = './cfg/jinan/config_jinan_11000.json'
    # cfg_file = './cfg/jinan/config_jinan_11000_save.json'
    # cfg_file = './cfg/manhattan/config_manhattan_16x3.json'

    # cfg_file = './cfg/nanchang/config_nanchang_round2_save.json
    # cfg_file = './cfg/nanchang/config_nanchang_round2_split1.json'
    # cfg_file = './cfg/nanchang/config_nanchang_round2_split2.json'
    # cfg_file = './cfg/nanchang/config_nanchang_round2_split3.json'
    # cfg_file = './cfg/nanchang/config_nanchang_round2_split4.json'
    # cfg_file = './cfg/nanchang/config_nanchang_warm_up.json'
    # cfg_file = './cfg/nanchang/config_nanchang_warm_up_save.json'
    # cfg_file = './cfg/xuancheng/config_xuancheng.json'
 
    """ mynet """
    # cfg_file = 'cfg/config_mynet.json'
    # cfg_file = 'cfg/config_mynet_6000.json'
    # cfg_file = 'cfg/config_mynet8.json'
    # cfg_file = 'cfg/config_mynetred8.json'
    # cfg_file = 'cfg/config_mynetleft.json'

    """" xuancheng"""
    # cfg_file = 'cfg/config_xuancheng.json'
    # cfg_file = 'cfg/config_xuancheng_save.json'
    from agent import FTAgent, MPAgent, OAMAgent,DRQNAgent,GRUAgent,MPLight,OAMOAgent,WebsterAgent # DQNAgent
    from agent.oam_agent_plus import OAMAgentPlus 
    from agent.baseline import CoLight,FRAP ,AttendLight,MPLight

    duration = 3600
    gen_flow =None 
    # duration =1800
    gen_flow= {'duration': 3600, 'volume': 11000, 'prob': random_prob(3600)}
    # gen_flow= {'duration': 3600, 'volume':7000, 'prob': random_prob(3600)}
    
    # versions = range(1,4)
    versions = [1]
    parallel = True if len(versions)>1 else False
    evaluate = False
    evaluate = True
 
    if  (not parallel) and evaluate:
        version = versions[0]
        last_cfg_file = cfg_file
        # output_dir ='./output/road_model/nc/'+str(agents_id)+'_train'
        # output_dir ='./output/baseline/MPLight_nanchang_update_10_v5'
        # output_dir = f'./output/OAM/oam_no_constraint_v{str(version)}'
        # output_dir ='./output/OAM_hard/oam_no_constraint_v'+str(version)

        output_dir = f'./output/OAMO/phase_mask_v{str(version)}'
        # output_dir =f'./output/road_model/jn/{version}_train'
        # output_dir =f'./output/road_model/jn/jn_cdq_{version}_train'
        # output_dir =f'./output/road_model/hz/{version}_train'
        # output_dir =f'./output/OAMO/single_process_v{version}'
        if gen_flow:
            new_cfg_file = generate_flow_file(output_dir,gen_flow,cfg_file)
            last_cfg_file = new_cfg_file
            result_dir=get_result_dir(output_dir,gen_flow,last_cfg_file,change=False)
        else:
            result_dir= output_dir+'/result_jn_11000'
        volume =gen_flow['volume'] if gen_flow else None 
        print(f'evaluate_cfg: {cfg_file}')
        print(f'evaluate_gen: {volume}')
        run_episodes(output_dir=output_dir, cfg_file=last_cfg_file, gen_flow=gen_flow,duration=duration,result_dir=result_dir)
    elif parallel and evaluate:
        output_dirs = [f'./output/road_model/jn/{version}_train' for version in versions]
        if gen_flow:
            new_cfg_files = [generate_flow_file(output_dir,gen_flow,cfg_file) for output_dir in output_dirs]
            last_cfg_files = new_cfg_files
            result_dirs = [get_result_dir(output_dir,gen_flow,last_cfg_file,change=False) \
                for output_dir,last_cfg_file in zip(output_dirs,last_cfg_files)]
        else:
            result_dirs = [output_dir+'/result_4phase' for output_dir in output_dirs]
        process = [Process(target=run_episodes, args=(output_dir, cfg_file, gen_flow, duration, result_dir)) \
            for output_dir, result_dir in zip(output_dirs, result_dirs)]

        for p in process:
            p.start()
        for p in process:
            p.join()
        for p in process:
            p.terminate()
    
    compare_baseline_switch = False
    # compare_baseline_switch = True
    """与benchmark进行比较"""
    CDQ_output_dirs=['./output/road_model/hz/1_train',
    './output/road_model/hz/2_train',
    './output/road_model/hz/3_train']
    OAM_output_dirs=[
    './output/OAM_hz_9k/1',
    './output/OAM_hz_9k/3',
    './output/OAM_hz_9k/5'
    ]
    ### constraint abalation
    no_constraint =[ 
        './output/OAM/oam_no_constraint_v0',
        './output/OAM/oam_no_constraint_v1',
        './output/OAM/oam_no_constraint_v5',
    ]
    hard_constraint =[
        './output/OAM_hard/oam_no_constraint_v0',
        './output/OAM_hard/oam_no_constraint_v1',
        './output/OAM_hard/oam_no_constraint_v5',
    ]
    soft_constraint =[
        './output/road_model/jn/jn_cdq_1_train',
        './output/road_model/jn/jn_cdq_2_train',
        './output/road_model/jn/jn_cdq_5_train',
    ]
    ### regularize abalation
    OAM_with_regularize =[
        './output/OAM/oam_with_regularize_v6',
        './output/OAM/oam_with_regularize_v7',
        './output/OAM/oam_with_regularize_v8',
    ]
    OAM_without_regularize =[
        './output/OAM/oam_without_regularize_v3',
        './output/OAM/oam_without_regularize_v5',
        './output/OAM/oam_without_regularize_v8',
    ]
    ### phase mask
    phase4_mask =[
        './output/OAMO/phase_mask_v6',
        './output/OAMO/phase_mask_v10',
        './output/OAMO/phase_mask_v11',
    ]
    phase8_mask =[
        './output/OAMO/3_train_4phase',
        './output/OAMO/4_train_4phase',
        './output/OAMO/2_train_4phase',
    ]
    
    ### parallel
    OAM=[
        './output/OAM/oam_no_constraint_v0',
        './output/OAM/oam_no_constraint_v12',
        './output/OAM/oam_v25',
    ]
    single_process = [
        "./output/OAMO/single_process_v4",
        "./output/OAMO/single_process_v1",
        "./output/OAMO/single_process_v2",
    ]
    hard_constraint2=[
        './output/OAM_hard/oam_no_constraint_v0',
        './output/OAM_hard/oam_no_constraint_v6',
        './output/OAM_hard/oam_no_constraint_v12',
    ]
    parallel_process = [
        "./output/road_model/jn/3_train",
        "./output/road_model/jn/6_train",
        "./output/road_model/jn/7_train",
    ]

    agents_output ={}
    # agents_output['Without_constraint'] = no_constraint
    # agents_output['Hard_constraint'] = hard_constraint
    # agents_output['Soft_constraint'] = soft_constraint
    # agents_output['OAM_with_regularize'] = OAM_with_regularize
    # agents_output['OAM_without_regularize'] = OAM_without_regularize
    agents_output['OAMO(without complete action space)'] = phase4_mask
    # agents_output['OAMO(with phase mask)'] = phase8_mask
    agents_output['OAMO(without constraint)'] = OAM
    agents_output['OAMO(without parallel training)'] = single_process
    # agents_output['OAMO(with hard constraint)'] = hard_constraint2
    agents_output['OAMO'] = parallel_process



    if compare_baseline_switch:
    # compare_baseline(agents_output,result='/result_mynet_6000',env_name ='1x2',png_name='./constraint_compare.png')
        # compare_baseline(agents_output,result='/result_jn_4phase',env_name ='Jn',png_name='./compare.png')
        compare_baseline(agents_output,result='/result_jn_11000',env_name ='Jn_abalation',png_name='./compare.png')
