


#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File: train.py
@Time: 2022/03/15 14:04:15
@Author: Cl
@Desc: Train RL agent for signal control with parallel simulation
'''



import sys, os
import json
from time import time
import copy
from multiprocessing import Process, cpu_count
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import inspect
import shutil
from cityflow_env import CityFlowEnv, datetime_utc8


def simulate(e, agent,output_dir, parallel_num, duration, cfg_file, gen_flow=None, thread_num=1):
    np.random.seed(int(time() * 100000 % 100))
    eps_str = f'{(e-1)//parallel_num+1}-{(e-1)%parallel_num+1}'

    # select env
    simulator_cfg_file = cfg_file
    if gen_flow:
        #保证流量生成的随机性
        simulator_cfg_file = simulator_cfg_file.split('/')[-1]
        simulator_cfg_file = simulator_cfg_file.replace('.json', f'_random_{(e-1)%parallel_num+1}.json')
        simulator_cfg_file = '/'.join((agent.cache_dir, simulator_cfg_file))
        with open(cfg_file, 'r') as file:
            cfg = json.load(file)

        # if gen_flow['volume'] == 'length':
        #     env = CityFlowEnv(simulator_cfg_file)
        #     gen_flow['volume'] = sum([r['length'] for r in env.roads.values()]) // 5 * agent.road_lane_num

        cfg['flowFile'] = cfg['roadnetFile'].replace('.json', f'_flow_random_{(e-1)%parallel_num+1}.json')
        cfg['flowFile'] = cfg['flowFile'].split('/')[-1]
        cfg['flowFile'] = '/'.join((agent.cache_dir, cfg['flowFile']))
        if "replayLogFile" in cfg.keys():
            if (e % 1== 0):
                cfg["replayLogFile"] =  output_dir +'/' +cfg["replayLogFile"].split('.')[0]+'_'+str(e)+'.log'
                cfg["roadnetLogFile"] = output_dir +'/' +cfg["roadnetLogFile"]
            else:
                del cfg["replayLogFile"]
                del cfg["roadnetLogFile"]
                cfg['saveReplay'] = False
        with open(simulator_cfg_file, 'w') as file:
            json.dump(cfg, file,indent=4)


    env = CityFlowEnv(simulator_cfg_file, thread_num, gen_flow=gen_flow)
    # , simu_log=['net_accum', 'net_flow']

    agent_middle = Agent(env, agent.decision_interval, agent.output_dir)
    agent_middle.set_para((e - 1) // parallel_num + 1)


    ##   
    # 载入上一次的模型
    try:
        last_eps = (e-1)//parallel_num
        agent_middle.load_model(last_eps)
        print(f'{datetime_utc8()} Episode {eps_str}: Loading Middle Model {last_eps} Success')
    except:
        print(f'{datetime_utc8()} Episode {eps_str}: No Middle Model')

    agent_middle.init_memory()


    # simulation initilization
    env.reset()
    last_states = agent_middle.observe(env)

    t = 0
    while t < duration:
        print('\t\t' * ((e - 1) % parallel_num) + f'e={eps_str} t={t}', end='\r')

        # take action for all agents
        actions = agent_middle.act_explore(last_states)

        # # check vehicle speed
        # env.veh_check_speed()

        # record reward during action intervals
        dyna = env.step(actions, agent_middle.decision_interval)
        states, rewards = agent_middle.observe(env, cal_reward=True, dyna=dyna)
        t += agent_middle.decision_interval

        # add experience to replay memory
        agent_middle.memorize(last_states, actions, rewards, states)

        # update states
        last_states = states
        last_actions = actions

    print(' ' * 70 + f'\r{datetime_utc8()} Episode {eps_str} finished')

    # env.plot_MFD(x='net_accum', y='net_flow', width=60, desc=f'DQN_{e}', save=True, fig_dir=agent.fig_dir, log_dir=agent.log_dir)


    # 保存该轮memory
    agent_middle.save_memory(e)


def replay_mp(e, agent):
    process_tmp = Process(target=lambda a: a.replay(e), args=(agent,))
    process_tmp.start()
    process_tmp.join()
    process_tmp.terminate()


### Training
def train(agent, cfg_file, simu_dura, parallel_num=0, thread_num=1, output_dir=None, gen_flow=False):
    #parallel_num 
    def random_prob(num):
        #随机分布
        prob = np.random.random(num)
        prob /= prob.sum()
 
        return prob

    def random_prob_half(num):
        #一半时间生成，另一半时间放空
        prob = np.random.random(int(num/2))
        prob /= prob.sum()
        prob = np.concatenate((prob,np.zeros(int(num/2))),axis=0)
        return prob

    parallel = parallel_num > 0
    parallel_num = min(cpu_count(), parallel_num) if parallel_num > 0 else 1

    if parallel:
        print(f'{datetime_utc8()} The number of jobs to run in parallel is {parallel_num}/{cpu_count()}')

    if isinstance(cfg_file, str):
        cfg_file = [cfg_file] * parallel_num
    elif isinstance(cfg_file, list) and parallel_num % len(cfg_file) == 0:
        cfg_file *= parallel_num // len(cfg_file)

    assert len(cfg_file) == parallel_num

     ## read the historty if continue tarining 
    if not os.path.exists(agent.output_dir + '/reward.npy'):
        episodes_reward = [];training_loss = [];training_target=[];training_batch_reward=[];training_lr = []
    else:
        episodes_reward = np.load(agent.output_dir + '/reward.npy', allow_pickle=True).tolist()
        training_loss = np.load(agent.output_dir + '/training_loss.npy', allow_pickle=True).tolist()
        # training_batch_reward = np.load(agent.output_dir + '/training_reward.npy', allow_pickle=True).tolist()
        # training_target = np.load(agent.output_dir + '/training_target.npy', allow_pickle=True).tolist()
        # training_lr = np.load(agent.output_dir + '/training_lr.npy', allow_pickle=True).tolist()

    ## define start_eps if continue training
    if len(os.listdir(agent.model_dir))>0:
        listdir=os.listdir(agent.model_dir)
        listdir.sort(key=lambda x:int(x.split('_')[1].split('.')[0]))
        start_eps=int(listdir[-1].split('_')[1].split('.')[0])
    else:
        start_eps=0

    episode_reward = defaultdict(list)
    total_start = time()
  
    loss_n_epoch = []
    travel_time_list = []   
    for eps in range(start_eps*4+1, agent.episodes*parallel_num, parallel_num):
        # decay learning rate and epsilon，
        # parallel_num，并行的仿真数，多个仿真同时进行，（因为一轮仿真样本不够，因而可以并行跑，提升学习的效率）取0表示传统的取样训练
        agent.set_para((eps - 1) // parallel_num + 1)
        print(agent.model_dir)
        print(f'{datetime_utc8()} Episode {(eps - 1) // parallel_num + 1} start: lr={format(agent.lr, ".2e")}, epsilon={format(agent.epsilon, ".2e")}')

        start = time()

        if parallel:

            """ jinan config  """
            volumes = [500*(i+7) for i in range(parallel_num)] 
            # volumes = [500*(i+4) for i in range(parallel_num)]
            # volumes = [500*(3*i+1) for i in range(parallel_num)]
            """ nanchang config  """
            # volumes = [2000,2000,2000,2000]
            # volumes = [1500*(i+1) for i in range(parallel_num)] 
            process = [Process(target=simulate, args=(eps + i, agent, output_dir,parallel_num, simu_dura, cfg_file[i], \
                    # None,thread_num)) for i in range(parallel_num)]
                    # {'duration': simu_dura, 'volume': volumes[i], 'prob': random_prob(simu_dura)} ,\
                    # {'duration': simu_dura, 'volume': simu_dura * len(agent.roads) //80, 'prob': random_prob(simu_dura)},thread_num\
                    # 单训练环境时用路网自带的流量进行训练，多训练环境时才使用随机流量
                     {'duration': simu_dura, 'volume':volumes[i] , 'prob': random_prob(simu_dura)},thread_num\
                       )) for i in range(parallel_num)]# thread_num
                    #包含了gen_flow的参数，周期、流量、概率分布；加了i+1表明每轮生成的流量大小不一样，从而生成不同规模的流量
                    #分开采样，集中训练

            for p in process:
                p.start()
            for p in process:
                p.join()
            for p in process:
                p.terminate()

        else:
            simulate(eps, agent, output_dir,parallel_num, simu_dura, cfg_file[0],\
                    {'duration': simu_dura, 'volume': 2450, 'prob': random_prob(simu_dura)},thread_num )


        """
        load memory after simulation
        """
        
        print(f'{datetime_utc8()} Loading memory', end='\r')
        agent.init_memory()
        replay_index = [i for i in range(max(1, eps - (agent.buffer_eps - 1) * parallel_num), eps + parallel_num)]
        delete_index = [i for i in range(1, max(1, eps - (agent.buffer_eps - 1) * parallel_num))]
        for e in delete_index:
            file_name = '/'.join((agent.cache_dir, f'/memory_eps_{e}.npy'))
            try:
                os.remove(file_name)
            except:
                pass
        new_reward = []
        if agent.name=='CoLight':
            for i, e in enumerate(replay_index):
                memory_path = os.path.join(agent.cache_dir, f"memory_eps_{e}.npy")
                if not os.path.exists(memory_path):
                    print(f"{datetime_utc8()} Skip missing memory file: {memory_path}")
                    continue
                em = agent.load_memory(e)
                em_reward=[]
                if i >= len(replay_index) - parallel_num:
                    for a in agent.agent_list:
                       em_reward.append(em[a]['reward'])
                    em_reward=np.array(em_reward)
                    new_reward.append(np.mean(em_reward.sum(axis=0)))
            episodes_reward.append(new_reward)
        else:
            for i, e in enumerate(replay_index):
                memory_path = os.path.join(agent.cache_dir, f"memory_eps_{e}.npy")
                if not os.path.exists(memory_path):
                    print(f"{datetime_utc8()} Skip missing memory file: {memory_path}")
                    continue
                em = agent.load_memory(e)
                if i >= len(replay_index) - parallel_num:
                    new_reward.append(np.mean(np.reshape(em['reward'], [-1, len(agent.agent_list)]).sum(axis=1)))
                    del em
                    # new_reward.append(np.mean(np.reshape(em['reward'], (simu_dura // agent.decision_interval, -1)).sum(axis=1)))
            episodes_reward.append(new_reward)
            del new_reward




        """
        update Q network after simulation
        """
        print('------------------------------------------')
        print(f'{datetime_utc8()} Episode {(eps-1)//parallel_num+1} replay start')

        if parallel:
            replay_mp((eps-1)//parallel_num+1, agent)
        else:
            agent.replay(eps)

        print(f'{datetime_utc8()} Episode {(eps-1)//parallel_num+1} replay finished')
        print('------------------------------------------')



        """
        save simulation results
        """
        if parallel and parallel_num > 1:
            fig, ax = plt.subplots(parallel_num, sharex=True, figsize=(10, 8 * parallel_num))
            ax.flatten()
            for i in range(parallel_num):
                ax[i].plot(np.array(episodes_reward)[:, i])
        else:     
                   plt.plot(np.array(episodes_reward)[:, 0])
        plt.tight_layout()
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig('/'.join((agent.fig_dir, 'reward.png')))
        plt.close()
        np.save('/'.join((agent.output_dir, 'reward.npy')), episodes_reward)
        loss = np.load('/'.join((agent.cache_dir, 'loss.npy')), allow_pickle=True).tolist()
        # batch_target = np.load('/'.join((agent.cache_dir, 'batch_target.npy')), allow_pickle=True).tolist()
        # batch_reward = np.load('/'.join((agent.cache_dir, 'batch_reward.npy')), allow_pickle=True).tolist()
        # batch_lr = np.load('/'.join((agent.cache_dir, 'batch_lr.npy')), allow_pickle=True).tolist()      
        training_loss += loss
        # training_target+= batch_target
        # training_batch_reward+= batch_reward
        # training_lr+=batch_lr
        # loss_n_epoch.append(len(loss))

        plt.plot([i for i in range(len(training_loss))], training_loss)
        plt.savefig('/'.join((agent.fig_dir, 'loss.png')))
        plt.close()

        # plt.plot([i for i in range(len(training_target))], training_target)
        # plt.savefig('/'.join((agent.fig_dir, 'target.png')))
        # plt.close()
        # plt.plot([i for i in range(len(training_batch_reward))], training_batch_reward)
        # plt.savefig('/'.join((agent.fig_dir, 'batch_reward.png')))
        # plt.close()
        # plt.plot([i for i in range(len(training_lr))], training_lr)
        # plt.savefig('/'.join((agent.fig_dir, 'batch_lr.png')))
        # plt.close
        np.save('/'.join((agent.output_dir, 'training_loss.npy')), training_loss)
        # np.save('/'.join((agent.output_dir, 'loss_n_epoch.npy')), loss_n_epoch)
        # np.save('/'.join((agent.output_dir, 'training_target.npy')), training_target)
        # np.save('/'.join((agent.output_dir, 'training_reward.npy')), training_batch_reward) 
        # np.save('/'.join((agent.output_dir, 'training_lr.npy')), training_lr) 



        print(f'{datetime_utc8()} Cost time of {eps // parallel_num + 1}: {round(time() - start)} / {round(time() - total_start)}\n')


def add_code(agent):
    agent_file = inspect.getfile(agent.__class__)
    model_file = inspect.getfile(agent.model.__class__)
    _,agent_fname = os.path.split(agent_file)    
    _,model_fname = os.path.split(model_file)
    if not os.path.exists(agent.code_dir +'/'+ agent_fname):
        shutil.copy(agent_file, agent.code_dir +'/'+ agent_fname)
    if not os.path.exists(agent.code_dir +'/'+ model_fname):
        shutil.copy(model_file, agent.code_dir +'/'+ model_fname)   



if __name__ == '__main__':
    """  每个线程具有不同的仿真环境  """
    # cfg_file = ['./cfg/hangzhou/config_hangzhou_4x4.json', 'cfg/jinan/config_jinan_3x4.json', './cfg/manhattan/config_manhattan_16x3.json']
    # cfg_file = ['./cfg/nanchang/config_nanchang_round2_split1.json', 'cfg/nanchang/config_nanchang_round2_split2.json', './cfg/nanchang/config_nanchang_round2_split3.json']


    """  每个线程具有相同的仿真环境  """
    # cfg_file = './cfg/hangzhou/config_hangzhou_4x4.json'
    # cfg_file = './cfg/jinan/config_jinan_3x4.json'
    # cfg_file = './cfg/jinan/config_jinan_3x4_save.json'
    # cfg_file = './cfg/manhattan/config_manhattan_16x3.json'
    # cfg_file = './cfg/nanchang/config_nanchang_round2_save.json'
    # cfg_file = './cfg/nanchang/config_nanchang_round2_split1.json'
    # cfg_file = './cfg/nanchang/config_nanchang_round2_split2.json'
    # cfg_file = './cfg/nanchang/config_nanchang_round2_split3.json'
    # cfg_file = './cfg/nanchang/config_nanchang_warm_up.json'
    # cfg_file = './cfg/nanchang/config_nanchang_warm_up_save.json'
    # cfg_file = 'cfg/xuancheng/config_xuancheng.json'
    cfg_file = './cfg/xuancheng/config_xuancheng_test.json'

    """ perimeter control """
    # cfg_file = './cfg/RL.json'
    # cfg_file = './cfg/RL_save.json'
    # cfg_file = './cfg/RL_8p.json'
    # cfg_file='./cfg/RL_8p_save.json'
    

    #仿真周期，1200的周期已经可以拿到想要的数据了，训练得也更快，3600的话道路在后面也已经堵上了，也无意义
    duration = 1200

    # from agent.dqn_agent import DQNAgent as Agent
    # from agent.oam_agent import OAMAgent as Agent
    # from agent.cdq_agent import CDQAgent as Agent
    # from agent.ms_agent import MSAgent as Agent
    # from agent.drqn_agent import DRQNAgent as Agent
    # from agent.Tau_GRU_agent import TauAgent as Agent
    # from agent.gru_agent import GRUAgent as Agent

    # from agent.oam_agent import OAMAgent as Agent
    # output_dir = './output/OAM/oam_hz_v1'

    # from agent.baseline.CoLight_agent import CoLight as Agent
    # output_dir = './output/baseline/CoLight_update_1000_flow_4'

    # from agent.baseline.FRAP_agent import FRAP as Agent
    # output_dir = './output/baseline/FRAP_jn_update_64_0.99_v5'

    # from agent.baseline.MPLight_agent import MPLight as Agent
    # output_dir = './output/baseline/MPLight_jn_update_64_0.99_v3'
    for version in range(31,36):

        from agent.baseline.AttendLight_agent import AttendLight as Agent
        output_dir = './output/baseline/AttendLight_nc_v'+str(version)

        # from agent.baseline.MPLight_agent import MPLight as Agent
        # output_dir = './output/baseline/MPLight_nanchang_update_1000_v'+str(version)

        # from agent.baseline.CoLight_agent import CoLight as Agent
        # output_dir = './output/baseline/CoLight_hz_v'+str(version)



        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        with open('/'.join((output_dir, 'config.txt')), 'w') as file:
            print(cfg_file, file=file)
            print(duration, file=file)
        # virtual environment and agent for transfer parameterswanshan
        if isinstance(cfg_file, str): 
            env = CityFlowEnv(cfg_file)
        elif isinstance(cfg_file, list):
            env = CityFlowEnv(cfg_file[0])
        agent = Agent(env, 10, output_dir=output_dir)
        agent.episodes = 150

  

        add_code(agent)
        # simulate, parallel_num
        train(agent, cfg_file, duration, parallel_num=1, output_dir=output_dir, gen_flow=True, thread_num=1)
