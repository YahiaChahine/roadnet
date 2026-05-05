#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''

@Time: 2024/08/15 13:41:39
@Author: QZ
@Desc: Webster signal control agent
'''


import numpy as np
from .base_agent import BaseAgent
import json, os

class WebsterAgent(BaseAgent):
    ''' fix-time agent env, decision_interval,flow_file,red_time '''

    name = 'Webster'

    def __init__(self, env, decision_interval,flow_file,red_time=2):
        super().__init__(env, decision_interval)
        self.env = env
        self.flow_file = flow_file
        self.current_phase = dict.fromkeys(self.agent_list, 4)
        self.cur_phase_time = dict.fromkeys(self.agent_list, 0)
        self.red_time = red_time
        self.decision_interval = decision_interval
        self.red_countdown = {agent_id: 0 for agent_id in env.agents}

    def road_to_inter(self, road_id,road_id2):
        split,split2 = road_id.split('_'),road_id2.split('_')
        x,y,dir = int(split[1]),int(split[2]),int(split[3])
        x2,y2,dir2 = int(split2[1]),int(split2[2]),int(split2[3])
        if dir ==0:
            x+=1
        elif dir == 1:
            y+=1
        elif dir ==2:
            x-=1
        elif dir == 3:
            y-=1
        if int(dir2)==int(dir):
            turn = 'straight'
        elif int(dir2) == (int(dir)+1)%4:
            turn = 'left'
        else:
            turn = 'right'
        inter = f'intersection_{x}_{y}'
        return inter,(dir+2)%4,turn

    def webster(self,demand,env):
        ### 根据self.demand生成websterSignal
        signal = {agent_id:[]  for agent_id in env.agents}
        ### 读取flow文件,计算各方向的需求
        for agent_id in env.agents:
            flow = self.demand[agent_id] 
            L = 3*4+3*4    # 启动损失及全红时间
            # y_list =[(flow[0][1]+flow[2][1])/3000,(flow[0][0]+flow[2][0])/3600,(flow[1][1]+flow[3][1])/3000,(flow[1][0]+flow[3][0])/3600]
            y_list =[(flow[dir][0]+flow[dir][1])/3200 for dir in range(4)]
            Y = sum(y_list)  # 饱和度总和
            C = int((1.5*L+5)/(1-Y))
            G_list = [int(y_list[dir]/Y*(C-L)+6-3) for dir in range(4)]
            signal[agent_id] = G_list
        return signal


    def observe(self, env,dyna):
        ##  Jinan Hangzhou
        # demand ={agent_id: {dir:[0,0,0] for dir in range(4)}  for agent_id in env.agents}
        # ### 读取flow文件,计算各方向的需求
        # with open(self.flow_file, 'r') as f:
        #     self.flow_data = json.load(f)
        #     for flow in self.flow_data:
        #         route = flow['route']
        #         for i in range(len(route)-1):
        #             inter,dir,turn = self.road_to_inter(route[i],route[i+1])
        #             if inter in env.agents:
        #                 if turn == 'straight':
        #                     demand[inter][dir][0] += 1
        #                 elif turn == 'left':
        #                     demand[inter][dir][1] += 1
        #                 else:
        #                     demand[inter][dir][2] += 1
        # self.demand = demand
        # self.signal = self.webster(demand,env)

        ##   validate_trip
        # self.signal = {agent_id: [30,30,30,30] for agent_id in env.agents}
        # self.signal['34180200005443'] = [14,15,7,14,15,7]
        # phase = {agent_id: [4,5,6,7] for agent_id in env.agents}
        # phase['34180200005443'] = [5,7,2,4,6,0]
        ##   validate_control
        self.agent_list = ['34180200000190','34180200000001','34180200000239','34180200000301','34180200008260','34180200000173','34180200000195','34180200006536','34180200006185','34180200000444','34180200000304']
        self.signal = {agent_id: [30,30,30,30] for agent_id in self.agent_list}
        """Perimeter Control"""
        self.signal['34180200000190'] = [30,30,30,10]
        self.signal['34180200000001'] = [30,30,20,30]
        self.signal['34180200000239'] = [30,30,10,30]
        self.signal['34180200000301'] = [30,30,30,10]
        self.signal['34180200008260'] = [10,30,30,30]
        self.signal['34180200000173'] = [30,20,30,30]
        self.signal['34180200000195'] = [20,30,30,30]
        """Unuseful MP"""
        self.signal['34180200006536'] = [20,20,20,20]
        self.signal['34180200006185'] = [20,20,20,20]
        self.signal['34180200000444'] = [20,20,20,20]
        self.signal['34180200000304'] = [20,20,20,20]
    
        phase = {agent_id: [4,5,6,7] for agent_id in self.agent_list}
        phase['34180200000239'] = [5,6,7,1]
        self.current_phase['34180200000239'] = 5

        return phase
        # return {agent_id: [0,1,2,3] for agent_id in env.agents}

    def _get_action(self, state):
        actions = {}


        for a, s in state.items():
            if not s:
                actions[a] = -1

            else:
                if self.cur_phase_time[a] < self.signal[a][s.index(self.current_phase[a])]:
                    actions[a] = self.current_phase[a]
                    self.cur_phase_time[a]+=self.decision_interval
                    if self.cur_phase_time[a] >= self.signal[a][s.index(self.current_phase[a])]:
                        self.red_countdown[a] = self.red_time
                else:
                    if self.red_countdown[a] > 0:
                        actions[a] = -1
                        self.red_countdown[a] -=self.decision_interval
                    else:
                        actions[a] = s[(s.index(self.current_phase[a]) + 1) % len(s)]
                        self.current_phase[a] = actions[a]
                        self.cur_phase_time[a] = self.decision_interval
        return actions
