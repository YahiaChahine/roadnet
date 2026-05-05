#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File: mp_agent.py
@Time: 2022/06/03 14:37:20
@Author: Cl
@Desc: SOTL signal control agent
'''


from .base_agent import BaseAgent


class SOTLAgent(BaseAgent):
    ''' Self-organizing traffic lights agent '''
    name = 'SOTL'

    def __init__(self, env, decision_interval, max_dist, max_dist_veh_num, min_red_veh_num, min_green_itv):
        super().__init__(env, decision_interval)
        self.env_phase_map = env.env_phase_map
        self.max_dist = max_dist
        self.max_dist_veh_num = max_dist_veh_num
        self.min_red_veh_num = min_red_veh_num
        self.min_green_itv = min_green_itv

    def observe(self, env):
        state = {}
        lane_veh_num = env.get_lane_veh_num()
        veh_dist = env.get_veh_distance(True)
        lane_max_dist_veh_num = {}
        for lane_id, veh_id_list in env.get_lane_veh_id().items():
            lane_max_dist_veh_num[lane_id] = len(list(filter(lambda x: veh_dist[x] < self.max_dist, veh_id_list)))

        for agent_id in env.agents:
            valid_phase = [i for i, p in enumerate(env.env_phase_map(agent_id, True)) if p]
            state[agent_id] = {'valid_phase': valid_phase}

            if valid_phase:
                inlane = env.get_inter_sort_lane(agent_id)[:self.road_lane_num * 4]
                all_red_inlane = env.get_inter_phase_lane(agent_id, -1, 'in')
                phase_inlane = env.get_inter_phase_lane(agent_id, self.cur_phase[agent_id], 'in')

                state[agent_id]['no_phase_inlane_veh_num'] = sum([lane_veh_num.get(lane_id, 0) for lane_id in inlane if lane_id not in all_red_inlane and lane_id not in phase_inlane])
                state[agent_id]['phase_max_dist_veh_num'] = sum([lane_max_dist_veh_num[lane_id] for lane_id in phase_inlane])

        return state

    def _get_action(self, state):
        actions = {}

        for a, s in state.items():
            agent_cur_phase = self.cur_phase[a]

            if not s['valid_phase']:
                actions[a] = -1
            elif agent_cur_phase == -1:
                actions[a] = s['valid_phase'][0]
            else:
                phase_max_dist_veh_num = s['phase_max_dist_veh_num']
                no_phase_inlane_veh_num = s['no_phase_inlane_veh_num']

                if self.cur_phase_itv[a] >= self.min_green_itv and phase_max_dist_veh_num < self.max_dist_veh_num and no_phase_inlane_veh_num >= self.min_red_veh_num:
                    phase_no = (s['valid_phase'].index(agent_cur_phase) + 1) % len(s['valid_phase'])
                    actions[a] = s['valid_phase'][phase_no]
                else:
                    actions[a] = agent_cur_phase

        return actions
