#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File: cityflow_env.py
@Time: 2022/03/15 13:28:26
@Author: Cl
@Desc: Simulator based on CityFlow
'''


from cmath import phase
import os
import json
import copy
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import cityflow
# from memory_profiler import profile


def datetime_utc8():
    return (datetime.utcnow() + timedelta(hours=8)).strftime("%Y-%m-%d_%H-%M-%S")


class EnvConfig:
    def __init__(self):
        self.phase_expansion = {0: [1, 0, 0, 0, 1, 0, 0, 0],
                                1: [0, 1, 0, 0, 0, 1, 0, 0],
                                2: [0, 0, 1, 0, 0, 0, 1, 0],
                                3: [0, 0, 0, 1, 0, 0, 0, 1],
                                4: [1, 1, 0, 0, 0, 0, 0, 0],
                                5: [0, 0, 1, 1, 0, 0, 0, 0],
                                6: [0, 0, 0, 0, 1, 1, 0, 0],
                                7: [0, 0, 0, 0, 0, 0, 1, 1]}

        self.road_links = [(0, 7), (0, 6), (1, 4), (1, 7), (2, 5), (2, 4), (3, 6), (3, 5)]
        self.road_lane_num = 3
        self.action_lane = [1,1,0, 1,1,0, 1,1,0, 1,1,0]
        if self.road_lane_num ==4:
            self.action_lane = [1, 1,1, 0, 1, 1, 1,0, 1, 1, 1,0, 1, 1,1, 0]
        self.phase_relation = [ [0,0,0,1,0,1,0],
                                [0,0,0,1,0,1,0],
                                [0,0,0,0,1,0,1],
                                [0,0,0,0,1,0,1],
                                [1,1,0,0,0,0,0],
                                [0,0,1,1,0,0,0],
                                [1,1,0,0,0,0,0],
                                [0,0,1,1,0,0,0]]
        self.phase_expansion_full = {0: [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
                                     1: [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
                                     2: [0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1],
                                     3: [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
                                     4: [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                                     5: [0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
                                     6: [0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
                                     7: [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1]}
        self.out_road = {0:[16,20,24,19,23,15,22,14,18,13,17,21],
                         1:[-1,-1,-1,19,23,23,22,18,18,17,17,21],
                         2:[20,20,24,-1,-1,-1,22,14,14,13,21,21],
                         3:[16,24,24,23,23,15,-1,-1,-1,13,17,17],
                         4:[16,20,20,19,15,15,14,14,18,-1,-1,-1]}
        # self.phase_lane = {0: ('NL', 1, 16), 1: ('NT', 2, 19),
        #                    2: ('EL', 4, 19), 3: ('ET', 5, 22),
        #                    4: ('SL', 7, 22), 5: ('ST', 8, 13),
        #                    6: ('WL', 10, 13), 7: ('WT', 11, 16)}


class CityFlowEnv:
    '''
    Simulator Environment with CityFlow
    '''
    def __init__(self, config, thread_num=1, simu_log=['phase'], gen_flow=None):
        ''' simu_log: phase, net_speed, net_flow, net_accum, net_density, lane_veh_delay, lane_in, lane_out '''
        self.env_config = EnvConfig()

        with open(config, 'r') as file:
            self.config = json.load(file)

        self.intersections, self.roads, self.agents = self.process_roadnet(self.config['roadnetFile'])

        if gen_flow:
            self.gen_flow_file(self.config['flowFile'], **gen_flow)

        self.eng = cityflow.Engine(config, thread_num=thread_num)
    
        lane_id_list = list(self.get_lane_veh_num().keys())

        self.current_phase = None

        # self.lane_veh_log = {lane_id: {} for lane_id in lane_id_list}

        self.simu_log = {f: [] for f in simu_log}

        self.interval = self.config['interval']
        self.step_time = self.config['step_time']
        self.all_red_time = self.config['all_red_time']

        self._inter_sort_lanes = {}
        self._env_phase_map = {}
        self.travel_times = []
        self.vehicle_enter_time = {}
        self.selected_veh_dict ={}

    def reset(self):
        self.eng.reset()
        self.current_phase = None
        # for veh_log in self.lane_veh_log.values():
        #     veh_log.clear()
        for r in self.simu_log.values():
            r.clear()

    def shot(self):
        return self.eng.snapshot()

    def move_replay_log(self, output_dir):
        ''' If save replay, move replay log file to output directory. '''
        if self.config['saveReplay']:
            from shutil import move
            for path in [self.config['roadnetLogFile'], self.config['replayLogFile']]:
                new_path = '/'.join((output_dir, path.split('/')[-1]))
                move(path, new_path)

    def _next_step(self, phase=None, route=None):
        # lane_veh_id = self.get_lane_veh_id()
        # veh_dist = self.get_veh_distance()

        if 'net_flow' in self.simu_log:
            lane_veh_id = self.get_lane_veh_id()

        if phase is not None:
            for agent_id, phase_no in phase.items():
                if phase_no > -1:

                    phase_no = np.array(self.env_phase_map(agent_id))[phase_no]
                    self.eng.set_tl_phase(agent_id, phase_no)
                else:
                    self.eng.set_tl_phase(agent_id, 0)

        if route is not None:
            for veh_id, veh_route in route.items():
                self.eng.set_vehicle_route(veh_id, veh_route)

        self.eng.next_step()


        # new_lane_veh_id = self.get_lane_veh_id()
        # new_veh_dist = self.get_veh_distance()
        # for lane_id, veh_log in self.lane_veh_log.items():
        #     for veh_id in new_lane_veh_id[lane_id]:
        #         if veh_id not in veh_log:
        #             veh_log[veh_id] = {'start_t': self.get_current_time(), 'end_t': self.get_current_time(), 'dist': new_veh_dist[veh_id]}
        #         else:
        #             veh_log[veh_id].update({'end_t': self.get_current_time(), 'dist': new_veh_dist[veh_id]})


        if 'net_speed' in self.simu_log:
            self.simu_log['net_speed'].append(self.get_network_speed())

        if 'net_flow' in self.simu_log:
            lane_flows = {l: [len(set(v) - set(lane_veh_id[l])) / self.interval, len(set(lane_veh_id[l]) - set(v)) / self.interval] \
                for l, v in self.get_lane_veh_id().items()}
            self.simu_log['net_flow'].append(self.get_network_flow(lane_flows))

        if 'net_density' in self.simu_log:
            self.simu_log['net_density'].append(self.get_network_density())

        if 'net_accum' in self.simu_log:
            self.simu_log['net_accum'].append(self.get_network_veh_num())

    def _update_lane_delay(self, lane_veh_id, veh_dist, lane_delay):
        new_lane_veh_id = self.get_lane_veh_id()
        new_veh_dist = self.get_veh_distance()

        for lane_id in lane_delay:
            count_veh = set(lane_veh_id[lane_id]) & set(new_lane_veh_id[lane_id])
            lane_delay[lane_id] += max(0, len(count_veh) * self.interval - \
                sum([(new_veh_dist[veh_id] - veh_dist[veh_id]) for veh_id in count_veh]) \
                    / self.get_lane_speed_limit(lane_id))

        return new_lane_veh_id, new_veh_dist


    def update_travel_time(self):
        vehicles = self.eng.get_vehicles(include_waiting=True)
        current_time = self.eng.get_current_time()
        # print(len(self.vehicle_enter_time))

        cur_vehicles_set = set(vehicles)
        exis_vehicles_set = set(list(self.vehicle_enter_time.keys()))

        running_vehicles = cur_vehicles_set & exis_vehicles_set

        for v_id in (cur_vehicles_set - running_vehicles):
            self.vehicle_enter_time[v_id] = current_time

        for v_id in (exis_vehicles_set - running_vehicles):
            self.travel_times.append(current_time - self.vehicle_enter_time[v_id])
            self.vehicle_enter_time.pop(v_id)

    def step(self, next_phase=None, step_time=None):
        if not step_time:
            step_time = self.step_time

        ini_lane_veh_id = lane_veh_id = self.get_lane_veh_id()
        veh_dist = self.get_veh_distance()
        lane_delay = dict.fromkeys(lane_veh_id.keys(), 0)
        # lane_veh_delay = {lane_id: defaultdict(int) for lane_id in lane_veh_id}

        if next_phase:
            next_phase = next_phase.copy()

            switch_phase = next_phase.copy()
            if self.current_phase:
                for agent_id in self.agents:
                    if self.current_phase[agent_id] != next_phase[agent_id]:
                        switch_phase[agent_id] = -1  # all red
            if step_time < self.all_red_time:
                for _ in range(step_time):
                    self._next_step(next_phase)
                self.current_phase = next_phase
            else:
                for _ in range(self.all_red_time):
                    self._next_step(switch_phase)
                    lane_veh_id, veh_dist = self._update_lane_delay(lane_veh_id, veh_dist, lane_delay)
                    # self.update_travel_time()

                for _ in range(step_time - self.all_red_time):
                    self._next_step(next_phase)
                    lane_veh_id, veh_dist = self._update_lane_delay(lane_veh_id, veh_dist, lane_delay)
                    # self.update_travel_time()

            self.current_phase = next_phase
        else:
            for _ in range(step_time):
                self._next_step()
        

        if 'phase' in self.simu_log:
            self.simu_log['phase'].append(self.current_phase)
        if step_time < self.all_red_time:
            lane_in = {}
            lane_out = {}
            lane_queen = {}
        else:
            new_lane_veh_id = lane_veh_id
            lane_in = {l: set(v) - set(ini_lane_veh_id[l]) for l, v in new_lane_veh_id.items()}
            lane_out = {l: set(ini_lane_veh_id[l]) - set(v) for l, v in new_lane_veh_id.items()}
            lane_queue = self.get_lane_queue()
            lane_nums = self.get_lane_veh_num()
        return {'lane_in': lane_in, 'lane_out': lane_out, 'lane_delay': lane_delay,'lane_queue':lane_queue,'lane_nums':lane_nums}

    # def step_without_switch():


    def process_roadnet(self, roadnet_file):
        with open(roadnet_file, 'r') as file:
            roadnet = json.load(file)

        intersections = {}
        roads = {}
        agents = []

        for inter_attr in roadnet['intersections']:
            inter_id = inter_attr['id']
            intersections[inter_id] = {
                'id': inter_id,
                'coord': (inter_attr['point']['x'], inter_attr['point']['y']),
                'have_signal': not inter_attr['virtual'] and inter_attr['roadLinks'] and \
                    len(inter_attr['trafficLight']['lightphases']) > 1,
                'virtual': inter_attr['virtual'],
                'out_roads': [],
                'in_roads': [],
                'road_links': {},
                'phase': [],
                'phase_lane_link': []
            }
            if intersections[inter_id]['have_signal']:
                agents.append(inter_id)

            road_links = intersections[inter_id]['road_links']
            lane_links = {}
            for i, rl in enumerate(inter_attr['roadLinks']):
                road_links[i] = (rl['startRoad'], rl['endRoad'])
                lane_links[i] = [(f"{rl['startRoad']}_{ll['startLaneIndex']}", \
                    f"{rl['endRoad']}_{ll['endLaneIndex']}") \
                    for ll in rl['laneLinks']]

            if not road_links:
                continue

            for p in inter_attr['trafficLight']['lightphases']:
                intersections[inter_id]['phase'].append(sorted([road_links[i] for i in p['availableRoadLinks']]))
                intersections[inter_id]['phase_lane_link'].append(sorted([lane_links[i] for i in p['availableRoadLinks']]))

        for road_attr in roadnet['roads']:
            road_id = road_attr['id']
            num_lanes = len(road_attr['lanes'])
            start_inter = road_attr['startIntersection']
            end_inter = road_attr['endIntersection']
            intersections[start_inter]['out_roads'].append(road_id)
            intersections[end_inter]['in_roads'].append(road_id)

            length = np.sum([np.linalg.norm([road_attr['points'][i]['x'] - road_attr['points'][i+1]['x'], road_attr['points'][i]['y'] - road_attr['points'][i+1]['y']]) for i in range(len(road_attr['points'])-1)])

            # suppose maxSpeed of each lane is equal
            speed_limit = road_attr['lanes'][0]['maxSpeed']
            lanes = [road_id + f'_{i}' for i in range(num_lanes)]
            roads[road_id] = {
                'id': road_id,
                'coord': tuple((p['x'], p['y']) for p in road_attr['points']),
                'start_inter': start_inter,
                'end_inter': end_inter,
                'length': length,
                'speed_limit': speed_limit,
                'num_lanes': num_lanes,
                'lanes': lanes,
                'start_roads': [sr for sr, er in intersections[start_inter]['road_links'].values() if er == road_id],
                'end_roads': [er for sr, er in intersections[end_inter]['road_links'].values() if sr == road_id]
            }


        for inter_id, inter_attr in intersections.items():
            in_roads = [None] * 4
            out_roads = [None] * 4
            for out_road_id in inter_attr['out_roads']:
                dir_no = int(out_road_id.split('_')[-1])
                out_roads[dir_no] = out_road_id
                for in_road_id in inter_attr['in_roads']:
                    if roads[in_road_id]['start_inter'] == roads[out_road_id]['end_inter']:
                        in_roads[dir_no] = in_road_id
                        break
            #### 该方向out没有，in有，即该方向为单行道
            for in_road_id in inter_attr['in_roads']:
                if in_road_id not in in_roads:
                    dir_no = (int(in_road_id.split('_')[-1]) + 2) % 4
                    in_roads[dir_no] = in_road_id

            intersections[inter_id]['sort_roads'] = in_roads + out_roads

        return intersections, roads, agents

    def env_outlane_phase(self, agent_id, outlanes_num,gap = 7.5):
        ''' phase of agent with outlane '''
        agent = self.intersections[agent_id]
        phase_map = np.copy(self.env_phase_map(agent_id))
        spill_outlanes = [l for l, num in outlanes_num.items() if num >= self.get_lane_length(l) //gap]
        for i, p in enumerate(phase_map):
            if p:
                phase_road_mask = np.array(agent['phase_lane_link'][p])
                for outlane in spill_outlanes:
                    if np.count_nonzero(phase_road_mask[:,:,1] == outlane)>=2:
                        phase_map[i] = 0
        if sum(phase_map) == 0:
            phase_map = self.env_phase_map(agent_id, mask=True)
        else:
            phase_map = np.array(phase_map)
            phase_map[np.where(phase_map>0)] = 1
        return phase_map

    def env_phase_map(self, agent_id, mask=False):
        if agent_id not in self._env_phase_map:
            agent = self.intersections[agent_id]

            assert len(agent['phase']) - 1 <= len(self.env_config.phase_expansion), f'Phase error of {agent_id}'

            agent_sort_roads = agent['sort_roads']
            phase_map = [0] * len(self.env_config.phase_expansion)
            full_phase_road = list(self.env_config.phase_expansion.values())
            for i, p in enumerate(agent['phase'][1:], 1):
                phase_road_mask = [0] * len(self.env_config.road_links)
                road_link_idx = tuple([tuple(map(lambda x: agent_sort_roads.index(x), rl)) for rl in p])
                for rl in road_link_idx:
                    if rl in self.env_config.road_links:
                        phase_road_mask[self.env_config.road_links.index(rl)] = 1
                ####   筛选到相应相位
                phase_idx_tuple =  np.where(np.linalg.norm(np.subtract(full_phase_road, phase_road_mask), axis=1)==0)
                if phase_idx_tuple[0].size==0:
                    ##  优先筛选到方向相位
                    phase_idx = np.argpartition(np.linalg.norm(np.subtract(full_phase_road, phase_road_mask), axis=1),2)[1]
                else:
                    phase_idx = phase_idx_tuple[0][0]
                phase_map[phase_idx] = i
                full_phase_road[phase_idx] = [10] * len(phase_road_mask)
            #TODO 
            phase_map[4:] =[0,0,0,0]
            self._env_phase_map[agent_id] = phase_map

        if mask:
            agent_phase_mask = list(map(lambda x: int(bool(x)), self._env_phase_map[agent_id]))
            return agent_phase_mask
        else:
            return self._env_phase_map[agent_id]

    def get_inter_sort_lane(self, inter_id):
        if inter_id not in self._inter_sort_lanes:
            inter = self.intersections[inter_id]
            inter_lane = []
            for road_id in inter['sort_roads']:
                if road_id:
                    inter_lane.extend([f'{road_id}_{i}' for i in range(self.env_config.road_lane_num)])
                else:
                    inter_lane.extend([None] * self.env_config.road_lane_num)
            self._inter_sort_lanes[inter_id] = inter_lane

        return self._inter_sort_lanes[inter_id]

    def get_inter_phase_lane(self, agent_id, phase_no, in_out=None):
        ''' lane links of phase of agent '''
        if phase_no > -1:
            phase_no = self.env_phase_map(agent_id)[phase_no]
        else:
            phase_no = 0

        phase_lane_link = self.intersections[agent_id]['phase_lane_link']
        if phase_lane_link[phase_no]:
            avail_lane_link = np.concatenate(phase_lane_link[phase_no])
            if in_out == 'in':
                return np.array(list(set(avail_lane_link[:, 0])))
            elif in_out == 'out':
                return np.array(list(set(avail_lane_link[:, 1])))
            else:
                return avail_lane_link
        else:
            return np.array([])
    

    def lane_road_id(self, lane_id):
        return '_'.join(lane_id.split('_')[:-1])

    def get_current_time(self):
        return self.eng.get_current_time()

    def get_avg_travel_time(self):
        return self.eng.get_average_travel_time()

    def get_veh_info(self, veh_id):
        """ Get vehicle information"""
        return self.eng.get_vehicle_info(veh_id)

    def get_veh_speed(self):
        return self.eng.get_vehicle_speed()

    def get_veh_distance(self, to_end=False):
        ''' Distance from start intersection '''
        veh_dist = self.eng.get_vehicle_distance()

        if to_end:
            lane_veh_id = self.get_lane_veh_id()
            for lane_id, veh_id_list in lane_veh_id.items():
                for veh_id in veh_id_list:
                    veh_dist[veh_id] = self.get_lane_length(lane_id) - veh_dist[veh_id]

        return veh_dist
        
    def get_veh_trajectory(self,agent_list,selected_route):
        lane_length = self.get_all_lane_length()
        veh_dist = self.get_veh_distance(to_end=True)
        def get_total_distance(veh_id,route):
            return sum([lane_length.get(l,0) for l in route]) - veh_dist[veh_id]
        selected_veh = {}
        for route in selected_route:
            for lane in self.roads[route]['lanes']:
                selected_veh[lane] = self.get_lane_veh_id(True)[lane]
                for veh_id in selected_veh[lane]:
                    if veh_id not in self.selected_veh_dict:
                        self.selected_veh_dict[veh_id] = {'route':[],'distance':[],'t':[],'speed':[]}
                    if lane not in self.selected_veh_dict[veh_id]['route']:
                        self.selected_veh_dict[veh_id]['route'].append(lane)
                    if self.get_current_time() not in self.selected_veh_dict[veh_id]['t']:
                        self.selected_veh_dict[veh_id]['t'].append(self.get_current_time())
                        self.selected_veh_dict[veh_id]['distance'] .append(get_total_distance(veh_id,self.selected_veh_dict[veh_id]['route']))
                        self.selected_veh_dict[veh_id]['speed'].append(self.get_veh_speed()[veh_id])
        return self.selected_veh_dict



    def get_lane_length(self, lane_id=None):
        if not lane_id:
            return -1

        road_id = self.lane_road_id(lane_id)
        return self.get_road_length(road_id)

    def get_lane_veh_num(self):
        return self.eng.get_lane_vehicle_count()

    def get_lane_veh_id(self, sort=False):
        lane_veh_id = self.eng.get_lane_vehicles()
        if sort:
            veh_dist = self.get_veh_distance(True)
            for lane_id, veh_id_list in lane_veh_id.items():
                for veh_id in veh_id_list:
                    lane_veh_id[lane_id] = sorted(veh_id_list, key=lambda x: veh_dist[x])
        return lane_veh_id

    def get_lane_speed_limit(self, lane_id):
        ''' m/s '''
        road_id = self.lane_road_id(lane_id)
        return self.get_road_speed_limit(road_id)

    def get_lane_avg_speed(self):
        ''' m/s '''
        veh_speed = self.get_veh_speed()
        lane_avg_speed = {}
        for lane_id, veh_id_list in self.get_lane_veh_id(False).items():
            lane_avg_speed[lane_id] = np.mean([veh_speed[v] for v in veh_id_list]) if veh_id_list else self.get_lane_speed_limit(lane_id)
        return lane_avg_speed

    def get_lane_density(self):
        ''' veh/m '''
        lane_density = self.eng.get_lane_vehicle_count()
        for lane_id, lane_veh_num in lane_density.items():
            lane_density[lane_id] /= self.get_lane_length(lane_id)
        return lane_density

    def get_lane_queue(self, queue_speed=1):
        # defalt queue_speed=1
        veh_speed = self.get_veh_speed()
        veh_dis_from_start = self.get_veh_distance()
        lane_queue = {}
        for lane_id, veh_id_list in self.get_lane_veh_id(True).items():

            lane_queue[lane_id] = 0
            for veh_id in veh_id_list:
                veh_info = self.eng.get_vehicle_info(veh_id)
                if (veh_speed[veh_id] > queue_speed)|(veh_dis_from_start[veh_id]>self.get_lane_length(lane_id)):
                    break
                lane_queue[lane_id] += 1
        return lane_queue

    def get_all_lane_length(self):
        lane_veh_id = self.get_lane_veh_id(True)
        all_lane_length = dict.fromkeys(lane_veh_id.keys(), 0)
        for lane_id, veh_id_list in lane_veh_id.items():
            all_lane_length[lane_id] = self.get_lane_length(lane_id)
        return all_lane_length

    def get_road_density(self):
        ''' veh/km '''
        road_density = defaultdict(int)
        for lane_id, lane_veh_num in self.get_lane_veh_num().items():
            road_id = self.lane_road_id(lane_id)
            road_density[road_id] += lane_veh_num

        for road_id in road_density:
            road_density[road_id] /= self.get_road_length(road_id) / 1000

        return dict(road_density)

    def get_road_speed_limit(self, road_id):
        ''' m/s '''
        return self.roads[road_id]['speed_limit']

    def get_road_avg_speed(self):
        ''' km/h '''
        veh_speed = self.get_veh_speed()
        road_veh_id = defaultdict(list)
        for lane_id, veh_id_list in self.get_lane_veh_id().items():
            road_id = self.lane_road_id(lane_id)
            road_veh_id[road_id] += veh_id_list

        road_avg_speed = {}
        for road_id in road_veh_id:
            if road_veh_id[road_id]:
                road_avg_speed[road_id] = np.mean([veh_speed[veh_id] for veh_id in road_veh_id[road_id]]) * 3.6
            else:
                road_avg_speed[road_id] = self.get_road_speed_limit(road_id) * 3.6

        return road_avg_speed

    def get_road_flow(self, lane_flows):
        ''' input: veh/s, output: veh/h '''
        # road_density = self.get_road_density()
        # road_flow = self.get_road_avg_speed()

        # for road_id in road_flow:
        #     road_flow[road_id] *= road_density[road_id]

        road_flow = defaultdict(int)

        for lane_id, flows in lane_flows.items():
            road_id = self.lane_road_id(lane_id)
            road_flow[road_id] += float(np.mean(lane_flows[lane_id]) * 3600)

        return dict(road_flow)

    def get_road_length(self, road_id):
        ''' m '''
        return self.roads[road_id]['length']
    
    def get_road_veh_id(self, road_id):
        road_veh_id = []
        lane_veh_id = self.get_lane_veh_id()
        road_lanes = self.roads[road_id]['lanes']
        for lane in road_lanes:
            road_veh_id.extend(lane_veh_id[lane])
        return road_veh_id

    def get_network_veh_id(self):
        return self.eng.get_vehicles()

    def get_network_density(self):
        ''' veh/km '''
        road_density = self.get_road_density()
        if not road_density:
            return 0
        network_density = np.average(list(road_density.values()), weights=[self.get_road_length(road_id) for road_id in road_density.keys()])
        return network_density

    def get_network_speed(self):
        ''' km/h '''
        road_speed = self.get_road_avg_speed()
        if not road_speed:
            return 0
        return np.average(list(road_speed.values()), weights=[self.get_road_length(road_id) for road_id in road_speed.keys()])

    def get_network_flow(self, lane_flows):
        ''' veh/h '''
        road_flow = self.get_road_flow(lane_flows)
        if not road_flow:
            return 0
        return np.average(list(road_flow.values()), weights=[self.get_road_length(road_id) for road_id in road_flow.keys()])

    def get_network_veh_num(self):
        return self.eng.get_vehicle_count()

    def plot_MFD(self, x, y, width=1, desc='', save=False, fig_dir='./figure', log_dir='./log'):
        def cluster(arr, width):
            return np.reshape(arr[:len(arr) - len(arr) % width], (-1, width)).mean(axis=1)

        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        # road_accum = defaultdict(int)
        # for lane_id, veh_log in self.lane_veh_log.items():
        #     road_id = self.lane_road_id(lane_id)
        #     road_accum[road_id] += len(veh_log)
        # with open(f'{log_dir}/{desc}_road_accum.json', 'w') as file:
        #     json.dump(road_accum, file)

        with open(f'{log_dir}/{desc}_simu_log.json', 'w') as file:
            json.dump(self.simu_log, file)

        plt.scatter(cluster(self.simu_log[x], width), cluster(self.simu_log[y], width), label=desc, alpha=0.5)
        if save:
            plt.xlim(left=0)
            plt.ylim(bottom=0)
            plt.ylabel('车辆数')
            plt.xlabel('流量')
            plt.savefig(f'{fig_dir}/{desc}_{x}_{y}_MFD.png')
            plt.close()

    @staticmethod
    def _get_direction(road, out=True):
        if out:
            x = road['points'][1]['x'] - road['points'][0]['x']
            y = road['points'][1]['y'] - road['points'][0]['y']
        else:
            x = road['points'][-2]['x'] - road['points'][-1]['x']
            y = road['points'][-2]['y'] - road['points'][-1]['y']
        tmp = np.arctan2(x, y)
        return tmp if tmp >= 0 else (tmp + 2 * np.pi)

    def gen_flow_file(self, filename, duration, volume, full_route=False, prob=None, veh_attr=None):
        np.random.seed(0)
        start_road_id_list = [k for k, v in self.roads.items() if v['end_roads']]
        # start_road_id_list = [k for k, v in self.roads.items() if self.intersections[v['start_inter']]['virtual']==True]
        end_road_id_list = [k for k, v in self.roads.items() if v['start_roads']]
        
        flow_list = []

        if veh_attr is None:
            veh_attr = [{"length": 5.0, "width": 2.0, "maxPosAcc": 2.0, "maxNegAcc": 4.5, "usualPosAcc": 2.0, "usualNegAcc": 4.5, "minGap": 2.5, "maxSpeed": 16.666, "headwayTime": 2}]

        if len(veh_attr) == 1:
            veh_attr[0]['ratio'] = 1
        assert sum(map(lambda x: x['ratio'], veh_attr)) == 1

        if prob is None:
            prob = [1 / duration] * duration

        # for _ in trange(volume, desc='Generating flow'):
        for veh_dict in veh_attr:
            veh_dict = veh_dict.copy()
            for _ in range(round(volume * veh_dict.pop('ratio'))):
                # start_time += 1
                # start_time = np.random.randint(duration)
                start_time = np.random.choice(np.arange(duration, dtype=int), p=prob)
                start_time = round(max(0, start_time))
                start_time = int(round(min(duration, start_time)))
                flow_dict = {'vehicle': veh_dict, 'interval':1, 'startTime': start_time, 'endTime': start_time}

                # generate route informatation for each flow type
                try:
                    road_id = start_road_id = np.random.choice(start_road_id_list)
                except:
                    pass
                # road_id = start_road_id = np.random.choice(start_road_id_list,p=[1/4, 1/8, 1/8, 1/8, 1/8, 1/4])
                route = [start_road_id]
                max_n_road = int(max(1, np.sqrt(len(self.roads))) * 2 + 0.5)
                # for _ in range(np.random.randint(min_n_road, min_n_road * 2)):
                direction = road_id.split('_')[-1]
                for _ in range(2, max_n_road):
                    road = self.roads[road_id]
                    start_inter = self.intersections[road['end_inter']]
                    # road_options = start_inter['out_roads'].copy()  # find the possible roads for the next route
                    road_options = list(set(road['end_roads']) - set([start_road_id]))
                    directions =[road_id.split('_')[-1] for road_id in road_options]
                    p_list = []
                    #  设置转向率
                    for i in range(len(directions)):
                        if directions[i]==direction:
                            p_list.append(3/7)
                        else:
                            p_list.append(2/7)
                        
                    if not road_options:
                        break
                    try:
                        road_id = np.random.choice(road_options,p=p_list)
                    except:
                        road_id = np.random.choice(road_options)
                    route.append(road_id)

                if not full_route:
                    route = [start_road_id, route[-1]]

                # while 1:
                #     start_road = np.random.choice(start_road_id_list)
                #     end_road = np.random.choice(end_road_id_list)
                #     if start_road != end_road:
                #         route = [start_road, end_road]
                #         break

                flow_dict['route'] = route
                flow_list.append(flow_dict)

        flow_list = sorted(flow_list, key=lambda x: x['startTime'])
        with open(filename, 'w') as file:
            json.dump(flow_list, file,indent=4)

        print(f'{datetime_utc8()} {filename} generated success')

        return filename