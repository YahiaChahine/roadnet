#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File: converter_cbe.py
@Time: 2022/03/15 14:02:54
@Author: Cl
@Desc: Convert network file of CBEngine(KDD2021) to CityFlow
'''


import json
import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment


def process_roadnet(roadnet_file):
    # intersections[key_id] = {
    #     'have_signal': bool,
    #     'end_roads': list of road_id. Roads that end_time at this intersection. The order is random.
    #     'start_roads': list of road_id. Roads that start_time at this intersection. The order is random.
    #     'lanes': list, contains the lane_id in. The order is explained in Docs.
    # }
    # roads[road_id] = {
    #     'start_inter':int. Start intersection_id.
    #     'end_inter':int. End intersection_id.
    #     'length': float. Road length.
    #     'speed_limit': float. Road speed limit.
    #     'num_lanes': int. Number of lanes in this road.
    #     'inverse_road':  Road_id of inverse_road.
    #     'lanes': dict. roads[road_id]['lanes'][lane_id] = list of 3 int value. Contains the Steerability of lanes.
    #               lane_id is road_id*100 + 0/1/2... For example, if road 9 have 3 lanes, then their id are 900, 901, 902
    # }
    # agents[agent_id] = list of length 8. contains the inroad0_id, inroad1_id, inroad2_id,inroad3_id, outroad0_id, outroad1_id, outroad2_id, outroad3_id

    intersections = {}
    roads = {}
    agents = {}

    agent_num = 0
    road_num = 0
    signal_num = 0
    with open(roadnet_file, 'r') as f:
        lines = f.readlines()
        cnt = 0
        pre_road = 0
        is_obverse = 0
        for line in lines:
            line = line.rstrip('\n').split(' ')
            if ('' in line):
                line.remove('')
            if (len(line) == 1):
                if (cnt == 0):
                    agent_num = int(line[0])
                    cnt += 1
                elif (cnt == 1):
                    road_num = int(line[0]) * 2
                    cnt += 1
                elif (cnt == 2):
                    signal_num = int(line[0])
                    cnt += 1
            else:
                if (cnt == 1):
                    intersections[int(line[2])] = {'lat': float(line[0]), 'long': float(line[1]), 'have_signal': int(line[3]), 'end_roads': [], 'start_roads': [], 'lanes': []}
                elif (cnt == 2):
                    if (len(line) != 8):
                        road_id = pre_road[is_obverse]
                        roads[road_id]['lanes'] = {}
                        for i in range(roads[road_id]['num_lanes']):
                            roads[road_id]['lanes'][road_id * 100 + i] = list(map(int, line[i * 3:i * 3 + 3]))
                        is_obverse ^= 1
                    else:
                        roads[int(line[-2])] = {
                            'start_inter': int(line[0]),
                            'end_inter': int(line[1]),
                            'length': float(line[2]),
                            'speed_limit': float(line[3]),
                            'num_lanes': int(line[4]),
                            'inverse_road': int(line[-1])
                        }
                        roads[int(line[-1])] = {
                            'start_inter': int(line[1]),
                            'end_inter': int(line[0]),
                            'length': float(line[2]),
                            'speed_limit': float(line[3]),
                            'num_lanes': int(line[5]),
                            'inverse_road': int(line[-2])
                        }
                        intersections[int(line[0])]['end_roads'].append(int(line[-1]))
                        intersections[int(line[1])]['end_roads'].append(int(line[-2]))
                        intersections[int(line[0])]['start_roads'].append(int(line[-2]))
                        intersections[int(line[1])]['start_roads'].append(int(line[-1]))
                        pre_road = (int(line[-2]), int(line[-1]))
                else:
                    # 4 out-roads
                    signal_road_order = list(map(int, line[1:]))
                    now_agent = int(line[0])
                    in_roads = []
                    for road in signal_road_order:
                        if (road != -1):
                            in_roads.append(roads[road]['inverse_road'])
                        else:
                            in_roads.append(-1)
                    in_roads += signal_road_order
                    agents[now_agent] = in_roads
    for agent, agent_roads in agents.items():
        intersections[agent]['lanes'] = []
        for road in agent_roads:
            ## here we treat road -1 have 3 lanes
            if (road == -1):
                for i in range(3):
                    intersections[agent]['lanes'].append(-1)
            else:
                for lane in roads[road]['lanes'].keys():
                    intersections[agent]['lanes'].append(lane)

    return intersections, roads, agents


def _get_direction(road, out=True):
    if out:
        x = road['points'][1]['x'] - road['points'][0]['x']
        y = road['points'][1]['y'] - road['points'][0]['y']
    else:
        x = road['points'][-2]['x'] - road['points'][-1]['x']
        y = road['points'][-2]['y'] - road['points'][-1]['y']
    tmp = np.arctan2(y, x)
    return (tmp + np.pi * 2) % (np.pi * 2)


def position_turn(lat, long):
    R = 6371393
    L = R * np.pi / 180
    return lat * L, long * L


def convert(args):
    if args.cityflownet is None:
        args.cityflownet = args.cbenginenet.replace('.txt', '.json')

    phase = [
        {'time': 5, 'available_turn': []},
        {'time': 30, 'available_turn': [(0, 0), (0, 1)]},
        {'time': 30, 'available_turn': [(0, 0), (2, 2)]},
        {'time': 30, 'available_turn': [(2, 3), (2, 2)]},
        {'time': 30, 'available_turn': [(2, 3), (0, 1)]},
        {'time': 30, 'available_turn': [(1, 1), (1, 2)]},
        {'time': 30, 'available_turn': [(1, 1), (3, 3)]},
        {'time': 30, 'available_turn': [(3, 0), (3, 3)]},
        {'time': 30, 'available_turn': [(3, 0), (1, 2)]}
    ]
    turns = {
        1: {'type': 'turn_left', 'start_lane_index': 0},
        0: {'type': 'go_straight', 'start_lane_index': 1},
        3: {'type': 'turn_right', 'start_lane_index': 2}
    }


    intersections, roads, _ = process_roadnet(args.cbenginenet)

    new_inter = {}
    ori_x, ori_y = None, None
    for inter_id, inter_attr in intersections.items():
        y, x = position_turn(inter_attr['lat'], inter_attr['long'])
        if ori_x is None and ori_y is None:
            ori_x, ori_y = x, y
            x, y = 0, 0
        else:
            x -= ori_x
            y -= ori_y
        new_inter[inter_id] = {
            'id': f'intersection_{len(new_inter)}',
            'point': {
                'x': round(x, 2),
                'y': round(y, 2)
            },
            'width': 15,
            'virtual': False
        }

    new_road = {}
    for road_id, road_attr in roads.items():
        start_inter = road_attr['start_inter']
        end_inter = road_attr['end_inter']
        new_road[road_id] = {
            'points': [new_inter[start_inter]['point'], new_inter[end_inter]['point']],
            'startIntersection': new_inter[start_inter]['id'],
            'endIntersection': new_inter[end_inter]['id'],
            'lanes': [{'width': 4, 'maxSpeed': road_attr['speed_limit']} for _ in range(road_attr['num_lanes'])]
        }
        road_attr['out_direct'] = _get_direction(new_road[road_id])  # round(_get_direction(new_road[road_id]) / (np.pi / 2)) % 4
        road_attr['in_direct'] = _get_direction(new_road[road_id], False)  # round(_get_direction(new_road[road_id], False) / (np.pi / 2)) % 4

    for inter_id, inter_attr in intersections.items():
        inter_start_road_id = sorted(inter_attr['start_roads'], key=lambda x: roads[x]['out_direct'])
        out_direct = list(map(lambda x: roads[x]['out_direct'], inter_start_road_id))
        dir_diff = np.abs(np.linspace(0, np.pi * 3 / 2, 4) - np.reshape(out_direct, (-1, 1)))
        _, dir_idx = linear_sum_assignment(dir_diff)
        for road_id, dir_no in zip(inter_start_road_id, dir_idx):
            road_attr = roads[road_id]
            dir_no = int(dir_no)
            new_road[road_id]['id'] = f"{new_inter[road_attr['start_inter']]['id'].replace('intersection', 'road')}_{dir_no}"
            road_attr['out_dir_no'] = dir_no
            roads[roads[road_id]['inverse_road']]['in_dir_no'] = (dir_no + 2) % 4

    for inter_id, inter_attr in intersections.items():
        inter_road_id = sorted(inter_attr['end_roads'], key=lambda x: roads[x]['out_direct']) + sorted(inter_attr['start_roads'], key=lambda x: roads[x]['in_direct'])
        if len(inter_road_id) > 8:
            print(f"Warning: {new_road[start_road_id]['id']} has more than 4 legs")
            continue

        new_inter[inter_id]['roads'] = list(map(lambda x: new_road[x]['id'], inter_road_id))
        new_inter[inter_id]['roadLinks'] = []
        road_link_map = {}
        right_road_link = []
        for start_road_id in inter_attr['end_roads']:
            start_road = roads[start_road_id]
            for end_road_id in inter_attr['start_roads']:
                if end_road_id == start_road['inverse_road']:
                    continue
                end_road = roads[end_road_id]
                turn = turns[(end_road['out_dir_no'] - start_road['in_dir_no'] + 4) % 4]
                road_link = {
                    'type': turn['type'],
                    'startRoad': new_road[start_road_id]['id'],
                    'endRoad': new_road[end_road_id]['id'],
                    'direction': start_road['in_dir_no'],
                    'laneLinks': [{'startLaneIndex': turn['start_lane_index'], 'endLaneIndex': i, 'points': []} for i in range(end_road['num_lanes'])]
                }

                road_link_map[(start_road['in_dir_no'], end_road['out_dir_no'])] = len(new_inter[inter_id]['roadLinks'])
                if turn['type'] == 'turn_right':
                    right_road_link.append(len(new_inter[inter_id]['roadLinks']))
                new_inter[inter_id]['roadLinks'].append(road_link)

        if inter_attr['have_signal']:
            inter_phase = [{'time': phase[0]['time'], 'availableRoadLinks': right_road_link}]
            for p in phase[1:]:
                _p = {
                    'time': p['time'],
                    'availableRoadLinks': [road_link_map[t] for t in p['available_turn'] if t in road_link_map]
                }
                # if len(_p['availableRoadLinks']) < 2:
                if not _p['availableRoadLinks']:
                    continue
                _p['availableRoadLinks'] += right_road_link
                inter_phase.append(_p)

            if len(inter_phase) > 1:
                fix_road = set.intersection(*[set(p['availableRoadLinks']) for p in inter_phase[1:]])
                inter_phase[0]['availableRoadLinks'] = list(fix_road)

                inter_phase[1:] = sorted(inter_phase[1:], key=lambda x: len(x['availableRoadLinks']))
                for i, p1 in enumerate(inter_phase[1:], 2):
                    for p2 in inter_phase[i:]:
                        if p1['availableRoadLinks'] == inter_phase[0]['availableRoadLinks'] or set(p1['availableRoadLinks']) <= set(p2['availableRoadLinks']):
                            p1.clear()
                            break

            inter_phase = [p for p in inter_phase if p]

        else:
            inter_phase = [{'time': phase[0]['time'], 'availableRoadLinks': list(range(len(new_inter[inter_id]['roadLinks'])))}]

        new_inter[inter_id]['trafficLight'] = {
            # 'roadLinkIndices': list(range(len(new_inter[inter_id]['roadLinks']))),
            'lightphases': inter_phase
        }

    intersections = list(new_inter.values())
    roads = list(new_road.values())

    with open(args.cityflownet, 'w') as file:
        json.dump({'intersections': intersections, 'roads': roads}, file)


    # convert flow file
    if args.cbengineflow is not None:
        flow_list = []
        veh_dict = {"length": 5.0, "width": 2.0, "maxPosAcc": 2.0, "maxNegAcc": 4.5, "usualPosAcc": 2.0, "usualNegAcc": 4.5, "minGap": 2.5, "maxSpeed": 27.778, "headwayTime": 2}
        n_single_road = 0

        with open(args.cbengineflow, 'r') as file:
            n_flow = int(file.readline().strip())

            for _ in range(n_flow):
                start_time, end_time, interval = map(int, file.readline().split())
                len_route = int(file.readline().strip())
                if len_route == 1:  # route consisting of single road is not allowed in cityflow
                    n_single_road += 1
                    file.readline()
                    continue

                route = map(int, file.readline().strip().split())
                route = list(map(lambda x: new_road[x]['id'], route))
                flow_list.append({
                    'vehicle': veh_dict,
                    'route': route,
                    'interval': interval,
                    'startTime': start_time,
                    'endTime': end_time
                })

        if args.cityflowflow is None:
            args.cityflowflow = args.cbengineflow.replace('.txt', '.json')
        with open(args.cityflowflow, 'w') as file:
            json.dump(flow_list, file)

        if n_single_road:
            print(f'Warning: {n_single_road} single road routes (in {n_flow}) not in new flow file')

    return intersections, roads


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-sn", "--cbenginenet", type=str, default='shp_gdf/nanchang_warm_up/roadnet_warm_up_1000.txt')
    parser.add_argument("-sf", "--cbengineflow", type=str, default='shp_gdf/nanchang_warm_up/flow_warm_up_1000.txt')
    parser.add_argument("-tn", "--cityflownet", type=str, default=None)
    parser.add_argument("-tf", "--cityflowflow", type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    intersections, roads = convert(args)
    print("Cityflow net file generated successfully!")
