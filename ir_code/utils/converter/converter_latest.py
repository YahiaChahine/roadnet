import os
import sys
from sys import platform
import argparse
from collections import defaultdict
from tracemalloc import start
import sympy
from mpmath import degrees, radians
import copy
import math
import numpy as np
import json
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.geometry import Point

if platform == "linux" or platform == "linux2":
    # this is linux
    try:
        import traci
        import traci.constants as tc
        import sumolib
        from sumolib.net import Connection
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
            import traci
            import traci.constants as tc
            import sumolib
            from sumolib.net import Connection
        else:
            raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")
elif platform == "win32":
    os.environ['SUMO_HOME'] = 'D:\\software\\sumo-0.32.0'
    try:
        import traci
        import traci.constants as tc
        import sumolib
        from sumolib.net import Connection
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(
                os.path.join(os.environ["SUMO_HOME"], "tools")
            )
            import traci
            import traci.constants as tc
            import sumolib
            from sumolib.net import Connection
        else:
            raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")

elif platform =='darwin':
    os.environ['SUMO_HOME'] = "/Users/{0}/sumo/".format(os.environ.get('USER'))
    print(os.environ['SUMO_HOME'])
    try:
        import traci
        import traci.constants as tc
        import sumolib
        from sumolib.net import Connection
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(
                os.path.join(os.environ["SUMO_HOME"], "tools")
            )
            import traci
            import traci.constants as tc
            import sumolib
            from sumolib.net import Connection
        else:
            raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")
else:
    sys.exit("platform error")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sumonet", type=str,default='temp_xuancheng241008.net.xml')
    parser.add_argument("--cityflownet", type=str,default='roadnet_xuancheng250629.json')
    return parser.parse_args()

U_TURN_AS = "turn_left"
DEBUG = False
TRUE_CORRECTION_lane = True
SUMO_PROGRAM = True

############             edit              ###########
def get_direction_fron_connection(connection):
    _map = {
        Connection.LINKDIR_STRAIGHT: "go_straight",
        Connection.LINKDIR_TURN: "turn_u",
        Connection.LINKDIR_LEFT: "turn_left",
        Connection.LINKDIR_RIGHT: "turn_right",
        Connection.LINKDIR_PARTLEFT: "turn_left",
        Connection.LINKDIR_PARTRIGHT: "turn_right",
    }
    return _map[connection.getDirection()]

def process_edge(edge):
    lanes = []
    if TRUE_CORRECTION_lane:
        #### 逆序遍历车道,因为CityFlow中车道是从左到右的
        for inx,lane in enumerate(reversed(edge.getLanes())):
            ##  获取该车道对应的出口道
            outgoing_list = lane.getOutgoing()
            for outgoing in outgoing_list:
                new_lane = copy.copy(lane)
                direction = get_direction_fron_connection(outgoing)
                to_lane = outgoing.getToLane()
                # marky,add to_lane
                new_lane._cityflow_lane_id = f'{lane.getID()}|{to_lane.getID()}|{direction}'
                new_lane._cityflow_lane_inx = inx
                new_lane._direction = direction
                lanes.append(new_lane)
            if len(outgoing_list) == 0:
                new_lane = copy.copy(lane)
                new_lane._cityflow_lane_id = f'{lane.getID()}'
                new_lane._cityflow_lane_inx = inx
                new_lane._direction = 'go_end'
                lanes.append(new_lane)
    else:
        for lane in edge.getLanes():
            outgoing_list = lane.getOutgoing()
            for outgoing in outgoing_list:
                new_lane = copy.copy(lane)
                direction = get_direction_fron_connection(outgoing)
                to_lane = outgoing.getToLane()
                new_lane._cityflow_lane_id = f'{lane.getID()}|{to_lane.getID()}|{direction}'
                new_lane._direction = direction
                lanes.append(new_lane)
            if len(outgoing_list) == 0:
                new_lane = copy.copy(lane)
                new_lane._cityflow_lane_id = f'{lane.getID()}'
                new_lane._direction = 'go_end'
                lanes.append(new_lane)
    edge._cityflow_lanes = lanes[::-1]
    return edge




def _cityflow_get_lane_index_in_edge(lane, edge):
    for i, _lane in enumerate(edge._cityflow_lanes):
        if _lane._cityflow_lane_id == lane._cityflow_lane_id:
            return i
    raise Exception('lane in edge not found')

def _cityflow_get_lane_index_in_edge_cor(lane, edge):
    ## i = lane._cityflow_lane_id.split('|')[0]
    for i, _lane in enumerate(edge._cityflow_lanes):
        if _lane._cityflow_lane_id == lane._cityflow_lane_id:
            return _lane._cityflow_lane_inx
    raise Exception('lane in edge not found')


def point_tuple_to_dict(point_tuple):
    return {"x": point_tuple[0], "y": point_tuple[1]}


def _is_node_virtual(node):
    n = node
    edges = [edge for edge in n.getIncoming() + n.getOutgoing()]
    ids = list(set([e.getFromNode().getID() for e in edges] + [e.getToNode().getID() for e in edges]))
    if len(ids)<=2:
        return True
    else:
        return False


def group_connections_by_start_end(connections):
   ####  以起始点和终点为key，将connection分组
   ###   此处未获取roadlink车道的序号
    connection_group_result = defaultdict(list)
    for connection in connections:
        start_road = connection.getFrom()
        end_road = connection.getTo()
        direction = get_direction_fron_connection(connection)
        key = "{}|{}|{}".format(start_road.getID(), end_road.getID(), direction)
        connection_group_result[key].append(connection)
    return connection_group_result

def calc_edge_length(point1,point2):
    assert isinstance(point1, tuple)
    assert isinstance(point2, tuple)
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


def calc_edge_compass_angle(edge):
    north_ray = sympy.Ray((0, 0), (0, 1))
    # 反向算进入edge，直观。
    edge_ray = sympy.Ray(*edge.getShape()[:2][::-1])
    angle = north_ray.closing_angle(edge_ray)
    angle = (angle + 2 * sympy.pi) % (2 * sympy.pi)
    angle_degrees = float(degrees(angle))
    angle_radians = float(radians(degrees(angle)))
    edge._angle_degrees = round(angle_degrees,4)
    edge._angle_radians = round(angle_radians,4)
    return angle_degrees, angle_radians


def calc_edge_compass_angle_no_modify(edge):
    north_ray = sympy.Ray((0, 0), (0, 1))
    # 要算所有edge，所以不要反向。
    edge_ray = sympy.Ray(*edge.getShape()[:2])
    angle = north_ray.closing_angle(edge_ray)
    angle = (angle + 2 * sympy.pi) % (2 * sympy.pi)
    angle_degrees = float(degrees(angle))
    # angle_radians = float(radians(degrees(angle)))
    # edge._angle_degrees = round(angle_degrees,4)
    # edge._angle_radians = round(angle_radians,4)
    return angle_degrees


def process_intersection_simple_phase(intersection):
    if intersection['virtual']:
        return intersection

    all_green = {
        "time": 30,
        "availableRoadLinks": intersection['trafficLight']['roadLinkIndices']
    }
    all_red = {
        "time": 30,
        "availableRoadLinks": []
    }
    lightphases = [all_green]
    intersection['trafficLight']['lightphases'] = lightphases
    return intersection


def _cal_angle_pair(cluster):
    centroids = cluster['centroids']
    centroids = [x[0] for x in centroids]
    if len(centroids) == 4:
        pairs = [(centroids[0], centroids[2]), (centroids[1], centroids[3])]
    elif len(centroids) == 3:
        r1 = centroids[1] - centroids[0]
        r2 = centroids[2] - centroids[0]
        r3 = centroids[2] - centroids[1]
        near180_1 = abs(180 - r1)
        near180_2 = abs(180 - r2)
        near180_3 = abs(180 - r3)
        lista = [
            ([(centroids[0], centroids[1]), (centroids[2],)],near180_1),
            ([(centroids[0], centroids[2]), (centroids[1],)],near180_2),
            ([(centroids[0],),(centroids[1], centroids[2]),],near180_3),
        ]
        pairs = min(lista,key=lambda item:item[1])[0]
    elif len(centroids) == 2:
        pairs = [(centroids[0], centroids[1]), ]
    elif len(centroids) == 1:
        pairs = [(centroids[0],),]
    return pairs


def find_edges_by_angle(all_edges,angle):
    edges = []
    for edge in all_edges:
        if math.isclose(edge._angle_degrees , angle, abs_tol=0.0001):
        # if edge._angle_degrees == angle:
            edges.append(edge)
    if not edges:
        raise Exception('!!!no edge._angle_degrees = angle')
    return edges

def find_edges_by_cluster_centroid(all_edges,angle):
    edges = []
    for edge in all_edges:
        if math.isclose(edge._cluster_centroid[0] , angle, abs_tol=0.0001):
        # if edge._angle_degrees == angle:
            edges.append(edge)
    if not edges:
        raise Exception('!!!no edge._cluster_centroid[0] = angle')
    return edges



def get_all_turn_right_link_index(roadLinks):
    allow = []
    for index,roadlink in enumerate(roadLinks):
        if roadlink['type'] == 'turn_right':
            allow.append(index)
    return allow


def filter_roadlinks_by_startedge_and_turn_type(roadLinks,edge,turntype):
    result = []
    for index,roadlink in enumerate(roadLinks):
        if roadlink['startRoad'] == edge.getID() and roadlink['type']==turntype:
            result.append((index,roadlink))
    return result

def filter_roadlinks_by_startedge(roadLinks,lane_id,lane_num):
    result = []
    lane_split=lane_id.split('_')
    lane_index=lane_num-int(lane_split[-1])-1
    edge_id =lane_id[:-2]
    lane_id='_'.join(lane_split)
    for index,roadlink in enumerate(roadLinks):
        lane_index_list = []
        for laneLink in roadlink['laneLinks']:
            lane_index_list.append(laneLink['startLaneIndex'])
        lane_index_list = list(set(lane_index_list))

        if roadlink['startRoad'] == edge_id and int(lane_index) in lane_index_list:
            result.append((index,roadlink))
    return result

def filter_roadlinks_by_road_link(roadLinks,link):
    result = []
    for index,roadlink in enumerate(roadLinks):
        if (roadlink['startRoad'],roadlink['endRoad']) == link:
            result.append((index,roadlink))
    return result

def fill_empty_phase(current_phase,count):
    need_fill_count = count - len(current_phase)
    for x in range(need_fill_count):
        empty_phase_dict = {
            'availableRoadLinks': [],
            'time': 0,
        }
        current_phase.append(empty_phase_dict)
    return current_phase

all_phase_dict = {}
all_phase_link_dict = {}
node_outgoing_dict = {}


def min_square_width(node,coords):
    # Step 1: 计算凸包（Convex Hull）
    points = np.array(coords)
    ##  numpy去重
    points = np.unique(points, axis=0)
    if points.shape[0] == 1:
        center = points[0]
        return (5, Point(center[0], center[1]))
    elif points.shape[0] == 2:
        # print(node.getID())
        center = points.mean(axis=0)
        return (5, Point(center[0], center[1]))
    else:
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]  # 得到凸包点的坐标
        except:
            print(node.getID())
            print(points)
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]

        # Step 2: 找到最小外接矩形 (Oriented Bounding Box, OBB)
        polygon = Polygon(hull_points)
        

        # step 2.1 获取矩形中心
        min_rectangle = polygon.minimum_rotated_rectangle
        center = min_rectangle.centroid

        # Step 3: 获取最小外接矩形的边界坐标，并计算宽和高
        min_rect_coords = np.array(min_rectangle.exterior.coords)
        width = np.linalg.norm(min_rect_coords[0] - min_rect_coords[1])
        height = np.linalg.norm(min_rect_coords[1] - min_rect_coords[2])

        if node.getType() == 'traffic_light':
            width = min(15,min(width, height))
        else:
            width = min(5,min(width, height))
           
        # Step 4: 返回最小正方形的边长（最大边）
        return (width,center)


def get_laneidx_roadlink(start_lane,end_road,conections,intersection):
    end_lanes = []
    if len(intersection['roads'])==2:
        for idx,conection in enumerate(conections):
            if  conection in start_lane.getOutgoing():
                end_lanes.append(conection.getToLane())
    else:
        end_lanes = end_road._lanes
    return end_lanes

def get_lane_idx(lane,edge):
    for idx,_lane in enumerate(edge._cityflow_lanes):
        if lane.getOutgoing() == _lane.getOutgoing():
            return _lane._cityflow_lane_inx


def node_to_intersection(node,tls_dict,edge_dict):
    node_type = node.getType()
    node_coord = node.getCoord()
    node_shape = node.getShape()
    width,center = min_square_width(node,np.array(node_shape))
    intersection = {
        "id": node.getID(),
        "point": {"x": center.x, "y": center.y},
        "width": width,  # warning.路口宽度对于任意路口是未定义的.取15
        "roads": [edge.getID() for edge in node.getIncoming() + node.getOutgoing()],

        # "_roads":[{'id':}]
        "roadLinks": [],
        "trafficLight": {
            "roadLinkIndices": [],
            "lightphases": []
        },
        "virtual": _is_node_virtual(node)  # dead_end判断为virtual
    }


    connections_group = group_connections_by_start_end(node.getConnections())
    roadLinks = intersection['roadLinks']
    for k, v in connections_group.items():
        ## 1个connection有多个车道组成
        connection_template = v[0]
        start_road = connection_template.getFrom()
        end_road = connection_template.getTo()
        # 加上驶入方向的正北夹角
        raw_roadlink_type = get_direction_fron_connection(connection_template)
        roadLink = {
            "type": raw_roadlink_type,
            "startRoad": start_road.getID(),
            "endRoad": end_road.getID(),
            "direction": 0,  # WARNING: direction is falsely defined but it doesn't affect usage
            "laneLinks": []
        }
        ##  我认为是掉头
        if roadLink["type"] == "turn_u":
            roadLink["type"] = U_TURN_AS


        for start_lane in reversed(start_road._cityflow_lanes):
            if start_lane._direction != raw_roadlink_type:
                continue
            if TRUE_CORRECTION_lane:
                end_lane_group = get_laneidx_roadlink(start_lane,end_road, v,intersection)
                # for end_inx,end_lane in enumerate(end_lane_group): #根据sumo的换道规则，确定到达车道
                for end_inx, end_lane in enumerate(reversed(end_road._lanes)):
                    connection_template.getFromLane()
                    start_point = start_lane.getShape()[-1]
                    start_point = point_tuple_to_dict(start_point)
                    end_point = end_lane.getShape()[0]
                    end_point = point_tuple_to_dict(end_point)
                    ## startLaneIndex是否处理过
                    path = {
                        "startLaneIndex": _cityflow_get_lane_index_in_edge_cor(start_lane, start_road),
                        # "endLaneIndex": get_lane_idx(end_lane, end_road),#根据sumo的换道规则，确定到达车道序号
                        "endLaneIndex": end_inx,
                        # ytodo: 或许改为起始lane结束点，路口点，结束lane起始点。
                        "points": [start_point, end_point]
                    }
                    ###   防止加入重复路径
                    if path not in roadLink["laneLinks"]:
                        roadLink["laneLinks"].append(path)
            else:
                for end_lane in end_road._cityflow_lanes:
                    start_point = start_lane.getShape()[-1]
                    start_point = point_tuple_to_dict(start_point)
                    end_point = end_lane.getShape()[0]
                    end_point = point_tuple_to_dict(end_point)
                    path = {
                        "startLaneIndex": _cityflow_get_lane_index_in_edge(start_lane, start_road),
                        "endLaneIndex": _cityflow_get_lane_index_in_edge(end_lane, end_road),
                        "points": [start_point, end_point]  # warning 飞行模式
                    }
                    roadLink["laneLinks"].append(path)
        roadLinks.append(roadLink)


    for i, _ in enumerate(intersection["roadLinks"]):
        intersection["trafficLight"]["roadLinkIndices"].append(i)
    if node_type in ['traffic_light']:
        if node.getID() not in tls_dict:
            node._type = 'priority'
            node_type = 'priority'

    if node_type in ['dead_end']:
        pass
    if node_type in ['priority']:
        pass
    if node_type in ['right_before_left']:
        pass
    if node_type in ['dead_end','priority','right_before_left']:
        intersection = process_intersection_simple_phase(intersection)


    if node_type in ['traffic_light']:
        # print(node.getID())
        if SUMO_PROGRAM:
            all_phase = []
            nodeid = node.getID()
            all_phase_dict[nodeid] = []
            all_phase_link_dict[nodeid] = []
            ###  原始的方案，一个lane即一个link
            G_to_lane_dict = {}
            ###   现在的方案，一个lane对应数个link
            G_to_link_dict = {}

            ###   fix the bug
            try:   ###  connec[-1] 是index
                for connec in tls_dict[nodeid]._connections:
                    G_to_lane_dict[connec[-1]] = connec[0].getID()
                    G_to_link_dict[connec[-1]] = (connec[0].getID()[:-2],connec[1].getID()[:-2])
            except KeyError:
                print(f'KeyError: The key {nodeid} does not exist in tls_dict.')
                return intersection
            for phase in tls_dict[nodeid]._programs['0']._phases:
                # 绿灯时间
                duration = phase.duration
                # print(f'phase:{phase}')
                lane_list = []
                link_list = []
                # print(f'duration:{duration}')
                if 'y'in phase.state:
                    continue
                for i, alpha in enumerate(phase.state):
                    # print(f'i:{i}')
                    # print(f'alpha:{alpha}')
                    # print(f'G_to_lane_dict.keys():{G_to_lane_dict.keys()}')
                    if (alpha == 'G' or alpha == 'g') and i in G_to_lane_dict.keys():
                        lane_list.append(G_to_lane_dict[i])
                    if (alpha == 'G' or alpha == 'g') and i in G_to_link_dict.keys():
                        link_list.append(G_to_link_dict[i])

                lane_list_ = []
                # print(f'lane_list:{lane_list}')
                # for lane in lane_list:
                #     edge_id, lane_id = lane.split('_')
                #     lane_id = int(lane_id)
                #     lane_ = edge_id + '_' + str(len(edge_dict[edge_id]) - lane_id - 1)
                #     lane_list_.append(lane_)
                lane_list_=lane_list
                link_list_= link_list
                # print(f'lane_list_:{lane_list_}')

                all_phase_dict[nodeid].append(list(set(lane_list_)))
                all_phase_link_dict[nodeid].append(list(set(link_list_)))
                index_list = []

                # for _lane in lane_list_:
                #     ###   SUMO车道从右开始编号，CityFlow车道从左开始编号
                #     lane_num = len(edge_dict[_lane[:-2]])
                #     ###    bug：算法默认一个车道仅具一个功能
                #     index_roadlink_list = filter_roadlinks_by_startedge(roadLinks, _lane,lane_num)
                    # index_list += [item[0] for item in index_roadlink_list]
                for _link in link_list_:
                    index_roadlink_list = filter_roadlinks_by_road_link(roadLinks, _link)
                    index_list += [item[0] for item in index_roadlink_list]
                phase_dict = {
                    'time': duration,
                    'availableRoadLinks': list(set(index_list))
                }
                all_phase.append(phase_dict)
            intersection["trafficLight"]["lightphases"] = all_phase

            outgoing_lane_list = []
            edge_list_ = [edge_.getID() for edge_ in node.getOutgoing()]
            for edge in edge_list_:
                for i in range(len(edge_dict[edge])):
                    outgoing_lane_list.append(edge+'_'+str(i))
            node_outgoing_dict[nodeid] = outgoing_lane_list

        exiting_lane_list = []
        for edge in node.getOutgoing():
            exiting_lane_list.extend([lane.getID() for lane in edge.getLanes()])
    return intersection

def get_final_intersections(net,tls_dict,edge_dict):

    final_intersections = []
    net_nodes = net.getNodes()
    net_nodes_sorted = sorted(net_nodes,key=lambda n:n.getID())
    nodes = [(index,node) for index,node in enumerate(net_nodes_sorted)]
    nodes = nodes[:]
    for obj in nodes:

        index = obj[0]
        node = obj[1]

        intersection = node_to_intersection(node,tls_dict,edge_dict)
        if intersection["roads"] != []:
            final_intersections.append(intersection)

    return final_intersections

def get_final_roads(net,intersections):
    edges = net.getEdges()
    final_roads = []
    for edge in edges:
        start_intersection = edge.getFromNode()
        start_coord = start_intersection.getCoord()
        end_intersection = edge.getToNode()
        end_coord = end_intersection.getCoord()
        for intersection in intersections:
            if intersection['id'] == start_intersection.getID():
                start_intersection_width = intersection['width']
                start_point = intersection['point']
            if intersection['id'] == end_intersection.getID():
                end_intersection_width = intersection['width']
                end_point = intersection['point']
        if len(edge.getShape())<=-1:
            points = []

            points.append({
                "x": start_point['x'],
                "y": start_point['y'],
            })

            for i in range(len(edge.getShape())//2):
                if calc_edge_length(start_coord,edge.getShape()[i])>start_intersection_width:
                    first_point = (edge.getShape()[i][0], edge.getShape()[i][1])
                    second_point = (edge.getShape()[i+1][0], edge.getShape()[i+1][1])
                    break
            for i in range(len(edge.getShape())-1,len(edge.getShape())//2,-1):
                if calc_edge_length(end_coord,edge.getShape()[i])>end_intersection_width:
                    fourth_point = (edge.getShape()[i][0], edge.getShape()[i][1])
                    third_point = (edge.getShape()[i-1][0], edge.getShape()[i-1][1])
                    break
            # Calculate vectors
            angle = 0
            if 'first_point' in locals() and 'second_point' in locals():
                vec1 = ( first_point[0]-second_point[0],  first_point[1]-second_point[1])
                vec2 = (third_point[0] - fourth_point[0], third_point[1] - fourth_point[1])
                
                # Calculate the angle between the vectors
                dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
                mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
                mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
                if mag1 != 0 and mag2 != 0:
                    angle = math.acos(dot_product / (mag1 * mag2)) * (180 / math.pi)
            
            # Check if the angle is greater than 45 degrees
            if angle >=10:
                for i in range(len(edge.getShape())//2):
                    if calc_edge_length(start_coord,edge.getShape()[i])>start_intersection_width:
                        points.append({
                            "x": edge.getShape()[i][0],
                            "y": edge.getShape()[i][1],
                        })
                for i in range(len(edge.getShape())//2,len(edge.getShape())):
                    if calc_edge_length(end_coord,edge.getShape()[i])>end_intersection_width:
                        points.append({
                            "x": edge.getShape()[i][0],
                            "y": edge.getShape()[i][1],
                        })
            points.append({
                "x": end_point['x'],
                "y": end_point['y'],
            })
        else:
            points=[
                {
                    "x": start_point['x'],
                    "y": start_point['y'],
                },
                {
                    "x": end_point['x'],
                    "y": end_point['y'],
                }
            ]
        road = {
            "id": edge.getID(),
            "points": points,
            "lanes": [
            ],
            "startIntersection": start_intersection.getID(),
            "endIntersection": end_intersection.getID(),
        }
        if DEBUG:
            road['_compass_angle'] = calc_edge_compass_angle_no_modify(edge)
        lane_template = {
            "width":3.2 ,
            "maxSpeed": 11.111  # warning 放弃速度
        }
        if TRUE_CORRECTION_lane:
            for _v in edge._lanes:
                road["lanes"].append(lane_template)
        else:
            for _v in edge._cityflow_lanes:
                road["lanes"].append(lane_template)
        final_roads.append(road)
    return final_roads



def main(args):
    print("Converting sumo net file",args.sumonet)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    net = sumolib.net.readNet(os.path.join(dir_path,args.sumonet), withPrograms=True)
    # net = sumolib.net.readNet(os.path.join('/',args.sumonet), withPrograms=True)

    for edge in net.getEdges():
        ## 获取车道对应的出口道
        process_edge(edge)
    tls_dict = {}
    for tls in net.getTrafficLights():
        tls_dict[tls.getID()] = tls
    print(tls_dict)
    print('Start processing '+str(len(tls_dict))+" traffic lights")
    edge_dict = {}
    for edge_ in net.getEdges():
        edge_dict[edge_.getID()] = edge_._lanes

    final_intersections = get_final_intersections(net,tls_dict,edge_dict)

    for intersection in final_intersections:
        if intersection['virtual']:
            intersection['roadLinks'] = []

    final_roads = get_final_roads(net,final_intersections)

    result = {
        "intersections": final_intersections,
        "roads": final_roads
    }

    f = open(os.path.join(os.path.dirname(__file__),args.cityflownet), 'w')
    json.dump(result, f, indent=2)
    f.close()


if __name__ == '__main__':
    args = parse_args()

    main(args)
    print("Cityflow net file generated successfully!")


'''

Direction is meaningless
u turn type exists
'''