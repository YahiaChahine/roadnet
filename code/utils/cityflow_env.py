import pickle
import numpy as np
import json
import sys
import pandas as pd
import os
import cityflow as engine
import time
from multiprocessing import Process


class Intersection:
    def __init__(self, inter_id, dic_traffic_env_conf, eng, light_id_dict, path_to_log, lanes_length_dict,
                 topology_data=None):
        self.inter_id = inter_id
        if topology_data is None:
            self.inter_name = "intersection_{0}_{1}".format(inter_id[0], inter_id[1])
        else:
            self.inter_name = topology_data["inter_name"]
        self.eng = eng
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.lane_length = lanes_length_dict
        self.obs_length = dic_traffic_env_conf["OBS_LENGTH"]

        self.list_approachs = ["W", "E", "N", "S"]
        # corresponding exiting lane for entering lanes
        self.dic_approach_to_node = {"W": 0, "E": 2, "S": 1, "N": 3}
        self.list_phases = dic_traffic_env_conf["PHASE"]

        if topology_data is None:
            self.dic_entering_approach_to_edge = {"W": "road_{0}_{1}_0".format(inter_id[0] - 1, inter_id[1])}
            self.dic_entering_approach_to_edge.update({"E": "road_{0}_{1}_2".format(inter_id[0] + 1, inter_id[1])})
            self.dic_entering_approach_to_edge.update({"N": "road_{0}_{1}_3".format(inter_id[0], inter_id[1] + 1)})
            self.dic_entering_approach_to_edge.update({"S": "road_{0}_{1}_1".format(inter_id[0], inter_id[1] - 1)})
            self.dic_exiting_approach_to_edge = {
                approach: "road_{0}_{1}_{2}".format(inter_id[0], inter_id[1], self.dic_approach_to_node[approach]) for
                approach in self.list_approachs}
            # generate all lanes
            self.list_entering_lanes = []
            for (approach, lane_number) in zip(self.list_approachs, dic_traffic_env_conf["NUM_LANES"]):
                self.list_entering_lanes += [self.dic_entering_approach_to_edge[approach] + "_" + str(i) for i in
                                             range(lane_number)]
            self.list_exiting_lanes = []
            for (approach, lane_number) in zip(self.list_approachs, dic_traffic_env_conf["NUM_LANES"]):
                self.list_exiting_lanes += [self.dic_exiting_approach_to_edge[approach] + "_" + str(i) for i in
                                            range(lane_number)]
        else:
            self.dic_entering_approach_to_edge = topology_data.get("incoming_approach_edges", {})
            self.dic_exiting_approach_to_edge = topology_data.get("outgoing_approach_edges", {})
            self.list_entering_lanes = topology_data["incoming_lanes"]
            self.list_exiting_lanes = topology_data["outgoing_lanes"]
        self.movement_outgoing_lane_groups = None if topology_data is None else topology_data.get("movement_outgoing_lane_groups")
        if self.movement_outgoing_lane_groups is None:
            self.movement_outgoing_lane_groups = self._build_legacy_movement_outgoing_lane_groups(
                len(self.list_entering_lanes)
            )

        self.list_lanes = self.list_entering_lanes + self.list_exiting_lanes

        self.adjacency_row = light_id_dict["adjacency_row"]
        self.neighbor_ENWS = light_id_dict["neighbor_ENWS"]

        # ========== record previous & current feats ==========
        self.dic_lane_vehicle_previous_step = {}
        self.dic_lane_vehicle_previous_step_in = {}
        self.dic_lane_waiting_vehicle_count_previous_step = {}
        self.dic_vehicle_speed_previous_step = {}
        self.dic_vehicle_distance_previous_step = {}

        # in [entering_lanes] out [exiting_lanes]
        self.dic_lane_vehicle_current_step_in = {}
        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        self.dic_vehicle_speed_current_step = {}
        self.dic_vehicle_distance_current_step = {}

        self.list_lane_vehicle_previous_step_in = []
        self.list_lane_vehicle_current_step_in = []

        self.dic_vehicle_arrive_leave_time = dict()  # cumulative

        self.dic_feature = {}  # this second
        self.dic_feature_previous_step = {}  # this second

        # =========== signal info set ================
        # -1: all yellow, -2: all red, -3: none
        self.all_yellow_phase_index = -1
        self.all_red_phase_index = -2

        self.current_phase_index = 1
        self.previous_phase_index = 1
        self.eng.set_tl_phase(self.inter_name, self.current_phase_index)
        path_to_log_file = os.path.join(path_to_log, "signal_inter_{0}.txt".format(self.inter_name))
        df = [self.get_current_time(), self.current_phase_index]
        df = pd.DataFrame(df)
        df = df.transpose()
        df.to_csv(path_to_log_file, mode="a", header=False, index=False)

        self.next_phase_to_set_index = None
        self.current_phase_duration = -1
        self.all_red_flag = False
        self.all_yellow_flag = False
        self.flicker = 0

    def set_signal(self, action, action_pattern, yellow_time, path_to_log):
        if self.all_yellow_flag:
            # in yellow phase
            self.flicker = 0
            if self.current_phase_duration >= yellow_time:  # yellow time reached
                self.current_phase_index = self.next_phase_to_set_index
                self.eng.set_tl_phase(self.inter_name, self.current_phase_index)  # if multi_phase, need more adjustment
                path_to_log_file = os.path.join(path_to_log, "signal_inter_{0}.txt".format(self.inter_name))
                df = [self.get_current_time(), self.current_phase_index]
                df = pd.DataFrame(df)
                df = df.transpose()
                df.to_csv(path_to_log_file, mode="a", header=False, index=False)
                self.all_yellow_flag = False
        else:
            # determine phase
            if action_pattern == "switch":  # switch by order
                if action == 0:  # keep the phase
                    self.next_phase_to_set_index = self.current_phase_index
                elif action == 1:  # change to the next phase
                    self.next_phase_to_set_index = (self.current_phase_index + 1) % len(self.list_phases)
                    # if multi_phase, need more adjustment
                else:
                    sys.exit("action not recognized\n action must be 0 or 1")

            elif action_pattern == "set":  # set to certain phase
                # self.next_phase_to_set_index = self.DIC_PHASE_MAP[action] # if multi_phase, need more adjustment
                self.next_phase_to_set_index = action + 1
            # set phase
            if self.current_phase_index == self.next_phase_to_set_index:
                # the light phase keeps unchanged
                pass
            else:  # the light phase needs to change
                # change to yellow first, and activate the counter and flag
                self.eng.set_tl_phase(self.inter_name, 0)  # !!! yellow, tmp
                path_to_log_file = os.path.join(path_to_log, "signal_inter_{0}.txt".format(self.inter_name))
                df = [self.get_current_time(), self.current_phase_index]
                df = pd.DataFrame(df)
                df = df.transpose()
                df.to_csv(path_to_log_file, mode="a", header=False, index=False)
                self.current_phase_index = self.all_yellow_phase_index
                self.all_yellow_flag = True
                self.flicker = 1

    # update inner measurements
    def update_previous_measurements(self):
        self.previous_phase_index = self.current_phase_index
        self.dic_lane_vehicle_previous_step = self.dic_lane_vehicle_current_step
        self.dic_lane_vehicle_previous_step_in = self.dic_lane_vehicle_current_step_in
        self.dic_lane_waiting_vehicle_count_previous_step = self.dic_lane_waiting_vehicle_count_current_step
        self.dic_vehicle_speed_previous_step = self.dic_vehicle_speed_current_step
        self.dic_vehicle_distance_previous_step = self.dic_vehicle_distance_current_step

    def update_current_measurements(self, simulator_state):
        def _change_lane_vehicle_dic_to_list(dic_lane_vehicle):
            list_lane_vehicle = []
            for value in dic_lane_vehicle.values():
                list_lane_vehicle.extend(value)
            return list_lane_vehicle

        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1

        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_vehicle_current_step_in = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        for lane in self.list_entering_lanes:
            self.dic_lane_vehicle_current_step_in[lane] = simulator_state["get_lane_vehicles"][lane]

        for lane in self.list_lanes:
            self.dic_lane_vehicle_current_step[lane] = simulator_state["get_lane_vehicles"][lane]
            self.dic_lane_waiting_vehicle_count_current_step[lane] = simulator_state["get_lane_waiting_vehicle_count"][lane]

        self.dic_vehicle_speed_current_step = simulator_state["get_vehicle_speed"]
        self.dic_vehicle_distance_current_step = simulator_state["get_vehicle_distance"]

        # get vehicle list
        self.list_lane_vehicle_current_step_in = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_current_step_in)
        self.list_lane_vehicle_previous_step_in = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_previous_step_in)

        list_vehicle_new_arrive = list(set(self.list_lane_vehicle_current_step_in) - set(self.list_lane_vehicle_previous_step_in))
        # can't use empty set to - real set
        if not self.list_lane_vehicle_previous_step_in:  # previous step is empty
            list_vehicle_new_left = list(set(self.list_lane_vehicle_current_step_in) -
                                         set(self.list_lane_vehicle_previous_step_in))
        else:
            list_vehicle_new_left = list(set(self.list_lane_vehicle_previous_step_in) -
                                         set(self.list_lane_vehicle_current_step_in))
        # update vehicle arrive and left time
        self._update_arrive_time(list_vehicle_new_arrive)
        self._update_left_time(list_vehicle_new_left)
        # update feature
        self._update_feature()

    def _update_leave_entering_approach_vehicle(self):
        list_entering_lane_vehicle_left = []
        # update vehicles leaving entering lane
        if not self.dic_lane_vehicle_previous_step:  # the dict is not empty
            for _ in self.list_entering_lanes:
                list_entering_lane_vehicle_left.append([])
        else:
            last_step_vehicle_id_list = []
            current_step_vehilce_id_list = []
            for lane in self.list_entering_lanes:
                last_step_vehicle_id_list.extend(self.dic_lane_vehicle_previous_step[lane])
                current_step_vehilce_id_list.extend(self.dic_lane_vehicle_current_step[lane])

            list_entering_lane_vehicle_left.append(
                list(set(last_step_vehicle_id_list) - set(current_step_vehilce_id_list))
            )
        return list_entering_lane_vehicle_left

    def _update_arrive_time(self, list_vehicle_arrive):
        ts = self.get_current_time()
        # get dic vehicle enter leave time
        for vehicle in list_vehicle_arrive:
            if vehicle not in self.dic_vehicle_arrive_leave_time:
                self.dic_vehicle_arrive_leave_time[vehicle] = {"enter_time": ts, "leave_time": np.nan}

    def _update_left_time(self, list_vehicle_left):
        ts = self.get_current_time()
        # update the time for vehicle to leave entering lane
        for vehicle in list_vehicle_left:
            try:
                self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"] = ts
            except KeyError:
                print("vehicle not recorded when entering")
                sys.exit(-1)

    def _update_feature(self):
        dic_feature = dict()
        entering_lane_count = len(self.list_entering_lanes)
        exiting_lane_count = len(self.list_exiting_lanes)
        entering_lane_samples = self.list_entering_lanes[:3]
        exiting_lane_samples = self.list_exiting_lanes[:3]

        if entering_lane_count <= 0:
            raise ValueError(
                "Invalid entering lane configuration in Intersection._update_feature: "
                f"inter_name={self.inter_name}, expected entering lanes > 0, "
                f"actual={entering_lane_count}, sample_entering_lane_ids={entering_lane_samples}"
            )
        if exiting_lane_count <= 0:
            raise ValueError(
                "Invalid exiting lane configuration in Intersection._update_feature: "
                f"inter_name={self.inter_name}, expected exiting lanes > 0, "
                f"actual={exiting_lane_count}, sample_exiting_lane_ids={exiting_lane_samples}"
            )

        dic_feature["cur_phase"] = [self.current_phase_index]
        dic_feature["time_this_phase"] = [self.current_phase_duration]
        dic_feature["lane_num_vehicle"] = self._get_lane_num_vehicle_entring()
        dic_feature["lane_num_vehicle_downstream"] = self._get_lane_num_vehicle_downstream()
        lane_num_vehicle = dic_feature["lane_num_vehicle"]
        lane_num_vehicle_downstream = dic_feature["lane_num_vehicle_downstream"]
        if len(lane_num_vehicle) != entering_lane_count:
            raise ValueError(
                "Incompatible feature vector length in Intersection._update_feature: "
                f"inter_name={self.inter_name}, feature=lane_num_vehicle, "
                f"expected={entering_lane_count}, actual={len(lane_num_vehicle)}, "
                f"sample_entering_lane_ids={entering_lane_samples}"
            )
        if len(lane_num_vehicle_downstream) != entering_lane_count:
            raise ValueError(
                "Incompatible feature vector length in Intersection._update_feature: "
                f"inter_name={self.inter_name}, feature=lane_num_vehicle_downstream, "
                f"expected={entering_lane_count}, actual={len(lane_num_vehicle_downstream)}, "
                f"sample_entering_lane_ids={entering_lane_samples}"
            )
        if len(lane_num_vehicle) != len(lane_num_vehicle_downstream):
            raise ValueError(
                "Mismatched lane counts for delta_lane_num_vehicle in Intersection._update_feature: "
                f"inter_name={self.inter_name}, expected={len(lane_num_vehicle)}, "
                f"actual={len(lane_num_vehicle_downstream)}, "
                f"sample_entering_lane_ids={entering_lane_samples}"
            )
        dic_feature["delta_lane_num_vehicle"] = [
            entering - downstream
            for entering, downstream in zip(lane_num_vehicle, lane_num_vehicle_downstream)
        ]
        dic_feature["lane_num_waiting_vehicle_in"] = self._get_lane_queue_length(self.list_entering_lanes)
        dic_feature["lane_num_waiting_vehicle_out"] = self._get_lane_queue_length(self.list_exiting_lanes)
        if len(dic_feature["lane_num_waiting_vehicle_in"]) != entering_lane_count:
            raise ValueError(
                "Incompatible feature vector length in Intersection._update_feature: "
                f"inter_name={self.inter_name}, feature=lane_num_waiting_vehicle_in, "
                f"expected={entering_lane_count}, actual={len(dic_feature['lane_num_waiting_vehicle_in'])}, "
                f"sample_entering_lane_ids={entering_lane_samples}"
            )
        if len(dic_feature["lane_num_waiting_vehicle_out"]) != exiting_lane_count:
            raise ValueError(
                "Incompatible feature vector length in Intersection._update_feature: "
                f"inter_name={self.inter_name}, feature=lane_num_waiting_vehicle_out, "
                f"expected={exiting_lane_count}, actual={len(dic_feature['lane_num_waiting_vehicle_out'])}, "
                f"sample_exiting_lane_ids={exiting_lane_samples}"
            )

        dic_feature["traffic_movement_pressure_queue"] = self._get_traffic_movement_pressure_general(
            dic_feature["lane_num_waiting_vehicle_in"], dic_feature["lane_num_waiting_vehicle_out"],
            self.movement_outgoing_lane_groups)

        dic_feature["traffic_movement_pressure_queue_efficient"] = self._get_traffic_movement_pressure_efficient(
            dic_feature["lane_num_waiting_vehicle_in"], dic_feature["lane_num_waiting_vehicle_out"],
            self.movement_outgoing_lane_groups)

        dic_feature["traffic_movement_pressure_num"] = self._get_traffic_movement_pressure_general(
            dic_feature["lane_num_vehicle"], dic_feature["lane_num_vehicle_downstream"],
            self.movement_outgoing_lane_groups)

        tmp_part_n, tmp_part_q, tmp_efficient_part, enter_running_part, lepq = self._get_part_traffic_movement_features()

        dic_feature["lane_enter_running_part"] = list(enter_running_part)

        dic_feature["pressure"] = self._get_pressure()
        dic_feature["adjacency_matrix"] = self._get_adjacency_row()

        dic_feature["num_in_seg_attend"] = self._orgnize_several_segments_attend(dic_feature["lane_num_waiting_vehicle_in"],
                                                                                 dic_feature["lane_num_waiting_vehicle_out"])
        self.dic_feature = dic_feature

    def _orgnize_several_segments_attend(self, queue_in, queue_out):
        part1, part2, part3 = self._get_several_segments_attend(lane_vehicles=self.dic_lane_vehicle_current_step,
                                                                vehicle_distance=self.dic_vehicle_distance_current_step,
                                                                vehicle_speed=self.dic_vehicle_speed_current_step,
                                                                lane_length=self.lane_length,
                                                                list_lanes=self.list_lanes)
        run_in_part1 = [float(len(part1[lane])) for lane in self.list_entering_lanes]
        run_in_part2 = [float(len(part2[lane])) for lane in self.list_entering_lanes]
        run_in_part3 = [float(len(part3[lane])) for lane in self.list_entering_lanes]

        run_out_part1 = [float(len(part1[lane])) for lane in self.list_exiting_lanes]
        run_out_part2 = [float(len(part2[lane]))for lane in self.list_exiting_lanes]
        run_out_part3 = [float(len(part3[lane])) for lane in self.list_exiting_lanes]

        total_in, total_out = [], []
        if not (len(run_in_part1) == len(run_in_part2) == len(run_in_part3) == len(queue_in)):
            raise ValueError(
                "Unexpected entering movement vector lengths in _orgnize_several_segments_attend: "
                f"{len(run_in_part1)}, {len(run_in_part2)}, {len(run_in_part3)}, {len(queue_in)}"
            )
        if not (len(run_out_part1) == len(run_out_part2) == len(run_out_part3) == len(queue_out)):
            raise ValueError(
                "Unexpected exiting movement vector lengths in _orgnize_several_segments_attend: "
                f"{len(run_out_part1)}, {len(run_out_part2)}, {len(run_out_part3)}, {len(queue_out)}"
            )

        for p1, p2, p3, q in zip(run_in_part1, run_in_part2, run_in_part3, queue_in):
            total_in.extend([p1, p2, p3, q])
        for p1, p2, p3, q in zip(run_out_part1, run_out_part2, run_out_part3, queue_out):
            total_out.extend([p1, p2, p3, q])
        return total_in + total_out

    def _get_several_segments_attend(self, lane_vehicles, vehicle_distance, vehicle_speed,
                                           lane_length, list_lanes):
        obs_length = 100
        part1, part2, part3 = {}, {}, {}
        for lane in list_lanes:
            part1[lane], part2[lane], part3[lane] = [], [], []
            for vehicle in lane_vehicles[lane]:
                # set as num_vehicle
                if "shadow" in vehicle:  # remove the shadow
                    vehicle = vehicle[:-7]
                    continue
                if vehicle_speed[vehicle] > 0.1:
                    temp_v_distance = vehicle_distance[vehicle]
                    if temp_v_distance > lane_length[lane] - obs_length:
                        part1[lane].append(vehicle)
                    elif lane_length[lane] - 2 * obs_length < temp_v_distance <= lane_length[lane] - obs_length:
                        part2[lane].append(vehicle)
                    elif lane_length[lane] - 3 * obs_length < temp_v_distance <= lane_length[lane] - 2 * obs_length:
                        part3[lane].append(vehicle)
        return part1, part2, part3

    @staticmethod
    def _build_legacy_movement_outgoing_lane_groups(num_movements):
        if num_movements == 12:
            return [3, 0, 2, 2, 1, 3, 0, 2, 1, 1, 3, 0]
        if num_movements % 4 != 0:
            raise ValueError("Traffic movement vectors must be divisible by 4 approaches, got {0}".format(num_movements))
        lanes_per_approach = num_movements // 4
        movement_outgoing = []
        # legacy order follows approaches W,E,N,S and lane indices 0,1,2 => left,through,right
        turn_to_approach_index = [3, 0, 2]
        for _ in range(4):
            for lane_idx in range(lanes_per_approach):
                movement_outgoing.append(turn_to_approach_index[lane_idx % len(turn_to_approach_index)])
        return movement_outgoing

    @staticmethod
    def _get_traffic_movement_pressure_general(enterings, exitings, movement_outgoing_lane_groups):
        """
            Created by LiangZhang
            Calculate pressure with entering and exiting vehicles
            only for 3 x 3 lanes intersection
        """
        if len(enterings) != len(exitings):
            min_len = min(len(enterings), len(exitings), len(movement_outgoing_lane_groups))
            enterings = enterings[:min_len]
            exitings = exitings[:min_len]
            movement_outgoing_lane_groups = movement_outgoing_lane_groups[:min_len]
        if len(enterings) == 0:
            return []
        if len(enterings) % 4 != 0:
            avg_out = float(sum(exitings)) / float(len(exitings)) if len(exitings) > 0 else 0.0
            return [entering - avg_out for entering in enterings]
        lanes_per_approach = len(enterings) // 4
        out_sums = [sum(exitings[i * lanes_per_approach:(i + 1) * lanes_per_approach]) for i in range(4)]
        movement_pressures = []
        for entering, group_indices in zip(enterings, movement_outgoing_lane_groups):
            mapped_groups = group_indices if isinstance(group_indices, (list, tuple, set)) else [group_indices]
            valid_groups = [group_idx for group_idx in mapped_groups if group_idx is not None]
            outgoing_pressure = sum(out_sums[group_idx] for group_idx in valid_groups)
            movement_pressures.append(entering - outgoing_pressure)
        return movement_pressures

    @staticmethod
    def _get_traffic_movement_pressure_efficient(enterings, exitings, movement_outgoing_lane_groups):
        """
            Created by LiangZhang
            Calculate pressure with entering and exiting vehicles
            only for 3 x 3 lanes intersection
        """
        if len(enterings) != len(exitings):
            min_len = min(len(enterings), len(exitings), len(movement_outgoing_lane_groups))
            enterings = enterings[:min_len]
            exitings = exitings[:min_len]
            movement_outgoing_lane_groups = movement_outgoing_lane_groups[:min_len]
        if len(enterings) == 0:
            return []
        if len(enterings) % 4 != 0:
            avg_out = float(sum(exitings)) / float(len(exitings)) if len(exitings) > 0 else 0.0
            return [entering - avg_out for entering in enterings]
        lanes_per_approach = len(enterings) // 4
        out_sums = [sum(exitings[i * lanes_per_approach:(i + 1) * lanes_per_approach]) for i in range(4)]
        movement_pressures = []
        for entering, group_indices in zip(enterings, movement_outgoing_lane_groups):
            mapped_groups = group_indices if isinstance(group_indices, (list, tuple, set)) else [group_indices]
            valid_groups = [group_idx for group_idx in mapped_groups if group_idx is not None]
            outgoing_pressure = sum(out_sums[group_idx] for group_idx in valid_groups)
            movement_pressures.append(entering - outgoing_pressure / max(lanes_per_approach, 1))
        return movement_pressures

    def _get_part_traffic_movement_features(self):
        """
        return: part_traffic_movement_pressure_num:     both the end and the beginning of the lane
                part_patrric_movement_pressure_queue:   all at the end of the road
                part_entering_running_vehicles:         part obs of the running vehicles
        """
        f_p_num, l_p_num, l_p_q = self._get_part_observations(lane_vehicles=self.dic_lane_vehicle_current_step,
                                                              vehicle_distance=self.dic_vehicle_distance_current_step,
                                                              vehicle_speed=self.dic_vehicle_speed_current_step,
                                                              lane_length=self.lane_length,
                                                              obs_length=self.obs_length,
                                                              list_lanes=self.list_lanes)
        """calculate traffic_movement_pressure with part queue"""
        list_entering_part_queue = [len(l_p_q[lane]) for lane in self.list_entering_lanes]
        list_exiting_part_queue = [len(l_p_q[lane]) for lane in self.list_exiting_lanes]
        tmp_queue_efficient_part = self._get_traffic_movement_pressure_efficient(list_entering_part_queue,
                                                                                 list_exiting_part_queue,
                                                                                 self.movement_outgoing_lane_groups)
        tmp_queue_part = self._get_traffic_movement_pressure_general(list_entering_part_queue,
                                                                     list_exiting_part_queue,
                                                                     self.movement_outgoing_lane_groups)

        """calculate traffic_movement_pressure with part num vehicle"""
        # entering
        list_entering_num_f = [len(f_p_num[lane]) for lane in self.list_entering_lanes]
        list_entering_num_l = [len(l_p_num[lane]) for lane in self.list_entering_lanes]
        entering_num = np.array(list_entering_num_f) + np.array(list_entering_num_l)
        # exiting
        list_exiting_num_f = [len(f_p_num[lane]) for lane in self.list_exiting_lanes]
        list_exiting_num_l = [len(l_p_num[lane]) for lane in self.list_exiting_lanes]
        exiting_num = np.array(list_exiting_num_f) + np.array(list_exiting_num_l)
        traffic_movement_pressure_nums = self._get_traffic_movement_pressure_general(
            entering_num, exiting_num, self.movement_outgoing_lane_groups
        )
        # part of entering running vehicles, all at the end of the road
        part_entering_running = np.array(list_entering_num_l) - np.array(list_entering_part_queue)

        return traffic_movement_pressure_nums, tmp_queue_part, tmp_queue_efficient_part, part_entering_running, list_entering_part_queue

    @staticmethod
    def _get_part_observations(lane_vehicles, vehicle_distance, vehicle_speed,
                               lane_length, obs_length, list_lanes):
        """
            Input: lane_vehicles :      Dict{lane_id    :   [vehicle_ids]}
                   vehicle_distance:    Dict{vehicle_id :   float(dist)}
                   vehicle_speed:       Dict{vehicle_id :   float(speed)}
                   lane_length  :       Dict{lane_id    :   float(length)}
                   obs_length   :       The part observation length
                   list_lanes   :       List[lane_ids at the intersection]
        :return:
                    part_vehicles:      Dict{ lane_id, [vehicle_ids]}
        """
        # get vehicle_ids and speeds
        first_part_num_vehicle = {}
        first_part_queue_vehicle = {}  # useless, at the begin of lane, there is no waiting vechiles
        last_part_num_vehicle = {}
        last_part_queue_vehicle = {}

        for lane in list_lanes:
            first_part_num_vehicle[lane] = []
            first_part_queue_vehicle[lane] = []
            last_part_num_vehicle[lane] = []
            last_part_queue_vehicle[lane] = []
            last_part_obs_length = lane_length[lane] - obs_length
            for vehicle in lane_vehicles[lane]:
                """ get the first part of obs
                    That is vehicle_distance <= obs_length 
                """
                # set as num_vehicle
                if "shadow" in vehicle:  # remove the shadow
                    vehicle = vehicle[:-7]
                temp_v_distance = vehicle_distance[vehicle]
                if temp_v_distance <= obs_length:
                    first_part_num_vehicle[lane].append(vehicle)
                    # analyse if waiting
                    if vehicle_speed[vehicle] <= 0.1:
                        first_part_queue_vehicle[lane].append(vehicle)

                """ get the last part of obs
                    That is  lane_length-obs_length <= vehicle_distance <= lane_length 
                """
                if temp_v_distance >= last_part_obs_length:
                    last_part_num_vehicle[lane].append(vehicle)
                    # analyse if waiting
                    if vehicle_speed[vehicle] <= 0.1:
                        last_part_queue_vehicle[lane].append(vehicle)

        return first_part_num_vehicle, last_part_num_vehicle, last_part_queue_vehicle

    def _get_pressure(self):
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in self.list_entering_lanes] + \
               [-self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in self.list_exiting_lanes]

    def _get_lane_queue_length(self, list_lanes):
        """
        queue length for each lane
        """
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in list_lanes]

    def _get_lane_num_vehicle_entring(self):
        """
        vehicle number for each lane
        """
        return [len(self.dic_lane_vehicle_current_step[lane]) for lane in self.list_entering_lanes]

    def _get_lane_num_vehicle_downstream(self):
        """Return downstream counts aligned to entering movements."""
        exiting_counts = [len(self.dic_lane_vehicle_current_step[lane]) for lane in self.list_exiting_lanes]
        if len(exiting_counts) == len(self.list_entering_lanes):
            return exiting_counts

        if not exiting_counts:
            return [0 for _ in self.list_entering_lanes]

        downstream = []
        for group_indices in self.movement_outgoing_lane_groups:
            if not group_indices:
                downstream.append(0)
                continue
            valid = [idx for idx in group_indices if 0 <= idx < len(exiting_counts)]
            if not valid:
                downstream.append(0)
            else:
                downstream.append(float(sum(exiting_counts[idx] for idx in valid)) / len(valid))
        return downstream

    # ================= get functions from outside ======================
    def get_current_time(self):
        return self.eng.get_current_time()

    def get_dic_vehicle_arrive_leave_time(self):
        return self.dic_vehicle_arrive_leave_time

    def get_feature(self):
        return self.dic_feature

    def get_state(self, list_state_features):
        dic_state = {state_feature_name: self.dic_feature[state_feature_name] for
                     state_feature_name in list_state_features}
        return dic_state

    def _get_adjacency_row(self):
        return self.adjacency_row

    def get_reward(self, dic_reward_info):
        dic_reward = dict()
        # dic_reward["sum_lane_queue_length"] = None
        dic_reward["pressure"] = np.absolute(np.sum(self.dic_feature["pressure"]))
        dic_reward["queue_length"] = np.absolute(np.sum(self.dic_feature["lane_num_waiting_vehicle_in"]))
        reward = 0
        for r in dic_reward_info:
            if dic_reward_info[r] != 0:
                reward += dic_reward_info[r] * dic_reward[r]
        return reward


class CityFlowEnv:

    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf):
        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf

        self.current_time = None
        self.id_to_index = None
        self.traffic_light_node_dict = None
        self.eng = None
        self.list_intersection = None
        self.list_inter_log = None
        self.list_lanes = None
        self.system_states = None
        self.lane_length = None

        # check min action time
        if self.dic_traffic_env_conf["MIN_ACTION_TIME"] <= self.dic_traffic_env_conf["YELLOW_TIME"]:
            """ include the yellow time in action time """
            print("MIN_ACTION_TIME should include YELLOW_TIME")
            sys.exit()

        # touch new inter_{}.pkl (if exists, remove)
        for inter_ind in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            f.close()

    def reset(self):
        print(" ============= self.eng.reset() to be implemented ==========")
        cityflow_config = {
            "interval": self.dic_traffic_env_conf["INTERVAL"],
            "seed": 0,
            "laneChange": True,
            "dir": self.path_to_work_directory+"/",
            "roadnetFile": self.dic_traffic_env_conf["ROADNET_FILE"],
            "flowFile": self.dic_traffic_env_conf["TRAFFIC_FILE"],
            "rlTrafficLight": True,
            "saveReplay": False,
            "roadnetLogFile": "frontend/web/roadnetLogFile.json",
            "replayLogFile": "frontend/web/replayLogFile.txt"
        }
        # print(cityflow_config)
        with open(os.path.join(self.path_to_work_directory, "cityflow.config"), "w") as json_file:
            json.dump(cityflow_config, json_file)

        self.eng = engine.Engine(os.path.join(self.path_to_work_directory, "cityflow.config"), thread_num=1)

        # topology adapter (legacy default)
        topology_mode = self.dic_traffic_env_conf.get("TOPOLOGY_ADAPTER_MODE", "legacy_grid")
        self.topology_adapter = TopologyAdapter(
            mode=topology_mode,
            path_to_work_directory=self.path_to_work_directory,
            roadnet_file=self.dic_traffic_env_conf["ROADNET_FILE"],
            dic_traffic_env_conf=self.dic_traffic_env_conf
        )
        self.topology_artifacts = self.topology_adapter.build()

        # get adjacency
        self.traffic_light_node_dict = self.topology_artifacts["traffic_light_node_dict"]

        # get lane length
        _, self.lane_length = self.get_lane_length()

        # initialize intersections
        if topology_mode == "general_topology":
            all_topology_items = list(self.topology_artifacts["intersection_topology_data"].items())
            max_inters = int(self.dic_traffic_env_conf.get("NUM_INTERSECTIONS", len(all_topology_items)))
            selected_topology_items = all_topology_items[:max_inters]
            self.list_intersection = [
                Intersection(
                    inter_id=topo_data["legacy_inter_id"],
                    dic_traffic_env_conf=self.dic_traffic_env_conf,
                    eng=self.eng,
                    light_id_dict=self.traffic_light_node_dict[inter_id],
                    path_to_log=self.path_to_log,
                    lanes_length_dict=self.lane_length,
                    topology_data=topo_data
                )
                for inter_id, topo_data in selected_topology_items
            ]
        else:
            self.list_intersection = [Intersection((i+1, j+1), self.dic_traffic_env_conf, self.eng,
                                                   self.traffic_light_node_dict["intersection_{0}_{1}".format(i+1, j+1)],
                                                   self.path_to_log,
                                                   self.lane_length)
                                      for i in range(self.dic_traffic_env_conf["NUM_COL"])
                                      for j in range(self.dic_traffic_env_conf["NUM_ROW"])]
        self.list_inter_log = [[] for _ in range(len(self.list_intersection))]

        self.id_to_index = self.topology_artifacts["engine_tl_id_to_index"]

        self.list_lanes = []
        for inter in self.list_intersection:
            self.list_lanes += inter.list_lanes
        self.list_lanes = np.unique(self.list_lanes).tolist()

        # get new measurements
        self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_vehicle_speed": self.eng.get_vehicle_speed(),
                              "get_vehicle_distance": self.eng.get_vehicle_distance(),
                              }

        self._assert_lane_state_consistency(self.system_states["get_lane_vehicles"])

        for inter in self.list_intersection:
            inter.update_current_measurements(self.system_states)
        state, done = self.get_state()
        return state

    def step(self, action):

        step_start_time = time.time()

        list_action_in_sec = [action]
        list_action_in_sec_display = [action]
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]-1):
            if self.dic_traffic_env_conf["ACTION_PATTERN"] == "switch":
                list_action_in_sec.append(np.zeros_like(action).tolist())
            elif self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
                list_action_in_sec.append(np.copy(action).tolist())
            list_action_in_sec_display.append(np.full_like(action, fill_value=-1).tolist())

        average_reward_action_list = [0]*len(action)
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]):

            action_in_sec = list_action_in_sec[i]
            action_in_sec_display = list_action_in_sec_display[i]

            instant_time = self.get_current_time()
            self.current_time = self.get_current_time()

            before_action_feature = self.get_feature()
            # state = self.get_state()

            if i == 0:
                print("time: {0}".format(instant_time))
                    
            self._inner_step(action_in_sec)

            # get reward
            reward = self.get_reward()
            for j in range(len(reward)):
                average_reward_action_list[j] = (average_reward_action_list[j] * i + reward[j]) / (i + 1)
            self.log(cur_time=instant_time, before_action_feature=before_action_feature, action=action_in_sec_display)
            next_state, done = self.get_state()

        print("Step time: ", time.time() - step_start_time)
        return next_state, reward, done, average_reward_action_list

    def _inner_step(self, action):
        # copy current measurements to previous measurements
        for inter in self.list_intersection:
            inter.update_previous_measurements()
        # set signals
        # multi_intersection decided by action {inter_id: phase}
        for inter_ind, inter in enumerate(self.list_intersection):
            inter.set_signal(
                action=action[inter_ind],
                action_pattern=self.dic_traffic_env_conf["ACTION_PATTERN"],
                yellow_time=self.dic_traffic_env_conf["YELLOW_TIME"],
                path_to_log=self.path_to_log
            )

        # run one step
        for i in range(int(1/self.dic_traffic_env_conf["INTERVAL"])):
            self.eng.next_step()

        self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_vehicle_speed": self.eng.get_vehicle_speed(),
                              "get_vehicle_distance": self.eng.get_vehicle_distance()
                              }

        for inter in self.list_intersection:
            inter.update_current_measurements(self.system_states)

    def _assert_lane_state_consistency(self, lane_vehicles):
        missing_lanes = sorted(set(lane_vehicles.keys()) - set(self.lane_length.keys()))
        if missing_lanes:
            raise KeyError(
                "Lane consistency check failed: lanes from engine.get_lane_vehicles() "
                "missing in lane_length: {0}".format(missing_lanes)
            )

    def get_feature(self):
        list_feature = [inter.get_feature() for inter in self.list_intersection]
        return list_feature

    def get_state(self):
        list_state = [inter.get_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"]) for inter in self.list_intersection]
        done = False
        return list_state, done

    def get_reward(self):
        list_reward = [inter.get_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"]) for inter in self.list_intersection]
        return list_reward

    def get_current_time(self):
        return self.eng.get_current_time()

    def log(self, cur_time, before_action_feature, action):

        for inter_ind in range(len(self.list_intersection)):
            self.list_inter_log[inter_ind].append({"time": cur_time,
                                                   "state": before_action_feature[inter_ind],
                                                   "action": action[inter_ind]})

    def batch_log_2(self):
        """
        Used for model test, only log the vehicle_inter_.csv
        """
        for inter_ind in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            # changed from origin
            if int(inter_ind) % 100 == 0:
                print("Batch log for inter ", inter_ind)
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = pd.DataFrame.from_dict(dic_vehicle, orient="index")
            df.to_csv(path_to_log_file, na_rep="nan")

    def batch_log(self, start, stop):
        for inter_ind in range(start, stop):
            # changed from origin
            if int(inter_ind) % 100 == 0:
                print("Batch log for inter ", inter_ind)
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = pd.DataFrame.from_dict(dic_vehicle, orient="index")
            df.to_csv(path_to_log_file, na_rep="nan")
            
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_ind], f)
            f.close()

    def bulk_log_multi_process(self, batch_size=100):
        assert len(self.list_intersection) == len(self.list_inter_log)
        if batch_size > len(self.list_intersection):
            batch_size_run = len(self.list_intersection)
        else:
            batch_size_run = batch_size
        process_list = []
        for batch in range(0, len(self.list_intersection), batch_size_run):
            start = batch
            stop = min(batch + batch_size, len(self.list_intersection))
            p = Process(target=self.batch_log, args=(start, stop))
            print("before")
            p.start()
            print("end")
            process_list.append(p)
        print("before join")

        for t in process_list:
            t.join()
        print("end join")

    def _adjacency_extraction(self):
        traffic_light_node_dict = {}
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])
        with open("{0}".format(file)) as json_data:
            net = json.load(json_data)

        controlled_intersections = sorted(
            [inter for inter in net["intersections"] if not inter["virtual"]],
            key=lambda inter: inter["id"]
        )
        controlled_intersection_ids = [inter["id"] for inter in controlled_intersections]

        inter_id_to_index = {inter_id: index for index, inter_id in enumerate(controlled_intersection_ids)}
        total_inter_num = len(controlled_intersection_ids)
        top_k = self.dic_traffic_env_conf["TOP_K_ADJACENCY"]

        checksum = 0
        for idx, inter_id in enumerate(controlled_intersection_ids):
            checksum = (checksum + (idx + 1) * sum(ord(ch) for ch in inter_id)) % 1000000007
        print("[GeneralTopology] Controlled intersections: {0}, deterministic checksum: {1}".format(
            total_inter_num, checksum
        ))

        for inter in controlled_intersections:
            traffic_light_node_dict[inter["id"]] = {
                "location": {"x": float(inter["point"]["x"]), "y": float(inter["point"]["y"])},
                "total_inter_num": inter_id_to_index,
                "adjacency_row": None,
                "inter_id_to_index": inter_id_to_index,
                "neighbor_ENWS": None
            }

        edge_id_dict = {
            road["id"]: {"from": road["startIntersection"], "to": road["endIntersection"]}
            for road in net["roads"]
        }

        for inter_id in controlled_intersection_ids:
            location_1 = traffic_light_node_dict[inter_id]["location"]
            neighbor_distances = []
            for other_id in controlled_intersection_ids:
                if other_id == inter_id:
                    continue
                location_2 = traffic_light_node_dict[other_id]["location"]
                neighbor_distances.append((self._cal_distance(location_1, location_2), inter_id_to_index[other_id]))

            neighbor_distances.sort(key=lambda item: (item[0], item[1]))
            adjacency_neighbors = [idx for _, idx in neighbor_distances[:max(top_k - 1, 0)]]
            traffic_light_node_dict[inter_id]["adjacency_row"] = [inter_id_to_index[inter_id]] + adjacency_neighbors

        for inter_id in controlled_intersection_ids:
            traffic_light_node_dict[inter_id]["neighbor_ENWS"] = []
            for j in range(4):
                road_id = inter_id.replace("intersection", "road") + "_" + str(j)
                road_data = edge_id_dict.get(road_id)
                if road_data is None:
                    traffic_light_node_dict[inter_id]["neighbor_ENWS"].append(None)
                    continue
                neighbor_id = road_data["to"]
                if neighbor_id not in traffic_light_node_dict:
                    traffic_light_node_dict[inter_id]["neighbor_ENWS"].append(None)
                else:
                    traffic_light_node_dict[inter_id]["neighbor_ENWS"].append(neighbor_id)

        return traffic_light_node_dict

    @staticmethod
    def _cal_distance(loc_dict1, loc_dict2):
        a = np.array((loc_dict1["x"], loc_dict1["y"]))
        b = np.array((loc_dict2["x"], loc_dict2["y"]))
        return np.sqrt(np.sum((a-b)**2))

    @staticmethod
    def end_cityflow():
        print("============== cityflow process end ===============")

    def get_lane_length(self):
        """
        newly added part for get lane length
        Read the road net file
        Return: dict{lanes} normalized with the min lane length
        """
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])
        with open(file) as json_data:
            net = json.load(json_data)
        roads = net['roads']
        lanes_length_dict = {}
        lane_normalize_factor = {}

        for road in roads:
            points = road["points"]
            road_length = abs(points[0]['x'] + points[0]['y'] - points[1]['x'] - points[1]['y'])
            for lane_idx, _ in enumerate(road["lanes"]):
                lane_id = f"{road['id']}_{lane_idx}"
                lanes_length_dict[lane_id] = road_length
        min_length = min(lanes_length_dict.values())

        for key, value in lanes_length_dict.items():
            lane_normalize_factor[key] = value / min_length
        return lane_normalize_factor, lanes_length_dict


class TopologyAdapter:
    def __init__(self, mode, path_to_work_directory, roadnet_file, dic_traffic_env_conf):
        self.mode = mode
        self.path_to_work_directory = path_to_work_directory
        self.roadnet_file = roadnet_file
        self.dic_traffic_env_conf = dic_traffic_env_conf

    def build(self):
        if self.mode == "legacy_grid":
            return self._build_legacy_grid()
        if self.mode == "general_topology":
            return self._build_general_topology()
        raise ValueError("Unknown TOPOLOGY_ADAPTER_MODE: {0}".format(self.mode))

    def _build_legacy_grid(self):
        env = self._build_base_adjacency()
        inter_topology = {}
        for i in range(self.dic_traffic_env_conf["NUM_COL"]):
            for j in range(self.dic_traffic_env_conf["NUM_ROW"]):
                inter_name = "intersection_{0}_{1}".format(i + 1, j + 1)
                inter_topology[inter_name] = {
                    "inter_name": inter_name,
                    "legacy_inter_id": (i + 1, j + 1),
                }
        env["intersection_topology_data"] = inter_topology
        return env

    @staticmethod
    def _lane_ids_for_road(road):
        return ["{0}_{1}".format(road["id"], lane_idx) for lane_idx, _ in enumerate(road["lanes"])]

    def _build_general_topology(self):
        env = self._build_base_adjacency()
        roadnet_path = os.path.join(self.path_to_work_directory, self.roadnet_file)
        with open(roadnet_path) as f:
            net = json.load(f)
        roads = {r["id"]: r for r in net["roads"]}
        intersections = [i for i in net["intersections"] if not i["virtual"]]
        idx = 1
        inter_topology = {}
        for inter in intersections:
            inter_id = inter["id"]
            incoming_lanes, outgoing_lanes = [], []
            for road_id in inter["roads"]:
                road = roads[road_id]
                lane_ids = self._lane_ids_for_road(road)
                if road["endIntersection"] == inter_id:
                    incoming_lanes.extend(lane_ids)
                if road["startIntersection"] == inter_id:
                    outgoing_lanes.extend(lane_ids)
            outgoing_lane_to_approach_idx = {}
            if len(outgoing_lanes) % 4 == 0 and len(outgoing_lanes) > 0:
                lanes_per_approach = len(outgoing_lanes) // 4
                for approach_idx in range(4):
                    for lane_id in outgoing_lanes[approach_idx * lanes_per_approach:(approach_idx + 1) * lanes_per_approach]:
                        outgoing_lane_to_approach_idx[lane_id] = approach_idx

            movement_outgoing_lane_groups = [[] for _ in incoming_lanes]
            incoming_lane_to_movement_idx = {lane_id: idx for idx, lane_id in enumerate(incoming_lanes)}
            for road_link in inter.get("roadLinks", []):
                start_road = road_link.get("startRoad")
                end_road = road_link.get("endRoad")
                for lane_link in road_link.get("laneLinks", []):
                    start_lane = "{0}_{1}".format(start_road, lane_link["startLaneIndex"])
                    movement_idx = incoming_lane_to_movement_idx.get(start_lane)
                    if movement_idx is None:
                        continue
                    end_lane = "{0}_{1}".format(end_road, lane_link["endLaneIndex"])
                    mapped_group = outgoing_lane_to_approach_idx.get(end_lane)
                    if mapped_group is not None and mapped_group not in movement_outgoing_lane_groups[movement_idx]:
                        movement_outgoing_lane_groups[movement_idx].append(mapped_group)

            inter_topology[inter_id] = {
                "inter_name": inter_id,
                "legacy_inter_id": (idx, 1),
                "incoming_lanes": incoming_lanes,
                "outgoing_lanes": outgoing_lanes,
                "movement_outgoing_lane_groups": movement_outgoing_lane_groups
            }
            idx += 1
        env["intersection_topology_data"] = inter_topology
        return env

    def _build_base_adjacency(self):
        traffic_light_node_dict = {}
        file = os.path.join(self.path_to_work_directory, self.roadnet_file)
        with open(file) as json_data:
            net = json.load(json_data)
        for inter in net["intersections"]:
            if not inter["virtual"]:
                traffic_light_node_dict[inter["id"]] = {"location": {"x": float(inter["point"]["x"]),
                                                                     "y": float(inter["point"]["y"])},
                                                        "total_inter_num": None, "adjacency_row": None,
                                                        "inter_id_to_index": None,
                                                        "neighbor_ENWS": None}

        top_k = self.dic_traffic_env_conf["TOP_K_ADJACENCY"]
        total_inter_num = len(traffic_light_node_dict.keys())
        inter_id_to_index = {inter_id: idx for idx, inter_id in enumerate(traffic_light_node_dict.keys())}

        edge_id_dict = {}
        for road in net["roads"]:
            edge_id_dict[road["id"]] = {"from": road["startIntersection"], "to": road["endIntersection"]}

        for i in traffic_light_node_dict.keys():
            location_1 = traffic_light_node_dict[i]["location"]
            row = np.array([0] * total_inter_num)
            for j in traffic_light_node_dict.keys():
                location_2 = traffic_light_node_dict[j]["location"]
                row[inter_id_to_index[j]] = CityFlowEnv._cal_distance(location_1, location_2)
            if len(row) == top_k:
                adjacency_row_unsorted = np.argpartition(row, -1)[:top_k].tolist()
            elif len(row) > top_k:
                adjacency_row_unsorted = np.argpartition(row, top_k)[:top_k].tolist()
            else:
                adjacency_row_unsorted = [k for k in range(total_inter_num)]
            adjacency_row_unsorted.remove(inter_id_to_index[i])
            traffic_light_node_dict[i]["adjacency_row"] = [inter_id_to_index[i]] + adjacency_row_unsorted
            traffic_light_node_dict[i]["total_inter_num"] = total_inter_num

        for i in traffic_light_node_dict.keys():
            traffic_light_node_dict[i]["total_inter_num"] = inter_id_to_index
            traffic_light_node_dict[i]["neighbor_ENWS"] = []
            for j in range(4):
                road_id = i.replace("intersection", "road") + "_" + str(j)
                if road_id not in edge_id_dict or edge_id_dict[road_id]["to"] not in traffic_light_node_dict.keys():
                    traffic_light_node_dict[i]["neighbor_ENWS"].append(None)
                else:
                    traffic_light_node_dict[i]["neighbor_ENWS"].append(edge_id_dict[road_id]["to"])

        return {
            "controllable_intersection_ids": list(traffic_light_node_dict.keys()),
            "traffic_light_node_dict": traffic_light_node_dict,
            "engine_tl_id_to_index": inter_id_to_index,
        }
