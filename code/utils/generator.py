from .config import DIC_AGENTS
from .cityflow_env import CityFlowEnv
import time
import os
import copy
import json


MAPPING_FILE_NAME = "intersection_index_map.json"


def _build_intersection_index_map(path_to_work_directory, roadnet_file):
    roadnet_path = os.path.join(path_to_work_directory, roadnet_file)
    with open(roadnet_path, "r") as f:
        roadnet = json.load(f)
    intersections = [inter["id"] for inter in roadnet["intersections"] if not inter["virtual"]]
    return {
        str(idx): inter_id for idx, inter_id in enumerate(intersections)
    }


class Generator:
    def __init__(self, cnt_round, cnt_gen, dic_path, dic_agent_conf, dic_traffic_env_conf):

        self.cnt_round = cnt_round
        self.cnt_gen = cnt_gen
        self.dic_path = dic_path
        self.dic_agent_conf = copy.deepcopy(dic_agent_conf)
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.intersection_index_map = _build_intersection_index_map(
            self.dic_path["PATH_TO_WORK_DIRECTORY"],
            self.dic_traffic_env_conf["ROADNET_FILE"]
        )
        if len(self.intersection_index_map) != self.dic_traffic_env_conf["NUM_AGENTS"]:
            raise ValueError(
                "Intersection count mismatch: roadnet has {0} controllable intersections, "
                "but NUM_AGENTS={1}."
                .format(len(self.intersection_index_map), self.dic_traffic_env_conf["NUM_AGENTS"])
            )
        map_path = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], MAPPING_FILE_NAME)
        with open(map_path, "w") as f:
            json.dump(self.intersection_index_map, f, indent=2, sort_keys=True)
        self.agents = [None]*dic_traffic_env_conf['NUM_AGENTS']
        self.path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                        "round_"+str(self.cnt_round), "generator_"+str(self.cnt_gen))
        if not os.path.exists(self.path_to_log):
            os.makedirs(self.path_to_log)
            start_time = time.time()
            for i in range(dic_traffic_env_conf['NUM_AGENTS']):
                agent_name = self.dic_traffic_env_conf["MODEL_NAME"]
                agent = DIC_AGENTS[agent_name](
                    dic_agent_conf=self.dic_agent_conf,
                    dic_traffic_env_conf=self.dic_traffic_env_conf,
                    dic_path=self.dic_path,
                    cnt_round=self.cnt_round,
                    intersection_id=str(i)
                )
                self.agents[i] = agent
            print("Create intersection agent time: ", time.time()-start_time)

        self.env = CityFlowEnv(
            path_to_log=self.path_to_log,
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf
        )

    def generate(self):

        reset_env_start_time = time.time()
        done = False
        state = self.env.reset()
        step_num = 0
        reset_env_time = time.time() - reset_env_start_time
        running_start_time = time.time()
        while not done and step_num < int(self.dic_traffic_env_conf["RUN_COUNTS"] /
                                          self.dic_traffic_env_conf["MIN_ACTION_TIME"]):
            action_list = []
            step_start_time = time.time()
            for i in range(self.dic_traffic_env_conf["NUM_AGENTS"]):

                if self.dic_traffic_env_conf["MODEL_NAME"] in ["EfficientPressLight", "EfficientColight",
                                                               "EfficientMPLight", "Attend",
                                                               "AdvancedMPLight", "AdvancedColight", "AdvancedDQN"]:
                    one_state = state
                    action = self.agents[i].choose_action(step_num, one_state)
                    action_list = action
                else:
                    one_state = state[i]
                    action = self.agents[i].choose_action(step_num, one_state)
                    action_list.append(action)

            next_state, reward, done, _ = self.env.step(action_list)

            print("time: {0}, running_time: {1}".format(self.env.get_current_time() -
                                                        self.dic_traffic_env_conf["MIN_ACTION_TIME"],
                                                        time.time()-step_start_time))

            state = next_state
            step_num += 1
        running_time = time.time() - running_start_time
        log_start_time = time.time()
        print("start logging.......................")
        self.env.bulk_log_multi_process()
        log_time = time.time() - log_start_time
        self.env.end_cityflow()
        print("reset_env_time: ", reset_env_time)
        print("running_time: ", running_time)
        print("log_time: ", log_time)
