from .config import DIC_AGENTS
import pickle
import os
import time
import traceback
import random
import json


MAPPING_FILE_NAME = "intersection_index_map.json"


def _build_intersection_index_map(path_to_work_directory, roadnet_file):
    roadnet_path = os.path.join(path_to_work_directory, roadnet_file)
    with open(roadnet_path, "r") as f:
        roadnet = json.load(f)
    intersections = [inter["id"] for inter in roadnet["intersections"] if not inter["virtual"]]
    return {str(idx): inter_id for idx, inter_id in enumerate(intersections)}


def _load_and_validate_intersection_index_map(dic_path, dic_traffic_env_conf):
    map_path = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], MAPPING_FILE_NAME)
    if not os.path.exists(map_path):
        return None
    with open(map_path, "r") as f:
        stored_map = json.load(f)
    current_map = _build_intersection_index_map(
        dic_path["PATH_TO_WORK_DIRECTORY"], dic_traffic_env_conf["ROADNET_FILE"]
    )
    if stored_map != current_map:
        raise ValueError(
            "Mismatched intersection index mapping. Refusing to resume training. "
            "Stored map at {0} does not match intersections in ROADNET_FILE={1}."
            .format(map_path, dic_traffic_env_conf["ROADNET_FILE"])
        )
    if len(stored_map) != dic_traffic_env_conf["NUM_AGENTS"]:
        raise ValueError(
            "Mismatched intersection index mapping size. Refusing to resume training. "
            "Mapping has {0} entries but NUM_AGENTS={1}."
            .format(len(stored_map), dic_traffic_env_conf["NUM_AGENTS"])
        )
    return stored_map


class Updater:

    def __init__(self, cnt_round, dic_agent_conf, dic_traffic_env_conf, dic_path):

        self.cnt_round = cnt_round
        self.dic_path = dic_path
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_agent_conf = dic_agent_conf
        self.agents = []
        self.sample_set_list = []
        self.sample_indexes = None
        self.intersection_index_map = _load_and_validate_intersection_index_map(
            self.dic_path, self.dic_traffic_env_conf
        )

        print("Number of agents: ", dic_traffic_env_conf['NUM_AGENTS'])

        for i in range(dic_traffic_env_conf['NUM_AGENTS']):
            agent_name = self.dic_traffic_env_conf["MODEL_NAME"]
            agent= DIC_AGENTS[agent_name](
                self.dic_agent_conf, self.dic_traffic_env_conf,
                self.dic_path, self.cnt_round, intersection_id=str(i))
            self.agents.append(agent)

    def load_sample_with_forget(self, i):
        sample_set = []
        try:
            sample_file = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                            "total_samples_inter_{0}".format(i) + ".pkl"), "rb")
            try:
                cur_sample_set = []
                while True:
                    cur_sample_set += pickle.load(sample_file)
            except EOFError:
                print("===== load samples finished =====")
                sample_file.close()

            ind_end = len(cur_sample_set)
            ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
            # forget
            memory_after_forget = cur_sample_set[ind_sta: ind_end]
            print("==== memory size after forget ====:", len(memory_after_forget))
            if self.cnt_round % self.dic_traffic_env_conf["FORGET_ROUND"] == 0:
                with open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                       "total_samples_inter_{0}".format(i) + ".pkl"), "wb+") as f:
                    pickle.dump(memory_after_forget, f, -1)
            # sample the memory
            sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
            if self.sample_indexes is None:
                self.sample_indexes = random.sample(range(len(memory_after_forget)), sample_size)
            sample_set = [memory_after_forget[k] for k in self.sample_indexes]
            print("==== memory samples number =====:", sample_size)

        except:
            error_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"]).replace("records", "errors")
            if not os.path.exists(error_dir):
                os.makedirs(error_dir)
            f = open(os.path.join(error_dir, "error_info_inter_{0}.txt".format(i)), "a")
            f.write("Fail to load samples for inter {0}\n".format(i))
            f.write('traceback.format_exc():\n%s\n' % traceback.format_exc())
            f.close()
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            pass
        if i % 100 == 0:
            print("load_sample for inter {0}".format(i))
        return sample_set

    def load_sample_for_agents(self):
        start_time = time.time()
        print("Start load samples at", start_time)
        if self.dic_traffic_env_conf['MODEL_NAME'] in ["EfficientPressLight",  "EfficientMPLight",
                                                       "AdvancedMPLight", "AdvancedDQN", "Attend"]:
            sample_set_all = []
            for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
                sample_set = self.load_sample_with_forget(i)
                sample_set_all.extend(sample_set)
            self.agents[0].prepare_Xs_Y(sample_set_all)
        elif self.dic_traffic_env_conf['MODEL_NAME'] in ["PressLight"]:
            for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
                sample_set = self.load_sample_with_forget(i)
                self.agents[i].prepare_Xs_Y(sample_set)
        elif self.dic_traffic_env_conf['MODEL_NAME'] in ["EfficientColight", "AdvancedColight"]:
            samples_list = []
            for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
                sample_set = self.load_sample_with_forget(i)
                # [sample1, sample2, ...]
                samples_list.append(sample_set)
            self.agents[0].prepare_Xs_Y(samples_list)

    def update_network(self, i):
        print('update agent %d' % i)
        self.agents[i].train_network()
        self.agents[i].save_network("round_{0}_inter_{1}".format(self.cnt_round, self.agents[i].intersection_id))

    def update_network_for_agents(self):
        print("update_network_for_agents", self.dic_traffic_env_conf['NUM_AGENTS'])
        for i in range(self.dic_traffic_env_conf['NUM_AGENTS']):
            self.update_network(i)
