"""
Run the Fixed-Time model
On JiNan and HangZhou real data
"""
from utils.utils import oneline_wrapper
import os
import time
from multiprocessing import Process
import argparse
from utils.datasets import (
    get_dataset_config,
    get_supported_datasets,
    resolve_dataset_runtime,
    validate_dataset_files,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memo",       type=str,               default='benchmark_1001')
    parser.add_argument("-model",       type=str,               default="Fixedtime")
    parser.add_argument("-multi_process", action="store_true",  default=True)
    parser.add_argument("-workers",     type=int,               default=3)
    parser.add_argument("--dataset", type=str, default="jinan",
                        choices=get_supported_datasets("run_fixedtime.py"))

    return parser.parse_args()


def main(in_args):
    dataset_conf = get_dataset_config("run_fixedtime.py", in_args.dataset)
    count = dataset_conf["run_counts"]
    traffic_file_list = dataset_conf["traffic_files"]
    template = dataset_conf["template"]

    validate_dataset_files(dataset_conf)
    runtime_conf = resolve_dataset_runtime(dataset_conf)
    NUM_ROW = runtime_conf["NUM_ROW"]
    NUM_COL = runtime_conf["NUM_COL"]
    roadnet_file = runtime_conf["ROADNET_FILE"]
    path_to_data = runtime_conf["PATH_TO_DATA"]

    num_intersections = runtime_conf["NUM_INTERSECTIONS"]
    topology_mode = runtime_conf["TOPOLOGY_MODE"]
    print('num_intersections:', num_intersections)
    print(traffic_file_list)
    process_list = []
    for traffic_file in traffic_file_list:
        dic_traffic_env_conf_extra = {
            "NUM_AGENTS": num_intersections,
            "NUM_INTERSECTIONS": num_intersections,
            "MODEL_NAME": in_args.model,
            "RUN_COUNTS": count,
            "NUM_ROW": NUM_ROW,
            "NUM_COL": NUM_COL,
            "TRAFFIC_FILE": traffic_file,
            "ROADNET_FILE": roadnet_file,
            "LIST_STATE_FEATURE": [
                "cur_phase",
                "time_this_phase",
                "traffic_movement_pressure_queue",
            ],
            "DIC_REWARD_INFO": {
                "pressure": 0
            },
        }

        if topology_mode != "grid":
            dic_traffic_env_conf_extra["TOPOLOGY_ADAPTER_MODE"] = topology_mode

        dic_agent_conf_extra = {
            "FIXED_TIME": [15, 15, 15, 15],
        }

        dic_path_extra = {
            "PATH_TO_MODEL": os.path.join("model", in_args.memo, traffic_file + "_" +
                                          time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", in_args.memo, traffic_file + "_" +
                                                   time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
            "PATH_TO_DATA": path_to_data
        }
        if in_args.multi_process:
            process_list.append(Process(target=oneline_wrapper,
                                        args=(dic_agent_conf_extra,
                                              dic_traffic_env_conf_extra, dic_path_extra))
                                )
        else:
            oneline_wrapper(dic_agent_conf_extra, dic_traffic_env_conf_extra, dic_path_extra)

    if in_args.multi_process:
        list_cur_p = []
        for p in process_list:
            if len(list_cur_p) < in_args.workers:
                p.start()
                list_cur_p.append(p)
            if len(list_cur_p) < in_args.workers:
                continue

        for p in list_cur_p:
            p.join()


if __name__ == "__main__":
    args = parse_args()
    main(args)
