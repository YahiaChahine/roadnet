from utils.utils import pipeline_wrapper, merge
from utils import config
import time
from multiprocessing import Process
import argparse
import os
from utils.datasets import get_dataset_config, get_supported_datasets, validate_dataset_files, resolve_dataset_runtime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-memo", "--memo", type=str, default='benchmark_1001')
    parser.add_argument("-mod",        type=str,           default="EfficientColight")
    parser.add_argument("-eightphase",  action="store_true", default=False)
    parser.add_argument("-gen",        type=int,            default=1)
    parser.add_argument("-multi_process", action="store_true", default=True)
    parser.add_argument("-workers",    type=int,            default=3)
    parser.add_argument("--dataset", type=str, default="jinan",
                        choices=get_supported_datasets("run_colight.py"))
    return parser.parse_args()


def main(in_args=None):
    dataset_conf = get_dataset_config("run_colight.py", in_args.dataset)
    count = dataset_conf["run_counts"]
    traffic_file_list = dataset_conf["traffic_files"]
    num_rounds = dataset_conf["num_rounds"]
    template = dataset_conf["template"]

    validate_dataset_files(dataset_conf)
    runtime_conf = resolve_dataset_runtime(dataset_conf)
    NUM_ROW = runtime_conf["NUM_ROW"]
    NUM_COL = runtime_conf["NUM_COL"]
    roadnet_file = runtime_conf["ROADNET_FILE"]
    path_to_data = runtime_conf["PATH_TO_DATA"]

    num_intersections = runtime_conf["NUM_INTERSECTIONS"]
    print('num_intersections:', num_intersections)
    print(traffic_file_list)
    process_list = []
    for traffic_file in traffic_file_list:
        dic_agent_conf_extra = {
            "CNN_layers": [[32, 32]],
        }
        deploy_dic_agent_conf = merge(getattr(config, "DIC_BASE_AGENT_CONF"), dic_agent_conf_extra)

        dic_traffic_env_conf_extra = {

            "NUM_ROUNDS": num_rounds,
            "NUM_GENERATORS": in_args.gen,
            "NUM_AGENTS": num_intersections,
            "NUM_INTERSECTIONS": num_intersections,
            "RUN_COUNTS": count,

            "MODEL_NAME": in_args.mod,
            "NUM_ROW": NUM_ROW,
            "NUM_COL": NUM_COL,
            "TRAFFIC_FILE": traffic_file,

            "ROADNET_FILE": roadnet_file,

            "LIST_STATE_FEATURE": [
                "cur_phase",
                "lane_num_vehicle",
                "adjacency_matrix",
            ],

            "DIC_REWARD_INFO": {
                "queue_length": -0.25,
            },
        }

        if in_args.eightphase:
            dic_traffic_env_conf_extra["PHASE"] = {
                1: [0, 1, 0, 1, 0, 0, 0, 0],
                2: [0, 0, 0, 0, 0, 1, 0, 1],
                3: [1, 0, 1, 0, 0, 0, 0, 0],
                4: [0, 0, 0, 0, 1, 0, 1, 0],
                5: [1, 1, 0, 0, 0, 0, 0, 0],
                6: [0, 0, 1, 1, 0, 0, 0, 0],
                7: [0, 0, 0, 0, 0, 0, 1, 1],
                8: [0, 0, 0, 0, 1, 1, 0, 0]
            }
            dic_traffic_env_conf_extra["PHASE_LIST"] = ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL',
                                                        'WL_WT', 'EL_ET', 'SL_ST', 'NL_NT']
        dic_path_extra = {
            "PATH_TO_MODEL": os.path.join("model", in_args.memo, traffic_file + "_" +
                                          time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", in_args.memo, traffic_file + "_"
                                                   + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
            "PATH_TO_DATA": path_to_data,
            "PATH_TO_ERROR": os.path.join("errors", in_args.memo)
        }
        deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)
        deploy_dic_path = merge(config.DIC_PATH, dic_path_extra)

        if in_args.multi_process:
            ppl = Process(target=pipeline_wrapper,
                          args=(deploy_dic_agent_conf,
                                deploy_dic_traffic_env_conf,
                                deploy_dic_path))
            process_list.append(ppl)
        else:
            pipeline_wrapper(dic_agent_conf=deploy_dic_agent_conf,
                             dic_traffic_env_conf=deploy_dic_traffic_env_conf,
                             dic_path=deploy_dic_path)

    if in_args.multi_process:
        for i in range(0, len(process_list), in_args.workers):
            i_max = min(len(process_list), i + in_args.workers)
            for j in range(i, i_max):
                print(j)
                print("start_traffic")
                process_list[j].start()
                print("after_traffic")
            for k in range(i, i_max):
                print("traffic to join", k)
                process_list[k].join()
                print("traffic finish join", k)

    return in_args.memo


if __name__ == "__main__":
    args = parse_args()

    main(args)