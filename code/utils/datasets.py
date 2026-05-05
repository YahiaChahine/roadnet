import argparse
import json
import os
import re

DATASET_REGISTRY = {
    "run_mplight.py": {
        "jinan": {"template": "Jinan", "roadnet": "3_4", "roadnet_file": "roadnet_3_4.json", "traffic_files": ["anon_3_4_jinan_real.json", "anon_3_4_jinan_real_2000.json", "anon_3_4_jinan_real_2500.json"], "run_counts": 3600, "num_rounds": 80, "overrides": {"reward_weights": {"pressure": -0.25}, "state_features": ["cur_phase", "traffic_movement_pressure_num"], "phase_list_assumption": "optional eight-phase override"}},
        "hangzhou": {"template": "Hangzhou", "roadnet": "4_4", "roadnet_file": "roadnet_4_4.json", "traffic_files": ["anon_4_4_hangzhou_real.json", "anon_4_4_hangzhou_real_5816.json"], "run_counts": 3600, "num_rounds": 80, "overrides": {"reward_weights": {"pressure": -0.25}, "state_features": ["cur_phase", "traffic_movement_pressure_num"], "phase_list_assumption": "optional eight-phase override"}},
        "xuancheng_mod": {"template": "xuancheng_mod", "roadnet": "", "roadnet_file": "roadnet_xuancheng250319.json", "traffic_files": ["xuancheng_2023_04_01_8to9.json", "xuancheng_2023_04_02_8to9.json"], "default_traffic_files": ["xuancheng_2023_04_01_8to9.json", "xuancheng_2023_04_02_8to9.json"], "expected_flow_pattern": r"^xuancheng_\d{4}_\d{2}_\d{2}_8to9\.json$", "run_counts": 3600, "num_rounds": 80, "topology_mode": "general_topology", "non_grid": True, "num_intersections": 20, "overrides": {"reward_weights": {"pressure": -0.25}, "state_features": ["cur_phase", "traffic_movement_pressure_num"], "phase_list_assumption": "optional eight-phase override"}},
    },
}

_aliases = ["run_colight.py", "run_maxpressure.py", "run_efficient_mplight.py", "run_efficient_colight.py", "run_efficient_maxpressure.py", "run_advanced_mplight.py", "run_advanced_colight.py", "run_advanced_maxpressure.py"]
for k in _aliases:
    DATASET_REGISTRY[k] = DATASET_REGISTRY["run_mplight.py"]

DATASET_REGISTRY["run_efficient_presslight.py"] = DATASET_REGISTRY["run_mplight.py"]
DATASET_REGISTRY["run_fixedtime.py"] = {
    "jinan": {"template": "Jinan", "roadnet": "3_4", "roadnet_file": "roadnet_3_4.json", "traffic_files": ["anon_3_4_jinan_real.json", "anon_3_4_jinan_real_2000.json", "anon_3_4_jinan_real_2500.json"], "run_counts": 3600, "num_rounds": 0, "overrides": {"reward_weights": {"pressure": 0}, "state_features": ["cur_phase", "time_this_phase", "traffic_movement_pressure_queue"]}},
    "hangzhou": {"template": "Hangzhou", "roadnet": "4_4", "roadnet_file": "roadnet_4_4.json", "traffic_files": ["anon_4_4_hangzhou_real.json", "anon_4_4_hangzhou_real_5734.json", "anon_4_4_hangzhou_real_5816.json"], "run_counts": 3600, "num_rounds": 0, "overrides": {"reward_weights": {"pressure": 0}, "state_features": ["cur_phase", "time_this_phase", "traffic_movement_pressure_queue"]}},
    "xuancheng_mod": {"template": "xuancheng_mod", "roadnet": "", "roadnet_file": "roadnet_xuancheng250319.json", "traffic_files": ["xuancheng_2023_04_01_8to9.json", "xuancheng_2023_04_02_8to9.json"], "default_traffic_files": ["xuancheng_2023_04_01_8to9.json", "xuancheng_2023_04_02_8to9.json"], "expected_flow_pattern": r"^xuancheng_\d{4}_\d{2}_\d{2}_8to9\.json$", "run_counts": 3600, "num_rounds": 0, "topology_mode": "general_topology", "non_grid": True, "num_intersections": 20, "overrides": {"reward_weights": {"pressure": 0}, "state_features": ["cur_phase", "time_this_phase", "traffic_movement_pressure_queue"]}},
}
DATASET_REGISTRY["run_attendlight.py"] = {
    "jinan": {"template": "Jinan", "roadnet": "3_4", "roadnet_file": "roadnet_3_4.json", "traffic_files": ["anon_3_4_jinan_real.json", "anon_3_4_jinan_real_2000.json", "anon_3_4_jinan_real_2500.json"], "run_counts": 3600, "num_rounds": 80, "overrides": {"reward_weights": {"pressure": -0.25}, "state_features": ["num_in_seg_attend"]}},
    "hangzhou": {"template": "Hangzhou", "roadnet": "4_4", "roadnet_file": "roadnet_4_4.json", "traffic_files": ["anon_4_4_hangzhou_real.json", "anon_4_4_hangzhou_real_5816.json"], "run_counts": 3600, "num_rounds": 80, "overrides": {"reward_weights": {"pressure": -0.25}, "state_features": ["num_in_seg_attend"]}},
    "newyork": {"template": "newyork_28_7", "roadnet": "28_7", "roadnet_file": "roadnet_28_7.json", "traffic_files": ["anon_28_7_newyork_real_double.json"], "run_counts": 3600, "num_rounds": 80, "overrides": {"reward_weights": {"pressure": -0.25}, "state_features": ["num_in_seg_attend"]}},
    "xuancheng_mod": {"template": "xuancheng_mod", "roadnet": "", "roadnet_file": "roadnet_xuancheng250319.json", "traffic_files": ["xuancheng_2023_04_01_8to9.json", "xuancheng_2023_04_02_8to9.json"], "default_traffic_files": ["xuancheng_2023_04_01_8to9.json", "xuancheng_2023_04_02_8to9.json"], "expected_flow_pattern": r"^xuancheng_\d{4}_\d{2}_\d{2}_8to9\.json$", "run_counts": 3600, "num_rounds": 80, "overrides": {"reward_weights": {"pressure": -0.25}, "state_features": ["num_in_seg_attend"]}},
}


def get_supported_datasets(script_name):
    if script_name not in DATASET_REGISTRY:
        return []
    return sorted(DATASET_REGISTRY[script_name].keys())


def get_dataset_config(script_name, dataset_name):
    script_registry = DATASET_REGISTRY.get(script_name)
    if not script_registry:
        raise argparse.ArgumentTypeError(f"No dataset registry configured for script '{script_name}'.")
    if dataset_name not in script_registry:
        supported = ", ".join(sorted(script_registry.keys()))
        raise argparse.ArgumentTypeError(
            f"Dataset '{dataset_name}' is not supported by {script_name}. Supported datasets: {supported}."
        )
    return script_registry[dataset_name]


def validate_dataset_files(dataset_conf, data_root="data"):
    template = dataset_conf["template"]
    roadnet = dataset_conf["roadnet"]
    dataset_dir = os.path.join(data_root, template, str(roadnet))

    if template == "xuancheng_mod":
        expected_pattern = dataset_conf.get("expected_flow_pattern")
        roadnet_file = dataset_conf.get("roadnet_file")
        traffic_files = dataset_conf.get("traffic_files", [])
        if not traffic_files:
            default_files = dataset_conf.get("default_traffic_files", [])
            raise argparse.ArgumentTypeError(
                "No traffic files configured for xuancheng_mod. "
                f"Provide one-hour flow files like {default_files}. "
                "Use code/crop_x.py to generate valid one-hour files."
            )
        for traffic_file in traffic_files:
            if expected_pattern and not re.match(expected_pattern, traffic_file):
                raise argparse.ArgumentTypeError(
                    f"Invalid xuancheng_mod flow filename '{traffic_file}'. "
                    "Expected format: xuancheng_YYYY_MM_DD_8to9.json. "
                    "Use code/crop_x.py to generate valid one-hour files."
                )
            flow_path = os.path.join(dataset_dir, traffic_file)
            if not os.path.exists(flow_path):
                raise argparse.ArgumentTypeError(
                    f"Missing xuancheng_mod flow file: {flow_path}. "
                    "Expected format: xuancheng_YYYY_MM_DD_8to9.json. "
                    "Use code/crop_x.py to generate and place valid one-hour files in this directory."
                )
        if roadnet_file:
            roadnet_path = os.path.join(dataset_dir, roadnet_file)
            if not os.path.exists(roadnet_path):
                raise argparse.ArgumentTypeError(
                    f"Missing xuancheng_mod roadnet file: {roadnet_path}. "
                    "Verify dataset metadata or place the roadnet file in the xuancheng_mod data directory."
                )





def count_non_virtual_intersections(path_to_data, roadnet_file):
    roadnet_path = os.path.join(path_to_data, roadnet_file)
    with open(roadnet_path, "r", encoding="utf-8") as f:
        roadnet_data = json.load(f)

    intersections = roadnet_data.get("intersections", [])
    return sum(1 for inter in intersections if not inter.get("virtual", False))

def resolve_dataset_runtime(dataset_conf):
    template = dataset_conf["template"]
    road_net = dataset_conf.get("roadnet", "")
    roadnet_file = dataset_conf.get("roadnet_file")
    configured_num_intersections = dataset_conf.get("num_intersections")
    topology_mode = dataset_conf.get("topology_mode")
    non_grid = bool(dataset_conf.get("non_grid", False))

    is_grid_by_roadnet = "_" in str(road_net)
    is_grid_dataset = is_grid_by_roadnet and not non_grid

    if is_grid_dataset:
        num_row = int(str(road_net).split("_")[0])
        num_col = int(str(road_net).split("_")[1])
        path_to_data = os.path.join("data", template, str(road_net))
    else:
        num_row = 1
        num_col = 1
        path_to_data = os.path.join("data", template)

    if not roadnet_file:
        roadnet_file = "roadnet_{0}.json".format(road_net)

    if configured_num_intersections is not None:
        num_intersections = int(configured_num_intersections)
    elif is_grid_dataset:
        num_intersections = num_row * num_col
    else:
        num_intersections = count_non_virtual_intersections(path_to_data, roadnet_file)

    if topology_mode is None:
        topology_mode = "grid" if is_grid_dataset else "general_topology"

    return {
        "NUM_ROW": num_row,
        "NUM_COL": num_col,
        "NUM_INTERSECTIONS": num_intersections,
        "ROADNET_FILE": roadnet_file,
        "PATH_TO_DATA": path_to_data,
        "TOPOLOGY_MODE": topology_mode,
        "IS_GRID": is_grid_dataset,
    }
