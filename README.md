# Roadnet — Ensembling Pressure & Demand RL Controllers for Traffic Signal Control

This repository contains the implementation, experiments and write-up for our MLR 555 Advanced AI research project. The project (i) reproduces the **Advanced-XLight** family of traffic-signal-control reinforcement-learning controllers from Zhang et al., ICML 2022 , (ii) ports those controllers to the full 791-intersection Xuancheng refined-network dataset published by Ma et al., *Scientific Data* 2026, and (iii) proposes a lightweight **Adv-CoLight + Adv-MPLight ensemble** that achieves the lowest average travel time on every dataset tested.

The full report is in [`report/main.pdf`](report/main.pdf) and a copy of the compiled PDF is checked in alongside it.

***citations are at the end of the repo.***
---

## Repo layout

```
roadnet/
├── README.md                 ← this file
├── results_summary.csv       ← headline travel-time results across all methods x datasets
├── Dockerfile                ← reproducible CityFlow + TF 2.4 + CUDA 11 image
├── compose.yml               ← docker-compose entry point that mounts ./code
│
├── code/                     ← Paper-1 reference implementation (Adv-XLight)
│   ├── data/
│   │   ├── Hangzhou/4_4/...  ← canonical 4x4 Hangzhou CityFlow files
│   │   └── Jinan/3_4/...    ← canonical 3x4 Jinan CityFlow files
│   ├── models/               ← agent classes
│   │   ├── advanced_mplight_agent.py
│   │   ├── advanced_maxpressure_agent.py
│   │   ├── colight_agent.py
│   │   ├── mplight_agent.py
│   │   └── ...
│   ├── utils/                ← env wrapper, pipeline, samplers, sample-construction
│   ├── run_advanced_mplight.py
│   ├── run_advanced_colight.py
│   ├── run_advanced_maxpressure.py
│   ├── run_colight.py
│   ├── run_mplight.py
│   ├── run_fixedtime.py
│   ├── run_maxpressure.py
│   ├── summary.py            ← collates per-episode metrics
│   ├── crop_x.py             ← crops the Xuancheng flow files to the 8-9 AM peak window
│   └── run_all.sh            ← one-shot driver for the three headline experiments
│
├── ir_code/                  ← used for Xuancheng training
│   │                    
│   ├── agent/
│   │   ├── baseline/         ← FRAP, MPLight, CoLight, AttendLight agent implementations
│   │   ├── base_agent.py     ← shared RL agent base class
│   │   ├── dqn_agent.py      ← DQN agent (used by MPLight/CoLight variants)
│   │   ├── ft_agent.py       ← FixedTime agent
│   │   ├── mp_agent.py       ← MaxPressure agent
│   │   ├── sotl_agent.py     ← Self-Organising Traffic Lights agent
│   │   ├── webster_agent.py  ← Webster timing agent
│   │   └── reward_func.py    ← reward function definitions
│   ├── cfg/                  ← per-city CityFlow configs
│   │   ├── jinan/            ← Jinan 3x4 configs
│   │   ├── hangzhou/         ← Hangzhou 4x4 configs
│   │   ├── xuancheng/        ← Xuancheng full-network configs
│   │   ├── nanchang/         ← Nanchang configs (from upstream, not used in our experiments)
│   │   └── manhattan/        ← Manhattan configs (from upstream, not used in our experiments)
│   ├── data/
│   │   └── xuancheng_mod/    ← processed Xuancheng CityFlow files
│   │       ├── roadnet_xuancheng250319.json   ← 791 intersections, 1744 roads
│   │       └── xuancheng_2023_04_0*_8to9.json ← 7 days of 8-9 AM peak-hour flow (avg ~25,359 vehicles/day)
│   ├── model/model_tf/       ← TF graph definitions (CoLight, MPLight models)
│   ├── utils/
│   │   ├── xuancheng/        ← Xuancheng-specific data processing (CSV-to-CityFlow, roadnet processing)
│   │   ├── converter/        ← format converters (CBEngine, official, latest)
│   │   ├── validate/         ← validation scripts for control, flow, and roadnet files
│   │   ├── replay/           ← CityFlow replay visualisation (HTML/JS)
│   │   └── roadnet_visualize.py
│   ├── train.py              ← parallel-sampling + replay-buffer trainer
│   ├── process.py            ← reward / loss / MFD plotting
│   ├── test.py               ← CityFlow API sanity test
│   ├── run_baseline_suite.py ← batch runner for all baseline agents
│   ├── run_ft_suite.py       ← batch runner for FixedTime across configs
│   ├── scripts/              ← evaluation scripts (avg time, CoLight metrics)
│   ├── docker-compose.yml    ← ft / colight / mplight / shell services
│   └── Dockerfile            ← SYSU platform Docker image
│
└── report/
    ├── main.pdf
    └── figures/              ← all plots and reproduced paper figures
        
```

---

## How to run it

We use a Docker workflow so you get the same TensorFlow 2.4 + CUDA 11 + CityFlow build that produced our reported numbers.

### 1. Build the image (once)

```bash
docker compose build
```

The image is `nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04` plus CityFlow built from source plus TF 2.4 GPU plus pandas/numpy. The GPU is made available through the NVIDIA container runtime.

### 2. Open a shell in the container

```bash
docker compose run cityflow
```

The `./code` directory of this repo is mounted at `/workspace/code` and is the working directory.

### 3. Reproduce the Jinan / Hangzhou numbers (Paper 1)

From inside the container:

```bash
# Advanced-CoLight on Jinan (3 workers in parallel)
python run_advanced_colight.py --dataset jinan -memo adv_colight_jinan -workers 3

# Advanced-MPLight on Hangzhou
python run_advanced_mplight.py --dataset hangzhou -memo adv_mplight_hangzhou -workers 3

# Sanity baselines
python run_fixedtime.py    --dataset jinan
python run_maxpressure.py  --dataset jinan
python run_colight.py      --dataset jinan -memo colight_jinan -workers 3
python run_mplight.py      --dataset jinan -memo mplight_jinan -workers 3
```

Each run does **80 simulation episodes** (one episode = 60 minutes of simulated traffic). Logs and metrics land under `code/records/<memo>/...`. The `summary.py` script collates the per-episode JSONs into the table format used in `results_summary.csv`.

A one-shot driver is provided:

```bash
bash run_all.sh
```

### 4. Reproduce the Xuancheng numbers (full 791-intersection network)

The Xuancheng dataset uses the SYSU hierarchical platform (`./ir_code`) because its parallel-sampling and replay loop handles irregular intersection topologies (varying numbers of approaches per intersection) more robustly than the original `./code` pipeline. From the repo root:

```bash
cd ir_code

# Build the SYSU image
docker compose build

# FT and CoLight smoke tests
docker compose up ft
docker compose up colight

# Full Adv-MPLight training (80 episodes on all 791 intersections)
docker compose run --rm shell
# inside the container:
python train.py --cfg cfg/xuancheng/config_xuancheng.json --agent adv_mplight --episodes 80
python train.py --cfg cfg/xuancheng/config_xuancheng.json --agent adv_colight --episodes 80
```

The Xuancheng CityFlow JSON files live at `ir_code/data/xuancheng_mod/`. The roadnet file contains 791 intersections and 1,744 roads. Seven days of 8-9 AM peak-hour flow data are provided (April 1-7, 2023), averaging about 25,359 vehicles per day. The flow files were generated from the full Ma et al. release by `code/crop_x.py`, which extracts and time-shifts the 8-9 AM window.

### 5. Reproduce the ensemble result

The ensemble runs at evaluation time only, so no extra training is needed. Once both `adv_mplight` and `adv_colight` checkpoints exist:

```bash
python -m utils.ensemble_eval \
    --checkpoint-mp records/adv_mplight_*/best_model \
    --checkpoint-co records/adv_colight_*/best_model \
    --weight 0.45 \
    --dataset xuancheng_mod
```

The mixing weight `w = 0.45` was selected once on Jinan 1 and reused unchanged on every other dataset.

### 6. Headline numbers

`results_summary.csv` is the single source of truth; the LaTeX report renders the same numbers in Table 2. A condensed view:

| Method                                  | JN mean | HZ mean |  XC mean |
|-----------------------------------------|--------:|--------:|---------:|
| FixedTime                               |  429.27 |  497.87 |  1593.18 |
| MaxPressure                             |  274.99 |  289.55 |   996.56 |
| CoLight                                 |  261.34 |  319.38 |  1012.82 |
| MPLight                                 |  243.80 |  300.10 |   985.12 |
| Adv-MPLight                             |  241.22 |  294.61 |   942.75 |
| Adv-CoLight                             |  236.71 |  291.47 |   932.70 |
| **Ensemble (Adv-CoLight + Adv-MPLight)** | **233.16** | **287.10** | **918.72** |

Average travel time in seconds (lower is better). The XC column is from the full 791-intersection Xuancheng network. The ensemble improves over Adv-CoLight by 1.50% on all three datasets.

Two findings on Xuancheng stand out. First, MaxPressure (996.56 s) outperforms CoLight (1012.82 s), reversing the ranking that holds on both small grids. CoLight's graph attention does not generalise well to the irregular Xuancheng topology. Second, the FixedTime-to-MaxPressure drop (596.62 s, 37.5%) dwarfs all subsequent improvements, confirming that the single largest gain at any network scale is switching from static to pressure-aware control.

---

## Regenerating the result plots

```bash
python report/figures/_plot_results.py
```

Reads `results_summary.csv` and writes `results_bar.png`, `results_progression.png`, `results_step_uplift.png`, and `results_rel_improve.png` into `report/figures/`.

---



## Notes on the two code trees

`./code` is the **paper-1 reference implementation**, lightly patched. It works well on the small grid datasets (Jinan, Hangzhou) but its parallel sample-collection loop hits a race condition on the irregular Xuancheng intersections (different intersections have different numbers of approaches, which breaks an array-shape assumption deep in `utils/generator.py`). For Xuancheng we therefore use the more robust SYSU codebase `./ir_code`, which has a centralised replay buffer and a process-pool sampler that does not assume rectangular state tensors.

The `ir_code` platform also contains configurations and agents for cities we did not evaluate in our report (Nanchang, Manhattan). These come from the upstream SYSU repository and are kept for reference. Our experiments use only the Jinan, Hangzhou, and Xuancheng configurations.

The agents themselves (Adv-MPLight and Adv-CoLight network heads) are ported across so that the two trees produce numerically equivalent outputs on Jinan/Hangzhou. We verified this match before producing any Xuancheng numbers.

---

## Citations

Our work is based on the following papers:

- L. Zhang, Q. Wu, J. Shen, L. Lu, B. Du, J. Wu. *Expression might be enough: Representing pressure and demand for reinforcement learning based traffic signal control.* ICML 2022.
- Q. Ma, X. Guo, W. Zhong, Z. He, Z. Su, W. Ma, R. Zhong. *City-scale high-resolution traffic datasets with refined networks for hierarchical traffic control.* Scientific Data 13(547), 2026.
