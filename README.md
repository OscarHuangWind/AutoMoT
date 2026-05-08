
# [ICML'26]AutoMoT: A Unified Vision-Language-Action Model with Asynchronous Mixture-of-Transformers for End-to-End Autonomous Driving

<p align="center">
  <a href="https://icml.cc/">
    <img src="./assets/icml_logo.svg" alt="ICML" height="40">
  </a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.14851"><img src="https://img.shields.io/badge/arXiv-2603.14851-b31b1b?style=flat-square&logo=arxiv" alt="arXiv"></a>
  &nbsp;
  <a href="https://automot-website.github.io/"><img src="https://img.shields.io/badge/Project_Page-AutoMoT-blueviolet?style=flat-square&logo=googlechrome&logoColor=white" alt="Project Page"></a>
  &nbsp;
  <a href="https://huggingface.co/Oscar-Huang/AutoMoT"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Weights-AutoMoT-yellow?style=flat-square" alt="Weights"></a>
  &nbsp;
  <a href="https://huggingface.co/datasets/Oscar-Huang/NuSync"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Datasets-NuSync-orange?style=flat-square" alt="Datasets"></a>
</p>

https://github.com/user-attachments/assets/dcd08673-5ea5-49a1-8dca-5d4b4b8d91fa

**AutoMoT** is an asyncronous VLA end-to-end autonomous driving agent accepted at **ICML 2026**.

> **Current release**: Closed-loop inference on Bench2Drive (220 routes); model checkpoints and NuSync dataset are public. Training code coming soon — see [TODO](#todo-list).

---

## Table of Contents

1. [Method Overview](#method-overview)
2. [Repository Structure](#repository-structure)
3. [Environment Setup](#environment-setup)
4. [Model Weights](#model-weights)
5. [Running Evaluation](#running-evaluation)
6. [Benchmark Results](#benchmark-results)
7. [TODO List](#todo-list)
8. [Citation](#citation)

---

## Method Overview <a name="method-overview"></a>

AutoMoT uses an **Asynchronous Mixture-of-Transformers** design: a slow Understanding Expert (4B) performs low-frequency reasoning, while a fast Action Expert (1.6B) runs at high frequency to decode 3-second decisions and spatial-temporal waypoints via KV-cache bridging.

---

## Repository Structure <a name="repository-structure"></a>

```
Bench2Drive_opensource/
├── Automot/                          # AutoMoT model and agent utilities
│   ├── mot/
│   │   ├── modeling/
│   │   │   ├── automot/              # Core model: AutoMoT, configs, connectors
│   │   │   ├── bev_encoder/          # BEV encoder backbone 
│   │   │   ├── cache_utils/          # KV-cache utilities
│   │   │   └── qwen3/                # Qwen3 text backbone
│   │   ├── data/reasoning/           # Special token handling
│   │   └── evaluation/               # Inference engine (slow/fast KV-cache)
│   ├── team_code/                    # UKF, LiDAR preprocessing, prompt builders
│   └── checkpoints/                  # Model weights (downloaded separately)
│       ├── model.safetensors         # All weights: AutoMoT
│       ├── config.json               # Qwen3-VL model config
│       ├── tokenizer*.json           # Tokenizer files
│       ├── preprocessor_config.json  # Vision preprocessor
│       └── bev_config.json           # BEV encoder GlobalConfig
├── leaderboard/                      # Bench2Drive evaluation harness
│   ├── team_code/
│   │   ├── mot_b2d_agent.py          # Main CARLA agent entry point
│   │   ├── automot_utils.py          # Model loading + prompt utilities
│   │   └── bev_data_utils.py         # LiDAR → BEV histogram features
│   ├── data/bench2drive220/          # 220 route XML files
│   └── scripts/
│       └── run_evaluation_route.sh   # Route-by-route evaluation
├── scenario_runner/                  # CARLA scenario execution
```

---

## Environment Setup <a name="environment-setup"></a>

### 1. CARLA 0.9.15

```bash
mkdir carla && cd carla
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
tar -xvf CARLA_0.9.15.tar.gz
cd Import && wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
cd .. && bash ImportAssets.sh
export CARLA_ROOT=/path/to/carla  # set to the directory containing CarlaUE4.sh
```

### 2. Create the `automot` environment

```bash
conda create -n automot python=3.10
conda activate automot
```

### 3. PyTorch

```bash
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
```

### 4. Python dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# flash-attn (requires torch to be installed first)
pip install flash-attn==2.8.3 --no-build-isolation

# CARLA Python API
pip install $CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-cp310-cp310-linux_x86_64.whl
```

### 5. Environment variables

```bash
export CARLA_ROOT=/path/to/carla
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla:$PYTHONPATH
```

---

## Model Weights <a name="model-weights"></a>

<p align="center">
  <a href="https://huggingface.co/Oscar-Huang/AutoMoT">
    <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg" alt="Model on HuggingFace">
  </a>
</p>

All weights are hosted at **[Oscar-Huang/AutoMoT](https://huggingface.co/Oscar-Huang/AutoMoT)**.

| File | Local destination | Description | Size |
|------|------------------|-------------|------|
| `model.safetensors` | `Automot/checkpoints/model.safetensors` | All model weights | ~13 GB |
| `config.json` | `Automot/checkpoints/` | Qwen3-VL model config | < 1 MB |
| `tokenizer*.json` | `Automot/checkpoints/` | Tokenizer files | < 1 MB |
| `preprocessor_config.json` | `Automot/checkpoints/` | Vision preprocessor | < 1 MB |
| `bev_config.json` | `Automot/checkpoints/` | BEV encoder config | < 1 MB |

```bash
huggingface-cli download Oscar-Huang/AutoMoT \
    --local-dir Automot/checkpoints \
    --repo-type model
```

---

## Running Evaluation <a name="running-evaluation"></a>

### Route-by-route evaluation

```bash
cd leaderboard/scripts
bash run_evaluation_route.sh
```

This script:
- Runs all 220 routes sequentially, skipping already completed ones
- Saves per-route JSON to `leaderboard/scripts/v_2json_open/`

---

## Benchmark Results <a name="benchmark-results"></a>

Bench2Drive 220-route closed-loop evaluation (DS↑ / SR↑):

<p align="center">
  <img src="./assets/b2d.png" alt="Bench2Drive Results" width="85%">
</p>

**AutoMoT achieves DS=87.34 / SR=70.00**

---

## TODO List <a name="todo-list"></a>

- [x] Bench2Drive closed-loop inference (220 routes, CARLA 0.9.15)
- [x] Model checkpoint release ([HuggingFace](https://huggingface.co/Oscar-Huang/AutoMoT))
- [x] NuSync dataset release ([HuggingFace](https://huggingface.co/datasets/Oscar-Huang/NuSync))
- [ ] Training code release

---

## Citation <a name="citation"></a>

```bibtex
@article{huang2026automot,
  title   = {AutoMoT: A Unified Vision-Language-Action Model with Asynchronous Mixture-of-Transformers for End-to-End Autonomous Driving},
  author  = {Wenhui Huang and Songyan Zhang and Qihang Huang and Zhidong Wang and Zhiqi Mao and Collister Chua and Zhan Chen and Long Chen and Chen Lv},
  journal = {arXiv preprint arXiv:2603.14851},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.14851}
}

@inproceedings{jia2024bench,
  title     = {Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving},
  author    = {Xiaosong Jia and Zhenjie Yang and Qifeng Li and Zhiyuan Zhang and Junchi Yan},
  booktitle = {NeurIPS 2024 Datasets and Benchmarks Track},
  year      = {2024}
}
```

---

## Acknowledgements

We thank [TransFuser++](https://github.com/autonomousvision/carla_garage), [SimLingo](https://github.com/autonomousvision/simlingo), and [BAGEL](https://github.com/bytedance/BAGEL) for their open-source contributions, which this work builds upon.
