# **ScenePilot-Bench: A Large-Scale First-Person Dataset and Benchmark for Evaluation of Vision-Language Models in Autonomous Driving**
<div align="center">
  <img src="assets/The overall structure.png" width="800px">
  <p>Figure 1: Overview of the ScenePilot-Bench dataset and evaluation metrics.</p>
</div>


[![Project Page](https://img.shields.io/badge/Project-Website-blue?style=flat-square)](https://github.com/yjwangtj/ScenePilot-Bench)
[![Dataset](https://img.shields.io/badge/Dataset-Download-green?style=flat-square)](https://huggingface.co/datasets/larswangtj/ScenePilot-4K/tree/main) 
[![Paper](https://img.shields.io/badge/Paper-Arxiv-red?style=flat-square)](#)

# 📖 Introduction
We introduce ScenePilot-Bench, a large-scale first-person driving benchmark designed to evaluate vision-language models (VLMs) in autonomous driving scenarios. ScenePilot-Bench is built upon ScenePilot-4K, a diverse dataset comprising 3,847 hours of driving videos from 63 countries and regions, annotated with multi-granularity information including scene descriptions, risk assessments, key participant identification, ego trajectories, and camera parameters. The benchmark features a four-axis evaluation suite that assesses VLM capabilities in scene understanding, spatial perception, motion planning, and GPT-Score, with safety-aware metrics and cross-region generalization settings. We benchmark representative VLMs on ScenePilot-Bench, providing empirical analyses that clarify current performance boundaries and identify gaps for driving-oriented reasoning. ScenePilot-Bench offers a comprehensive framework for evaluating and advancing VLMs in safety-critical autonomous driving contexts.

# 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/yjwangtj/ScenePilot-Bench.git
cd ScenePilot-Bench

# 2. Create and activate a Conda environment
conda create -n scenepilot python=3.10 -y
conda activate scenepilot

# 3. Install required dependencies

pip install -r requirements.txt
```

# 🚀 Inference

```bash
# 1. Load Model & Processor
import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

# 1. Load Model & Processor
# Replace with your local model weight directory
model_path = "path/to/ScenePilot_model" 
model = AutoModelForImageTextToText.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    device_map="auto", 
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# 2. Prepare Input
# You can replace this URL with a local image path: Image.open("your_image.jpg")
url = "https://raw.githubusercontent.com/yjwangtj/ScenePilot-Bench/main/assets/sample_drive.jpg"
image = Image.open(BytesIO(requests.get(url).content)).convert("RGB")

# Define Autonomous Driving VQA Prompt
prompt = "Report the current weather, time, road type, how many lanes, if it’s an intersection, and the risk level."

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }
]

text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, _ = process_vision_info(messages)
inputs = processor(
    text=[text_prompt], 
    images=image_inputs, 
    return_tensors="pt"
).to(model.device)

# Generate  response
output_ids = model.generate(**inputs, max_new_tokens=128)
answer = processor.batch_decode(
    output_ids[:, inputs.input_ids.shape[1]:], 
    skip_special_tokens=True
)[0]

print(f"--- ScenePilot Result ---\n{answer}")
```


# 📊ScenePilot Benchmark

This repository provides a **two-step evaluation pipeline** for benchmarking Vision-Language Models (VLMs) on the ScenePilot-Bench dataset.


## Step 1: Scene Graph Parsing

Use `scene_graph_parser_final-all.py` to parse **model-generated answers** into a standardized **scene semantic graph representation**.
The parsed results will be saved as a JSON file, which serves as the input for the benchmark scoring stage.

```bash
python scene_graph_parser_final-all.py \
    --input_path path/to/model_outputs.json \
    --output_path path/to/parsed_scene_graph.json
```

**Output**

* A JSON file containing structured scene graph representations extracted from model predictions.


## Step 2: Benchmark Scoring

Use `benchmark_score_final-all.py` to compute **evaluation metrics and final benchmark scores** based on the parsed scene graphs.

```bash
python benchmark_score_final-all.py \
    --input_path path/to/parsed_scene_graph.json \
    --output_dir path/to/save_results
```

Before running the script, please ensure the following paths are properly configured inside the code or via arguments:

* **GPT output log path** (optional, can be commented out if not required)
* **Normalization parameters JSON file**, used for metric scaling and score normalization

**Outputs**

* A JSON file containing detailed evaluation results for each sample
* A CSV file summarizing all benchmark metrics in tabular form, suitable for comparison across models

---

## Citation

```bibtex
@article{scenepilot,
  title={ScenePilot-Bench: A Large-Scale First-Person Dataset and Benchmark for Evaluation of Vision-Language Models in Autonomous Driving},
  author={Yujin Wang, Yutong Zheng, Wenxian Fan, Jinlong Hong, Wei Tiana,Haiyang Yu, Bingzhao Gao, Jianqiang Wang, Hong Chen},
  journal={arXiv preprint},
  year={2025}
}
```

## License

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
