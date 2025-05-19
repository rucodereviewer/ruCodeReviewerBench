# code-review-paper

Repository for the paper `RuCodeReviewer: A Human-Validated Benchmark with LLM-Based Metrics for Code Review Evaluation`.

# Abstract

We present RuCodeReviewer, the first open benchmark for automated code review comment generation in Russian. Prioritizing quality and reproducibility over sheer size, we constructed a dataset of 699 merge-request diffs across Java, Python, Scala, and Go, ensuring issues are verifiable and comments are traceable to code changes. Our rigorous two-stage filtering pipeline combined LLM screening with dual human verification.

We developed a data-driven taxonomy grouping comments into semantic categories and evaluated state-of-the-art code LLMs using BLEU, ChrF, and two LLM-as-a-Judge approaches (reference-based and reference-free) reporting pass@k.

Our findings show the task remains challenging, with current LLMs providing moderate quality comments. We openly release RuCodeReviewer, including the dataset, scoring scripts, and baseline outputs under open licenses, providing a reproducible foundation for advancing multilingual automated code review research.

# Project structure

```
├── rucodereview_main.jsonl
├── pyproject.toml
├── few_shot.json
├── README.md
├── LICENSE
├── uv.lock
├── main.py
└── src
    ├── data_collection
    │   ├── script_python/
    │   ├── script_scala/
    │   ├── parsing.ipynb
    │   ├── script_java/
    │   └── script_go/
    ├── rucodereview
    │   ├── rucodereview.yaml
    │   └── utils.py
    ├── figures
    │   ├── visualize.ipynb
    │   └── corr.ipynb
    └── multimetrics
        ├── get_best_comment.py
        ├── multimetrics.py
        ├── async_compl.py        
        └── prompts.py
```

### File Description

- src - Scripts for data processing.
- rucodereview_main.jsonl - Main DataSet.
- few_shot.json - Few-shot results for LLM-as-a-Judge.
- data_collection - Notebook for downloading dataset and scripts to highlight a block of code.
- figures - Notebooks to create graphs for the paper. You need model results to reproduce them.

# Requirements

- Python 3.10+

Libs:

- ipykernel >= 6.29.5
- litellm >= 1.69.1
- lm-eval >= 0.4.8
- matplotlib >= 3.10.3
- pandas >= 2.2.3
- sacrebleu >= 2.5.1
- seaborn >= 0.13.2

Python | uv

```bash
uv sync
```

Python | pip

```bash
pip install --upgrade pip setuptools
pip install .
```

# LLM for Judge Configuration

This project supports interacting with Large Language Models (LLMs) in two ways:

1. **Local Execution (GPU Recommended):** If you have one or more 80GB GPU, you can run LLM for judge locally using `start_llm_model.sh` or `start_llm_model_docker.sh` script, its recommended to use docker for easy setup.
2. **Remote OpenAI-like Endpoint:** You can connect to any OpenAI-compatible API endpoint. To do this, you will need to set the following environment variables:
    - `BASE_URL`: The base URL of the API endpoint. In case of using local LLM, it should be `http://localhost:8000`.
    - `MODEL`: The model name to use. In case of using local LLM, it should be `qwen-coder-32b`.
    - `API_KEY`: Your API key for the service. In case of using local LLM, it should be '123'

# License Agreement

The source code is licensed under the Apache 2.0 License that can be found at the root directory.
The dataset is licensed under the CC0 1.0 Universal License that can be found at the root directory.

# Running Evaluation Scripts
Read file `benchmark/README.md` for more details.
The scripts are pre-configured to look for data in specific directories.
To modify the behavior of these scripts, you may need to edit the source files directly to adjust file paths or other parameters.

# Receiving validation results
The dataset used for evaluation is not publicly available. To evaluate your model on our dataset, please contact us at `rucodereview@gmail.com`. We will process your request and provide the evaluation results.
