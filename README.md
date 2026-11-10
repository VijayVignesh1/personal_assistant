<p align="center">
  <img src="docs/assets/personal_assistant_logo.jpg" width="50%" alt="Personal Assistant">
</p>

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776AB?style=plastic&logo=python&logoColor=white)
[![codecov](https://codecov.io/github/VijayVignesh1/personal_assistant/graph/badge.svg?token=0O7YQFLO46)](https://codecov.io/github/VijayVignesh1/personal_assistant)
[![Pre-merge checks](https://github.com/VijayVignesh1/personal_assistant/actions/workflows/pre-merge.yml/badge.svg)](https://github.com/VijayVignesh1/personal_assistant/actions/workflows/pre-merge.yml)
![GitHub Stars](https://img.shields.io/github/stars/VijayVignesh1/personal_assistant?style=plastic)
![GitHub Issues](https://img.shields.io/github/issues/VijayVignesh1/personal_assistant?style=plastic)
![Pull Requests](https://img.shields.io/github/issues-pr/VijayVignesh1/personal_assistant?style=plastic)
![GitHub License](https://img.shields.io/github/license/VijayVignesh1/personal_assistant?style=plastic)

A privacy-focused personal AI assistant with long-term memory, semantic retrieval, and local-first data storage, designed to evolve into a private multimodal assistant.

## Table of Contents

- [Demo](#demo)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Demo

<p align="center">
  <img src="docs/assets/personal_assistant_demo.gif" alt="Personal Assistant Demo" width="800">
</p>

## Features

- Persistent Memory: Retains important context across sessions for a personalized experience.

- Local Execution: Runs locally to keep your data private and under your control.

- Conversational Interface: Enables natural interactions through an intuitive chat-based interface.

- Extensible Architecture — Supports integration of additional tools and capabilities.

- Lightweight Setup — Simplifies dependency management and development using uv.

## Installation
The Python code is located in the python/ directory.

### Prequisites
- Python 3.11–3.13
- [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU with CUDA 12.4 support
- NVIDIA driver compatible with CUDA 12.4

### 1. Clonse the repository
```
git clone <repository-url> 
cd personal_assistant
```

### 2. Create the virtual environent
```
uv venv --python 3.11
```

### 3. Activate the virtual environment
#### macOS / Linux
```
source .venv/bin/activate
```

#### Windows:
```
.venv\Scripts\activate
```

### 4. Install dependencies

Install the project and development dependencies:

```
uv sync
```

This installs the dependencies defined in pyproject.toml and creates/updates uv.lock.

NOTE: In order to run the models on GPU CUDA, make sure to install the pytorch version compatible with the underlying CUDA version. To find the cuda version, run `nvidia-smi`. For example, if the cuda version is 12.5, install any pytorch version with cuda 12.5 and below as shown below:
```
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```


### 5. Verify the installation

#### Run the test suite
```
uv run pytest
```

#### Run linting
```
uv run ruff check . --fix
```

#### Run type checking
```
uv run mypy src
```

### 6. Adding a Package

#### To add a production dependency
```
uv add <package-name>
```
For example:
```
uv add numpy
```

#### To add a development dependency
```
uv add --dev pytest
```

After adding packages, pyproject.toml and uv.lock are updated automatically.


## Usage

1. Launch the chainlit application using:
```
chainlit run main.py -w
```
2. A chat window should open in the default browser automatically. If not, navigate to `http://localhost:8000/`


## Benchmarks

### HuggingFace vs Ollama

Hardware: NVIDIA RTX 4000 12GB
Model: DeepSeek R1 1.5B

| Model              |   concurrency |   total_requests |   total_tokens |   total_time |   requests_per_sec |   tokens_per_sec |   avg_ttft |   avg_e2e |    avg_itl |   avg_gpu_util |   max_gpu_util |   avg_gpu_memory_mb |   max_gpu_memory_mb |   avg_gpu_memory_util |   avg_cpu_util |   max_cpu_util |   avg_ram_util |   max_ram_util |
|--------------------|---------------|------------------|----------------|--------------|--------------------|------------------|------------|-----------|------------|----------------|----------------|---------------------|---------------------|-----------------------|----------------|----------------|----------------|----------------|
| huggingface (c=5)  |             5 |                5 |             45 |     1.10081  |            4.54211 |          40.879  |   0.592975 |  1.05824  | 0.0581584  |        14.1818 |             70 |             7184.23 |             7216.44 |               9.36364 |        7.42727 | 15.8 |        86.5    |           87.1 |
| ollama (c=5)       |             5 |                5 |             45 |     0.359227 |           13.9188  |         125.269  |   0.187847 |  0.213594 | 0.00321848 |        76.75   |             79 |             7210.12 |             7211.06 |              66.5     |        9.95    | 16.7 |        87.1    |           87.1 |
| huggingface (c=10) |            10 |               10 |             90 |     1.25321  |            7.9795  |          71.8155 |   0.193123 |  1.2214   | 0.128535   |        61.8462 |             74 |             7281.16 |             7290.19 |              54.0769  |       16.9538  |100   |        87.0846 |           87.1 |
| ollama (c=10)      |            10 |               10 |             90 |     0.550972 |           18.1497  |         163.348  |   0.277227 |  0.303159 | 0.00324158 |        42.8333 |             81 |             7275.52 |             7280.94 |              38       |       16.7333  | 29.5 |        86.9    |           87   |
| huggingface (c=15) |            15 |               15 |            135 |     2.39588  |            6.26076 |          56.3468 |   0.336903 |  2.33026  | 0.24917    |        53.9583 |             81 |             7378.7  |             7386.5  |              46.2917  |       11.3792  | 23.9 |        86.6875 |           86.7 |
| ollama (c=15)      |            15 |               15 |            135 |     0.851529 |           17.6154  |         158.538  |   0.431983 |  0.458881 | 0.0033623  |        48.5    |             82 |             7386.55 |             7386.56 |              39.125   |       12.8125  | 16.8 |        86.6    |           86.6 |
| huggingface (c=20) |            20 |               20 |            180 |     3.8845   |            5.14866 |          46.338  |   0.482625 |  3.79936  | 0.414592   |        57.5263 |            100 |             7481.28 |             7486.62 |              40.3158  |        9.99211 | 25   |        86.6026 |           86.7 |
| ollama (c=20)      |            20 |               20 |            180 |     1.99531  |           10.0235  |          90.2114 |   1.02491  |  1.0827   | 0.00722444 |        77.1579 |            100 |             7481.65 |             7488.62 |              40.3158  |       16.6263  | 30   |        86.6053 |           86.7 |

## Contributing

Contributions are welcome! If you'd like to improve the project, fix bugs, or suggest new features, feel free to contribute.

## Show Your Support

If you find this project useful, please consider giving it a star on GitHub. It helps increase the project's visibility and motivates further development.

If you have suggestions, encounter bugs, or want to contribute, feel free to open an issue or submit a pull request.

Thank you for your support!