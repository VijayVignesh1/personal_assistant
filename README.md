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

#### Ollama Cold Start

| Model       |   total_requests |   total_tokens |   total_time |   requests_per_sec |   tokens_per_sec |   avg_ttft |   avg_e2e |    avg_itl |   avg_gpu_util |   max_gpu_util |   avg_gpu_memory_mb |   max_gpu_memory_mb |   avg_gpu_memory_util |   avg_cpu_util |   max_cpu_util |   avg_ram_util|   max_ram_util |
|-------------|------------------|----------------|--------------|--------------------|------------------|------------|-----------|------------|----------------|----------------|---------------------|---------------------|-----------------------|----------------|----------------|----------------|----------------|
| huggingface |                5 |             45 |     0.933965 |            5.35352 |          48.1817 |   0.469586 |  0.901323 | 0.0539671  |    32.6    |             79 |             5792.16 |             5823.56 |              21.8     |        8.41    |           13.3 |        81.12|           81.8 |
| ollama      |                5 |             45 |     2.2304   |            2.24175 |          20.1758 |   2.06703  |  2.09362  | 0.00332438 |    14.9565 |             77 |             6350.37 |             7156.69 |               8.13043 |        6.11304 |           20.9 |        82.4609|           83   |
| huggingface |               10 |             90 |     1.25729  |            7.95359 |          71.5823 |   0.215188 |  1.2222   | 0.125877   |    58.4615 |             82 |             7238.84 |             7248.69 |              50.5385  |       15.0538  |           33.3 |        83.0923|           83.1 |
| ollama      |               10 |             90 |     0.557466 |           17.9383  |         161.445  |   0.283358 |  0.309495 | 0.0032671  |    45.8333 |             82 |             7248.65 |             7248.69 |              40.6667  |       10.9667  |           16.1 |        83.1|           83.1 |
| huggingface |               15 |            135 |     2.46901  |            6.07531 |          54.6778 |   0.343255 |  2.41223  | 0.258622   |    53.12   |             79 |             7356.64 |             7365.56 |              46.36    |        9.124   |           11.2 |        83.1|           83.1 |
| ollama      |               15 |            135 |     0.825724 |           18.1659  |         163.493  |   0.411513 |  0.437512 | 0.00324978 |    47.125  |             80 |             7365.56 |             7365.56 |              41.625   |       16.5125  |           33.3 |        83.1|           83.1 |
| huggingface |               20 |            180 |     3.40175  |            5.87933 |          52.914  |   0.478501 |  3.31196  | 0.354182   |    48.7353 |             79 |             7452.72 |             7457.5  |              42.5588  |        9.33529 |           12.7 |        83.1853|           83.2 |
| ollama      |               20 |            180 |     1.10949  |           18.0263  |         162.237  |   0.559982 |  0.585979 | 0.00324966 |    55      |             80 |             7456.56 |             7456.56 |              48.5     |       12.84    |           16.1 |        83.19|           83.2 |


#### Ollama Warm Start

| Model       |   total_requests |   total_tokens |   total_time |   requests_per_sec |   tokens_per_sec |   avg_ttft |   avg_e2e |    avg_itl |   avg_gpu_util |   max_gpu_util |   avg_gpu_memory_mb |   max_gpu_memory_mb |   avg_gpu_memory_util |   avg_cpu_util |   max_cpu_util |   avg_ram_util|   max_ram_util |
|-------------|------------------|----------------|--------------|--------------------|------------------|------------|-----------|------------|----------------|----------------|---------------------|---------------------|-----------------------|----------------|----------------|----------------|----------------|
| huggingface |                5 |             45 |     0.954515 |            5.23826 |          47.1444 |   0.463091 |  0.908201 | 0.0556388  |    25.6    |             77 |             7144.69 |             7177.81 |               16.2    |        8.85    |           13.3 |        82.71|           83.3 |
| ollama      |                5 |             45 |     0.33515  |           14.9187  |         134.268  |   0.175115 |  0.201273 | 0.00326967 |    57      |             79 |             7177.81 |             7177.81 |               49.5    |       10.825   |           17.4 |        83.3|           83.3 |
| huggingface |               10 |             90 |     1.48835  |            6.71885 |          60.4697 |   0.230774 |  1.4504   | 0.152453   |    51.6    |             65 |             7260.2  |             7268.81 |               43.2    |        9.5     |           14.3 |        83.36|           83.9 |
| ollama      |               10 |             90 |     0.561705 |           17.8029  |         160.226  |   0.285564 |  0.312152 | 0.00332347 |    49.6667 |             72 |             7265.96 |             7268.62 |               42.6667 |       11.85    |           18.1 |        84.1333|           84.7 |
| huggingface |               15 |            135 |     2.65428  |            5.65125 |          50.8612 |   0.378228 |  2.58419  | 0.275745   |    51      |             81 |             7364.86 |             7373.75 |               41.8462 |        9.93077 |           12.1 |        83.9231|           84.1 |
| ollama      |               15 |            135 |     0.831616 |           18.0372  |         162.334  |   0.418053 |  0.44443  | 0.00329703 |    49.5    |             81 |             7369.75 |             7369.75 |               42.125  |       12.8625  |           16.7 |        83.8|           83.8 |
| huggingface |               20 |            180 |     3.61361  |            5.53464 |          49.8117 |   0.53425  |  3.53066  | 0.374551   |    52.9722 |             81 |             7458.19 |             7463.75 |               44      |       11.3639  |           33.3 |        83.7|           83.8 |
| ollama      |               20 |            180 |     1.11282  |           17.9724  |         161.752  |   0.562839 |  0.589073 | 0.00327932 |    53.9    |             83 |             7462.59 |             7462.69 |               45.7    |       12.96    |           16.2 |        83.52|           83.6 |

## Contributing

Contributions are welcome! If you'd like to improve the project, fix bugs, or suggest new features, feel free to contribute.

## Show Your Support

If you find this project useful, please consider giving it a star on GitHub. It helps increase the project's visibility and motivates further development.

If you have suggestions, encounter bugs, or want to contribute, feel free to open an issue or submit a pull request.

Thank you for your support!