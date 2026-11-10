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

| Model       |   total_requests |   total_tokens |   total_time (s) |   requests_per_sec |   tokens_per_sec |   avg_ttft (s) |   avg_e2e (s) |   avg_itl (s) |   avg_gpu_util (%) |   max_gpu_util (%) |   avg_gpu_memory_used (MB) |   max_gpu_memory_used (MB) |   avg_gpu_memory_util (%) |   avg_cpu_util (%) |   max_cpu_util (%) |   avg_ram_util (%) |   max_ram_util (%) |
|-------------|------------------|----------------|------------------|--------------------|------------------|----------------|---------------|---------------|--------------------|--------------------|----------------------------|----------------------------|---------------------------|--------------------|--------------------|--------------------|--------------------|
| huggingface |                5 |             45 |         0.887048 |            5.63667 |          50.73   |       0.42593  |      0.858716 |    0.0540983  |            32.2222 |                 80 |                    5803.77 |                    5837.56 |                   13.4444 |   7.03333 |                9.3 |            80.5778 |               81.2 |
| ollama      |                5 |             45 |         2.19585  |            2.27702 |          20.4932 |       2.04178  |      2.06917  |    0.00342382 |            18.2273 |                 81 |                    6369.47 |                    7168.56 |                   11.5    |   5.53636 |               17.5 |            81.7455 |               82.4 |
| huggingface |               10 |             90 |         1.38372  |            7.2269  |          65.0421 |       0.218734 |      1.35345  |    0.14184    |            51.2143 |                 74 |                    7251.56 |                    7260.56 |                   43.3571 |  15.85    |               35.4 |            82.4    |               82.4 |
| ollama      |               10 |             90 |         0.550929 |           18.1512  |         163.361  |       0.277258 |      0.303087 |    0.00322857 |            60      |                 76 |                    7260.54 |                    7260.56 |                   53.6667 |  11.0667  |               16.1 |            82.4    |               82.4 |
| huggingface |               15 |            135 |         2.549    |            5.88466 |          52.9619 |       0.333746 |      2.48774  |    0.269249   |            50.76   |                 80 |                    7366.04 |                    7372.44 |                   44.36   |   9.104   |               11   |            82.4    |               82.4 |
| ollama      |               15 |            135 |         0.835866 |           17.9455  |         161.509  |       0.423583 |      0.449608 |    0.00325311 |            46.375  |                 81 |                    7372.44 |                    7372.44 |                   41      |  12.35    |               16.3 |            82.4    |               82.4 |
| huggingface |               20 |            180 |         3.45225  |            5.79333 |          52.14   |       0.475836 |      3.36457  |    0.361092   |            48.5294 |                 79 |                    7464.69 |                    7472.94 |                   42.4706 |   9.30882 |               12.6 |            82.4059 |               82.5 |
| ollama      |               20 |            180 |         1.10441  |           18.1093  |         162.983  |       0.557573 |      0.583549 |    0.003247   |            49.1    |                 80 |                    7472.94 |                    7472.94 |                   43.2    |  12.81    |               16.3 |            82.49   |               82.5 |


#### Ollama Warm Start

| Model       |   total_requests |   total_tokens |   total_time (s) |   requests_per_sec |   tokens_per_sec |   avg_ttft (s) |   avg_e2e (s) |   avg_itl (s) |   avg_gpu_util (%) |   max_gpu_util (%) |   avg_gpu_memory_used (MB) |   max_gpu_memory_used (MB) |   avg_gpu_memory_util (%) |   avg_cpu_util (%) |   max_cpu_util (%) |   avg_ram_util (%) |   max_ram_util (%) |
|-------------|------------------|----------------|------------------|--------------------|------------------|----------------|---------------|---------------|--------------------|--------------------|----------------------------|----------------------------|---------------------------|--------------------|--------------------|--------------------|--------------------|
| huggingface |                5 |             45 |         0.892541 |            5.60198 |          50.4178 |       0.429042 |      0.855132 |    0.0532612  |            17.5556 |                 50 |                    7173.49 |                    7211.38 |                   9.55556 |   7.57778 |                9.6 |            82.2333 |               82.9 |
| ollama      |                5 |             45 |         0.309376 |           16.1616  |         145.454  |       0.15097  |      0.177139 |    0.00327123 |            66.25   |                 82 |                    7196    |                    7196    |                  57.5     |   9.475   |               16.8 |            82.9    |               82.9 |
| huggingface |               10 |             90 |         1.54891  |            6.45615 |          58.1054 |       0.218015 |      1.50383  |    0.160727   |            51.6875 |                 65 |                    7278.2  |                    7285.81 |                  43.25    |   9.2125  |               11.4 |            82.8625 |               82.9 |
| ollama      |               10 |             90 |         0.562872 |           17.766   |         159.894  |       0.280089 |      0.306558 |    0.00330865 |            44      |                 80 |                    7285.81 |                    7285.81 |                  37.6667  |  11.4167  |               16.7 |            82.8    |               82.8 |
| huggingface |               15 |            135 |         2.65107  |            5.65809 |          50.9229 |       0.386518 |      2.59711  |    0.276324   |            50.2692 |                 80 |                    7387.39 |                    7395.19 |                  41.5385  |   9.95769 |               18.1 |            82.8    |               82.8 |
| ollama      |               15 |            135 |         0.832505 |           18.0179  |         162.161  |       0.418294 |      0.444686 |    0.00329907 |            49.5    |                 80 |                    7393.81 |                    7393.81 |                  42       |  12.5125  |               16.5 |            82.8    |               82.8 |
| huggingface |               20 |            180 |         4.05485  |            4.93237 |          44.3913 |       0.503831 |      3.96831  |    0.43306    |            67.25   |                100 |                    7484.66 |                    7489.88 |                  48.7     |   9.3175  |               11.4 |            82.8    |               82.8 |
| ollama      |               20 |            180 |         2.09271  |            9.557   |          86.013  |       1.06791  |      1.12909  |    0.00764726 |            80.25   |                 94 |                    7479.36 |                    7488.94 |                  42.05    |  16.12    |               31.4 |            82.72   |               82.8 |

## Contributing

Contributions are welcome! If you'd like to improve the project, fix bugs, or suggest new features, feel free to contribute.

## Show Your Support

If you find this project useful, please consider giving it a star on GitHub. It helps increase the project's visibility and motivates further development.

If you have suggestions, encounter bugs, or want to contribute, feel free to open an issue or submit a pull request.

Thank you for your support!