# Personal Assistant
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![GitHub License](https://img.shields.io/github/license/VijayVignesh1/personal_assistant?style=for-the-badge)
![GitHub Stars](https://img.shields.io/github/stars/VijayVignesh1/personal_assistant?style=for-the-badge)
![GitHub Issues](https://img.shields.io/github/issues/VijayVignesh1/personal_assistant?style=for-the-badge)
![Pull Requests](https://img.shields.io/github/issues-pr/VijayVignesh1/personal_assistant?style=for-the-badge)
A privacy-focused personal AI assistant with long-term memory, semantic retrieval, and local-first data storage, designed to evolve into a private multimodal assistant.

## Python Installation
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


### Starting the chat

1. Launch the chainlit application using:
```
chainlit run main.py -w
```
2. It should open the browser automatically. If not, navigate to `http://localhost:8000/`