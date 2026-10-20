# Personal Assistant
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

### 2. Go to the python project
```
cd python
```

### 3. Create the virtual environent
```
uv venv --python 3.11
```

### 4. Activate the virtual environment
#### macOS / Linux
```
source .venv/bin/activate
```

#### Windows:
```
.venv\Scripts\activate
```

### 5. Install dependencies

Install the project and development dependencies:

```
uv sync
```

This installs the dependencies defined in pyproject.toml and creates/updates uv.lock.

NOTE: In order to run the models on GPU CUDA, make sure to install the pytorch version compatible with the underlying CUDA version. To find the cuda version, run `nvidia-smi`. For example, if the cuda version is 12.5, install any pytorch version with cuda 12.5 and below as shown below:
```
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```


### 6. Verify the installation

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

### 7. Adding a Package

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


### 8. Starting the chat

1. Execute the main program
```
python main.py
```
2. Open the launched gradio UI using the given local URL. For example, `http://127.0.0.1:7860`


### Configure Google Gemini for DeepEval

DeepEval supports Google Gemini for generating synthetic evaluation datasets.

### 1. Install dependencies

#### Step 1: Create a Gemini API Key

1. Open [Google AI Studio](https://aistudio.google.com/).
2. Sign in with your Google account.
3. Navigate to the **API Keys** page in the dashboard, or open
   [Google AI Studio API Keys](https://aistudio.google.com/apikey).
4. Click **Create API Key**.
5. Select an existing Google Cloud project or create/import a project if prompted.
6. Copy the generated API key.

See the [official Gemini API documentation](https://ai.google.dev/gemini-api/docs/api-key) for details.

#### Step 2: Configure the API Key

```
export GOOGLE_API_KEY=gemini-api-key
```

#### Step 3: Run the evals

```
python tests/evals/eval.py
```