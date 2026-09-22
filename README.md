# Personal Assistant
A privacy-focused personal AI assistant with long-term memory, semantic retrieval, and local-first data storage, designed to evolve into a private multimodal assistant.

## Python
The Python code is located in the python/ directory.

### Prequisites
* Python 3.11+
* uv

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
uv sync --extra dev
```

This installs the dependencies defined in pyproject.toml and creates/updates uv.lock.

### 6. Verify the installation

#### Run the test suite
```
uv run pytest
```

#### Run linting
```
uv run ruff check .
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
