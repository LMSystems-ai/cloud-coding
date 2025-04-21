# cloudcode Python SDK

A simple Python SDK for AI-powered coding assistance in your scripts and applications.

## Installation

```bash
pip install cloudcode
```

### API Key

Every user gets $5 free credit with no credit card required
Get your key [here](https://cloudcoding.ai/home)
```bash
export CLOUD_CODE_API_KEY=...
```

check out our [Docs](https://docs.cloudcoding.ai/introduction)

## Quick Start

```python
from cloudcode import Local
import os

# Initialize the SDK
sdk = Local(
    working_dir="/path/to/your/project",
    api_key=os.getenv("CLOUD_CODE_API_KEY")
)

# Have AI modify your code
result = sdk.code(
    prompt="What's this project about?"
)

print(f"Success: {result['success']}")
print(result['diff'])
```

## Key Features

- AI-powered code generation and modification
- Only supporting OpenAI and Anthropic models currently
- File operations (create, read, search)
- Works with or without git repositories
- Built in [Remote Sandbox](https://github.com/LMSystems-ai/cloud-coding/blob/main/docs/sandbox.md) feature

## Example Usage

Here's a simple example showing the main capabilities:

```python
from cloudcode import Local
import os

cwd = os.getcwd()

# Initialize the SDK
sdk = Local(
    working_dir=cwd,
    model="o4-mini",
    editor_model="gpt-4.1",
    architect_mode=True,
    api_key=os.getenv("CLOUD_CODE_API_KEY")
)

# Create a file
sdk.create_file(
    "calculator.py",
    """def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""
)

# Have AI improve the code
result = sdk.code(
    prompt="Add multiply and divide functions to calculator.py. Handle division by zero.",
    editable_files=["calculator.py"]
)

if result["success"]:
    print("AI successfully modified the code!")
    print(result["diff"])

# Read the updated file
updated_content = sdk.read_file("calculator.py")
print(updated_content)
```

See `example_usage.py` for a complete working example.

## API Reference

### Initialization

```python
sdk = Local(
    working_dir="/path/to/your/project",  # Required
    model="o4-mini",
    editor_model="gpt-4.1",
    architect_mode=True,                  # Optional: specify editor model to enable planner/editor mode with 2 LLMs working together
    use_git=True,                         # Optional: set to False to disable git (this helps our agent understand the codebase better when git is enabled)
    api_key=os.getenv("CLOUD_CODE_API_KEY")
)
```

### Core Functions

# cloudcode SDK Documentation

cloudcode SDK is a Python-based software development kit that provides programmatic access to cloudcode, an AI-powered coding assistant. This SDK enables developers to easily integrate AI-powered code generation, modification, and analysis capabilities into their tools and workflows.

## Table of Contents

- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Local Class](#Local-class)
  - [Initialization](#initialization)
  - [List Available Models](#list-available-models)
  - [Run AI Coding Task](#run-ai-coding-task)
  - [File Operations](#file-operations)
  - [Headless Operations](#headless-operations)
- [SandboxSDK Class](#SandboxSDK-class)
  - [Initialization](#sandbox-initialization)
  - [Sandbox File Operations](#sandbox-file-operations)
  - [Running Commands](#running-commands)
  - [Running AI Coding Tasks](#running-ai-coding-tasks)
  - [Sandbox Management](#sandbox-management)
- [Examples](#examples)
  - [Basic Usage](#basic-usage-example)
  - [Headless Operation](#headless-operation-example)
  - [Sandbox Usage](#sandbox-usage-example)

## Installation

```bash
pip install cloudcode
```

## Basic Usage

```python
from cloudcode import Local
import os

# Initialize the SDK in Architect Mode
agent = Local(
    working_dir=cwd,
    model="o4-mini",  # Main (planner) model
    editor_model="gpt-4.1",  # Editor model for implementing changes
    architect_mode=True,
    api_key=os.getenv("CLOUD_CODE_API_KEY")
)

# Run an AI coding task
result = sdk.code(
    prompt="Add a multiply function to the calculator module",
    editable_files=["src/calculator.py"],
    readonly_files=["src/tests/test_calculator.py"]
)
```

## Local Class

The base `Local` class provides functionality for using our agent to perform coding tasks in a local environment.

### Initialization

```python
Local(
    working_dir: str,
    model: str = "gpt-4.1",
    editor_model: Optional[str] = None,
    use_git: bool = True,
    api_key: Optional[str] = None
)
```

**Parameters:**

- `working_dir`: Path to the project directory where operations will occur
- `model`: The AI model to use for coding tasks (default: "gpt-4.1")
- `editor_model`: Optional separate model for editing operations
- `use_git`: Whether to use git for tracking changes (default: True)
- `api_key`: API key for various providers

**Example:**

```python
sdk = Local(
    working_dir="/path/to/project/",
    model="gpt-4.1",
    use_git=True,
    api_key="sk-..."
)
```

### List Available Models

```python
list_models(substring: str = "") -> List[str]
```

List available AI models that match the provided substring.

**Parameters:**
- `substring`: String to match against available model names

**Returns:**
- List of model name strings that match the provided substring

**Example:**

```python
openai_models = sdk.list_models("openai")
print("Available OpenAI models:", openai_models)
```

### Run AI Coding Task

```python
code(
    prompt: str,
    editable_files: List[str],
    readonly_files: List[str] = None,
) -> Dict[str, Any]
```

Run an AI coding task with the specified prompt and files.

**Parameters:**
- `prompt`: Natural language instruction for the AI coding task
- `editable_files`: List of files that can be modified by the AI
- `readonly_files`: List of files that can be read but not modified

**Returns:**
- Dictionary with 'success' boolean, 'diff' string showing changes, and 'result' data

**Example:**

```python
result = sdk.code(
    prompt="Add a multiply and divide function to the calculator.py file",
    editable_files=["src/calculator.py"],
    readonly_files=["src/tests/test_calculator.py"]
)

if result["success"]:
    print("Changes made:", result["diff"])
```

### File Operations

#### Create File

```python
create_file(file_path: str, content: str) -> bool
```

Create a new file with the specified content.

**Parameters:**
- `file_path`: Path to the file to create (relative to working_dir)
- `content`: Content to write to the file

**Returns:**
- True if successful, False otherwise

**Example:**

```python
sdk.create_file(
    "src/calculator.py",
    """def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""
)
```

#### Read File

```python
read_file(file_path: str) -> Optional[str]
```

Read the content of a file.

**Parameters:**
- `file_path`: Path to the file to read (relative to working_dir)

**Returns:**
- Content of the file, or None if the file doesn't exist

**Example:**

```python
content = sdk.read_file("src/calculator.py")
print("File content:", content)
```

#### Search Files

```python
search_files(query: str, file_patterns: List[str] = None) -> Dict[str, List[str]]
```

Search for matches in files.

**Parameters:**
- `query`: String to search for
- `file_patterns`: List of glob patterns to limit the search to

**Returns:**
- Dictionary with file paths as keys and lists of matching lines as values

**Example:**

```python
results = sdk.search_files("def add", ["src/*.py"])
for file_path, lines in results.items():
    print(f"Matches in {file_path}:")
    for line in lines:
        print(f"  {line}")
```

### Headless Operations

#### Run AI Coding Task in Headless Mode

```python
code_headless(
    prompt: str,
    editable_files: List[str],
    readonly_files: List[str] = None,
    task_id: str = None
) -> Dict[str, Any]
```

Run an AI coding task in headless mode without waiting for the result.

**Parameters:**
- `prompt`: Natural language instruction for the AI coding task
- `editable_files`: List of files that can be modified by the AI
- `readonly_files`: List of files that can be read but not modified
- `task_id`: Optional identifier for the task (auto-generated if None)

**Returns:**
- Dictionary with 'task_id' string to identify the task and 'status' string

**Example:**

```python
task = sdk.code_headless(
    prompt="Add error handling to the calculator functions",
    editable_files=["src/calculator.py"],
    readonly_files=[],
    task_id="task-123"
)

print(f"Task started with ID: {task['task_id']}")
```

#### Get Headless Task Status

```python
get_headless_task_status(task_id: str) -> Dict[str, Any]
```

Get the status of a headless coding task.

**Parameters:**
- `task_id`: The ID of the task to check

**Returns:**
- Dictionary with task status information

**Example:**

```python
status = sdk.get_headless_task_status("task-123")
print(f"Task status: {status['status']}")

if status["status"] == "completed":
    print("Changes made:", status["result"]["diff"])
```

## SandboxSDK Class

The `SandboxSDK` class extends the base `Local` class to operate within a sandbox environment, enabling isolated code execution and testing.

### Sandbox Initialization

```python
SandboxSDK(
    model: str = "gpt-4.1",
    editor_model: Optional[str] = None,
    api_key: Optional[str] = None,
    sandbox_timeout: int = 300,
    sandbox_id: Optional[str] = None,
    user_id: Optional[str] = None,
)
```

**Parameters:**
- `model`: The AI model to use for coding tasks (default: "gpt-4.1")
- `editor_model`: Optional separate model for editing operations
- `api_key`: API key for various providers
- `sandbox_timeout`: Timeout in seconds for the sandbox (default: 300 seconds)
- `sandbox_id`: ID of existing sandbox to connect to (optional)
- `user_id`: User ID for tracking and persistence (optional)

**Example:**

```python
sandbox_sdk = SandboxSDK(
    model="gpt-4.1",
    api_key="sk-...",
    sandbox_timeout=600,
    user_id="user123"
)
```

### Sandbox File Operations

#### Upload File

```python
upload_file(local_path: str, sandbox_path: Optional[str] = None) -> str
```

Upload a local file to the sandbox.

**Parameters:**
- `local_path`: Path to local file
- `sandbox_path`: Path in sandbox (defaults to same filename in working_dir)

**Returns:**
- Path to the file in the sandbox

**Example:**

```python
sandbox_path = sandbox_sdk.upload_file("local/path/to/file.py", "/home/user/file.py")
print(f"File uploaded to: {sandbox_path}")
```

#### Write to Sandbox

```python
write_to_sandbox(
    content: Union[str, bytes, List[Dict[str, Union[str, bytes]]], str],
    path: Optional[str] = None,
    local_directory: Optional[str] = None,
    sandbox_directory: Optional[str] = None
) -> List[str]
```

Write file(s) to the sandbox filesystem. Supports single files, multiple files, or entire directories.

**Parameters:**
- `content`: File content or list of file objects with 'path' and 'data' keys, or ignored if local_directory is provided
- `path`: Path in the sandbox for a single file upload (required if content is str/bytes)
- `local_directory`: Local directory path containing files to upload
- `sandbox_directory`: Target directory in sandbox for directory uploads (defaults to working_dir)

**Returns:**
- List of paths written to the sandbox

**Example:**

```python
# Single file
paths = sandbox_sdk.write_to_sandbox(
    content="def hello(): return 'world'",
    path="/home/user/hello.py"
)

# Multiple files
paths = sandbox_sdk.write_to_sandbox([
    {"path": "/home/user/file1.py", "data": "print('hello')"},
    {"path": "/home/user/file2.py", "data": "print('world')"}
])

# Directory
paths = sandbox_sdk.write_to_sandbox(
    content="",  # Ignored
    local_directory="/local/project/src",
    sandbox_directory="/home/user/src"
)
```

#### Download File

```python
download_file(sandbox_path: str, local_path: Optional[str] = None) -> str
```

Download a file from the sandbox to local filesystem.

**Parameters:**
- `sandbox_path`: Path to file in sandbox
- `local_path`: Path to download to (defaults to same filename)

**Returns:**
- Path to the downloaded file

**Example:**

```python
local_path = sandbox_sdk.download_file("/home/user/output.py", "local/output.py")
print(f"File downloaded to: {local_path}")
```

#### Read Sandbox File

```python
read_sandbox_file(sandbox_path: str, as_string: bool = True, encoding: str = "utf-8") -> Union[str, bytes]
```

Read a file from the sandbox.

**Parameters:**
- `sandbox_path`: Path to the file in the sandbox
- `as_string`: Whether to return the content as a string (True) or bytes (False)
- `encoding`: Encoding to use when converting bytes to string (default: utf-8)

**Returns:**
- File content as string or bytes depending on as_string parameter

**Example:**
```python
content = sandbox_sdk.read_sandbox_file("/home/user/calculator.py")
print(f"File content: {content}")

# Get content as bytes
binary_content = sandbox_sdk.read_sandbox_file("/home/user/image.png", as_string=False)
```

### Running Commands

```python
run_command(command: str) -> Dict[str, Any]
```

Run a command in the sandbox.

**Parameters:**
- `command`: Command to run

**Returns:**
- Dictionary with command result info (exit_code, stdout, stderr)

**Example:**

```python
result = sandbox_sdk.run_command("python3 -m pytest test_calculator.py -v")
print(f"Exit code: {result['exit_code']}")
print(f"Output: {result['stdout']}")
if result['stderr']:
    print(f"Errors: {result['stderr']}")
```

### Running AI Coding Tasks

```python
sandbox_code(
    prompt: str,
    editable_files: List[str],
    readonly_files: List[str] = None,
) -> Dict[str, Any]
```

Run an AI coding task in the sandbox with the specified prompt and files.

**Parameters:**
- `prompt`: Natural language instruction for the AI coding task
- `editable_files`: List of files in the sandbox that can be modified by the AI
- `readonly_files`: List of files in the sandbox that can be read but not modified

**Returns:**
- Dictionary with 'success' boolean and 'diff' string showing changes

**Example:**

```python
result = sandbox_sdk.sandbox_code(
    prompt="Optimize the calculator functions for performance",
    editable_files=["/home/user/calculator.py"],
    readonly_files=["/home/user/tests/test_calculator.py"]
)

if result["success"]:
    print("Changes made:", result["diff"])
```

### Sandbox Management

#### Extend Sandbox Timeout

```python
extend_sandbox_timeout(seconds: int = 300) -> None
```

Extend the sandbox timeout.

**Parameters:**
- `seconds`: Number of seconds to extend the timeout by

**Example:**

```python
# Add 10 more minutes to the sandbox timeout
sandbox_sdk.extend_sandbox_timeout(600)
```

#### Get Sandbox Info

```python
get_sandbox_info() -> Dict[str, Any]
```

Get information about the current sandbox.

**Returns:**
- Dictionary with sandbox information (sandbox_id, template_id, started_at, end_at, metadata)

**Example:**

```python
info = sandbox_sdk.get_sandbox_info()
print(f"Sandbox ID: {info['sandbox_id']}")
print(f"Created at: {info['started_at']}")
print(f"Expires at: {info['end_at']}")
```

#### Kill Sandbox

```python
kill_sandbox() -> Dict[str, Any]
```

Shutdown the current sandbox.

**Returns:**
- Dictionary with kill status information

**Example:**

```python
result = sandbox_sdk.kill_sandbox()
print(f"Sandbox termination: {result['message']}")
print(f"Termination success: {result['success']}")
```

## Examples

### Basic Usage Example

```python
from cloudcode import Local
import os

sdk = Local(
    working_dir="/path/to/project/",
    model="gpt-4.1",
    api_key=os.getenv("CLOUD_CODE_API_KEY")
)

# Create a file
sdk.create_file(
    "calculator.py",
    """def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""
)

# Use AI to improve the code
result = sdk.code(
    prompt="Add multiply and divide functions to calculator.py",
    editable_files=["calculator.py"]
)

# View the updated file
updated_code = sdk.read_file("calculator.py")
print(updated_code)
```

### Headless Operation Example

```python
import time
from cloudcode import Local
import os

sdk = Local(
    working_dir="/path/to/project/",
    model="gpt-4.1",
    api_key=os.getenv("CLOUD_CODE_API_KEY")
)

# Start a headless task
task = sdk.code_headless(
    prompt="Add error handling to all functions",
    editable_files=["calculator.py"]
)

task_id = task["task_id"]
print(f"Task started with ID: {task_id}")

# Poll for completion
while True:
    time.sleep(2)
    status = sdk.get_headless_task_status(task_id)

    if status["status"] == "completed":
        print("Task completed!")
        print(status["result"]["diff"])
        break
    elif status["status"] == "failed":
        print(f"Task failed: {status.get('error')}")
        break
    else:
        print("Task still running...")
```

### Sandbox Usage Example

```python
from cloudcode import SandboxSDK
import os

sdk = SandboxSDK(
    model="gpt-4.1",
    api_key=os.getenv("CLOUD_CODE_API_KEY"),
    sandbox_timeout=600,
    user_id="user123"
)

# Get sandbox info
info = sdk.get_sandbox_info()
print(f"Sandbox ID: {info['sandbox_id']}")

# Create a file in the sandbox
sdk.create_file(
    "calculator.py",
    """def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""
)

# Upload a directory to the sandbox
sdk.write_to_sandbox(
    content="",
    local_directory="/local/test_scripts",
    sandbox_directory="/home/user/tests"
)

# Run a command in the sandbox
result = sdk.run_command("python3 -c 'import calculator; print(calculator.add(5, 3))'")
print(f"Command output: {result['stdout']}")

# Use AI to improve the code
result = sdk.sandbox_code(
    prompt="Add a multiply function",
    editable_files=["/home/user/calculator.py"]
)

# When done, extend timeout or kill the sandbox
sdk.extend_sandbox_timeout(600)  # Add 10 minutes
# Or kill the sandbox completely
kill_result = sdk.kill_sandbox()
print(f"Sandbox termination: {kill_result['message']}")
```

For more detailed examples, see the example scripts:
- [example_usage.py](./example_usage.py) - Basic SDK usage
- [headless_example.py](./headless_example.py) - Headless operation
- [sandbox_example.py](./sandbox_example.py) - Sandbox environment usage
