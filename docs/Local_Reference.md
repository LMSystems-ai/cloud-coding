# cloudcode Python SDK — `Local` Class Reference

The `Local` class is the primary entry point for using the cloudcode (Aider) Python SDK in your projects. It provides methods to run AI-assisted coding tasks, manage files, track costs/credits, and interact with the cloudcode API.

---

## Class: `Local`

```python
class Local:
    def __init__(
        self,
        working_dir: str,
        model: str = "gpt-4.1-nano",
        editor_model: Optional[str] = None,
        use_git: bool = False,
        api_key: Optional[str] = None,
        api_keys: Optional[Dict[str, str]] = None,
        architect_mode: bool = False,
        weak_model: Optional[str] = None,
        session_id: Optional[str] = None
    )
```

### Constructor Parameters

- **working_dir** (`str`, required)
  Directory where file operations and (optionally) git commands will run.
  Must be a valid path; if `use_git=True`, it must be inside a git repo.

- **model** (`str`, default=`"gpt-4.1-nano"`)
  The primary AI model to use for coding and editing tasks.
  Supports OpenAI and Anthropic model names or aliases (see `list_models()`).

- **editor_model** (`str`, optional)
  A separate model for editing operations. If omitted, `model` is used for both planning and editing.

- **use_git** (`bool`, default=`False`)
  If `True`, changes are staged and diffs are generated via git. Otherwise, file contents are returned without version control.

- **api_key** (`str`, optional)
  Your cloudcode (Aider backend) API key. If provided, the SDK will authenticate against cloudcode, fetch provider keys, and enable credit tracking.

- **api_keys** (`Dict[str,str]`, optional)
  Legacy method: supply provider API keys directly, e.g. `{"openai": "sk-...", "anthropic": "ak-..."}`.
  Only `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are recognized.

- **architect_mode** (`bool`, default=`False`)
  Enables a two-model “planner + editor” workflow.
  Planner uses `weak_model` (or `model`), and editor uses `editor_model` (or `model`).

- **weak_model** (`str`, optional)
  In architect mode, use this weaker “planner” model instead of `model`.

- **session_id** (`str`, optional)
  Custom session identifier for usage logging. If omitted, a UUID is auto‑generated.

---

## Public Methods

### list_models(substring: str = "") → `List[str]`

Return a sorted list of supported model names and aliases.
- **substring**: filter to models/aliases containing this (case‑insensitive).

### code(
    prompt: str,
    editable_files: List[str],
    readonly_files: Optional[List[str]] = None
) → `Dict[str, Any]`

Perform an AI coding task:

- **prompt**: Natural‑language instruction.
- **editable_files**: Paths (relative or absolute) the AI may modify.
- **readonly_files**: Paths the AI may read but not change.

Returns a dictionary including:
- `success` (`bool`): whether meaningful changes were made.
- `diff` (`str`): git diff or placeholder message.
- `result`: the raw result object from the underlying coder.
- `cost`: extracted cost info (`message_cost`, `session_cost`, `tokens` etc.).
- `marked_up_cost`: cost with 20% markup.
- `credits_remaining`: your remaining cloudcode credits (if authenticated).

Raises on critical errors (e.g. invalid model, insufficient credits).

### create_file(file_path: str, content: str) → `bool`

Create or overwrite a file in `working_dir`.
- Creates parent directories as needed.
- Returns `True` on success, `False` otherwise.

### read_file(file_path: str) → `Optional[str]`

Read and return the text content of `file_path` (relative to `working_dir`).
Returns `None` if the file does not exist or cannot be read.

### search_files(
    query: str,
    file_patterns: Optional[List[str]] = None
) → `Dict[str, List[str]]`

Search for lines containing `query` across files:

- **query**: substring or regex to match.
- **file_patterns**: glob patterns (default `["**/*"]`).

Returns a mapping of relative file paths to lists of matching lines.

### code_headless(
    prompt: str,
    editable_files: List[str],
    readonly_files: Optional[List[str]] = None,
    task_id: Optional[str] = None
) → `Dict[str, Any]`

Start an asynchronous (headless) coding task:

- Does not block on AI completion.
- Returns immediately with `{"task_id": ..., "status": "pending", "architect_mode": ...}`.
- The background thread will update `self._headless_tasks[task_id]` on completion.

### get_headless_task_status(task_id: str) → `Dict[str, Any]`

Query the status and result of a previously started headless task.
Possible statuses: `"pending"`, `"completed"`, `"failed"`, `"not_found"`.

### get_cost_history() → `List[Dict[str, Any]]`

Return a chronological list of cost entries from each `code()` call:
```json
[
  {
    "timestamp": "...",
    "prompt": "...",
    "files": ["..."],
    "cost": { "message_cost": ..., "session_cost": ..., ... },
    "marked_up_cost": { ... },
    "architect_mode": false
  },
  ...
]
```

### get_total_cost(
    include_current: bool = False,
    current_cost: Optional[Dict[str, float]] = None,
    include_markup: bool = True
) → `Dict[str, float]`

Compute aggregate costs:

- **include_current**: include `current_cost` in the totals.
- **current_cost**: a cost dict from a just‑completed run.
- **include_markup**: apply the 20% markup to all summed costs.

Returns:
```json
{
  "total_message_cost": ...,
  "total_session_cost": ...,
  "total_cost": ...
}
```

### get_credit_info() → `Dict[str, float]`

Fetch or refresh your cloudcode credits and limits:
```json
{
  "credits":           <remaining>,
  "credit_limit":      <max>,
  "credits_used":      <limit - remaining>
}
```

### check_credits_sufficient(estimated_cost: float = 0.0) → `bool`

Return `True` if you have at least `estimated_cost` credits remaining; otherwise `False`.

### buy_credits(amount: float) → `Dict[str, Any]`

Prepare to purchase additional credits (via Stripe):

- **amount**: dollars to add (minimum \$5.00).
- Returns payment configuration:
```json
{
  "stripe_publishable_key": "...",
  "api_url": ".../createPaymentIntent",
  "amount": <dollars>,
  "credits": <credits_added>,
  "session_token": "..."
}
```

### get_payment_history() → `List[Dict[str, Any]]`

Fetch your past Stripe payment records:
```json
[
  { "id": "...", "amount": ..., "timestamp": "...", ... },
  ...
]
```

---

## Quick Start Examples

### 1. Minimal Example
```python
from cloudcode import Local
import os

# Initialize the SDK (ensure api_key is set in your env)
sdk = Local(
    working_dir="/path/to/your/project",
    api_key=os.getenv("CLOUDCODE_API_KEY")
)

# Create a simple file and run a coding task
sdk.create_file(
    "src/calculator.py",
    "def add(a, b): return a + b\n"
)
result = sdk.code(
    prompt="Add multiply and divide functions to calculator.py",
    editable_files=["src/calculator.py"]
)
print(result["diff"])
```

### 2. Interactive Example
```python
from cloudcode import Local
import os

sdk = Local(
    working_dir="/Users/you/myrepo",
    api_key=os.getenv("CLOUDCODE_API_KEY")
)

# Create and inspect files
sdk.create_file(
    "src/calculator.py",
    """def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""
)
print("Before:", sdk.read_file("src/calculator.py"))
print("Search for 'add':", sdk.search_files("def add", ["src/*.py"]))

# Improve calculator with error handling
response = sdk.code(
    prompt="Add multiply and divide functions to the calculator. "
           "Ensure divide handles division by zero.",
    editable_files=["src/calculator.py"]
)
if response["success"]:
    print("Diff:\n", response["diff"])
    print("After:\n", sdk.read_file("src/calculator.py"))
```

### 3. Headless (Asynchronous) Example
```python
import time
from cloudcode import Local
import os

sdk = Local(
    working_dir="/Users/you/myrepo",
    use_git=False,
    api_key=os.getenv("CLOUDCODE_API_KEY")
)

# Start a background coding task
task = sdk.code_headless(
    prompt="Add comments to document the add and subtract functions.",
    editable_files=["src/calculator.py"]
)
task_id = task["task_id"]

# Poll until it's done
while True:
    status = sdk.get_headless_task_status(task_id)
    if status["status"] == "completed":
        print("Diff:\n", status["result"]["diff"])
        break
    elif status["status"] == "failed":
        print("Error:", status.get("error"))
        break
    time.sleep(2)
```

---

## Usage Tips

- Always call `list_models()` to discover valid model names and aliases before passing to `model` or `editor_model`.
- If you set `use_git=True`, ensure your `working_dir` is a git repo; diffs and rollback become much simpler.
- For long‑running or background tasks, prefer `code_headless()` + `get_headless_task_status()`.
- In “architect mode” you can designate a cheaper `weak_model` for planning; the editor still uses your premium `editor_model`.
- Track costs via `get_cost_history()` and `get_total_cost()` to monitor spending, and check `credits_remaining` to avoid unexpected failures.

---

_End of SDK Reference_
