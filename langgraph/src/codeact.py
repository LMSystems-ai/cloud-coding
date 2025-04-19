#!/usr/bin/env python
"""
LMSYS LangGraph App - AI-assisted coding using LangGraph and LMSYS SDK
"""

import builtins
import contextlib
import io
import os
import subprocess
from typing import Any, Dict, List, Optional

from langchain.chat_models import init_chat_model
from langgraph_codeact import create_codeact

# Import Aider SDK
from lmsys import Local

# Default working directory if none provided
DEFAULT_WORKING_DIR = os.getcwd()

def is_git_repository(path: str) -> bool:
    """Check if the given directory is a git repository."""
    git_dir = os.path.join(path, '.git')

    # Check if the .git directory exists
    if os.path.isdir(git_dir):
        return True

    # Try running git command as a fallback
    try:
        result = subprocess.run(
            ['git', '-C', path, 'rev-parse', '--is-inside-work-tree'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        return result.returncode == 0 and result.stdout.strip() == 'true'
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

# SDK cache to prevent repeated initialization
_sdk_cache = {}

def get_sdk(working_dir: str = os.getcwd()) -> Local:
    """Get or create an SDK instance for the given working directory."""
    # Use the working directory as the cache key
    if working_dir not in _sdk_cache:
        # Check if the directory is a git repository
        use_git = is_git_repository(working_dir)

        # Initialize SDK with the provided working directory
        _sdk_cache[working_dir] = Local(
            working_dir=working_dir,
            model="gpt-4.1",
            use_git=use_git,
            lmsys_api_key=os.getenv("LMSYS_API_KEY")
        )

    return _sdk_cache[working_dir]

# Initialize default SDK
sdk = get_sdk()

def eval(code: str, _locals: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Execute code and capture its output and new variables."""
    # Store original keys before execution
    original_keys = set(_locals.keys())

    try:
        with contextlib.redirect_stdout(io.StringIO()) as f:
            exec(code, builtins.__dict__, _locals)
        result = f.getvalue()
        if not result:
            result = "<code ran, no output printed to stdout>"
    except Exception as e:
        result = f"Error during execution: {repr(e)}"

    # Determine new variables created during execution
    new_keys = set(_locals.keys()) - original_keys
    new_vars = {key: _locals[key] for key in new_keys}
    return result, new_vars

# Create tool functions for the agent that take working_dir from state
def create_file(path: str, content: str, working_dir: str = DEFAULT_WORKING_DIR) -> bool:
    """Create a new file at the specified path with the given content."""
    # Get the SDK instance for this working directory
    sdk = get_sdk(working_dir)
    return sdk.create_file(path, content)

def read_file(path: str, working_dir: str = DEFAULT_WORKING_DIR) -> str:
    """Read the content of a file at the specified path."""
    # Get the SDK instance for this working directory
    sdk = get_sdk(working_dir)
    return sdk.read_file(path)

def search_files(query: str, glob_patterns: List[str], working_dir: str = DEFAULT_WORKING_DIR) -> Dict[str, List[Dict[str, Any]]]:
    """Search for text in files matching the glob patterns."""
    # Get the SDK instance for this working directory
    sdk = get_sdk(working_dir)
    return sdk.search_files(query, glob_patterns)

def code(prompt: str, editable_files: List[str], readonly_files: Optional[List[str]] = None, working_dir: str = DEFAULT_WORKING_DIR) -> Dict[str, Any]:
    """Run an AI coding task with the given prompt and files."""
    # Get the SDK instance for this working directory
    sdk = get_sdk(working_dir)
    return sdk.code(prompt, editable_files, readonly_files or [])

# Define the tools list
tools = [
    create_file,
    read_file,
    search_files,
    code
]

# Initialize the language model
model = init_chat_model("claude-3-7-sonnet-latest", model_provider="anthropic")

# Create the CodeAct agent with the state schema
# CodeActState includes working_dir field which is passed to tool functions
code_act = create_codeact(model, tools, eval)
agent = code_act.compile()

if __name__ == "__main__":
    # Default working directory for CLI mode
    working_dir = DEFAULT_WORKING_DIR

    # Check if the working directory is a git repository
    git_enabled = is_git_repository(working_dir)

    print(f"🚀 Aider LangGraph App initialized with:")
    print(f"   - Working directory: {working_dir}")
    print(f"   - Git support: {'enabled' if git_enabled else 'disabled'} (auto-detected)")
    print(f"   - Model: gpt-4.1")
    print("\nEnter your coding request (or 'quit' to exit):")

    while True:
        user_input = input("\n> ")
        if user_input.lower() in ("quit", "exit", "q"):
            break

        messages = [{"role": "user", "content": user_input}]

        print("\nProcessing your request...\n")

        # Stream the agent's response, passing working_dir in the state
        for typ, chunk in agent.stream(
            {"messages": messages, "working_dir": working_dir},
            stream_mode=["values", "messages"],
            config={"configurable": {"thread_id": 1}},
        ):
            if typ == "messages":
                print(chunk[0].content, end="")
            elif typ == "values":
                print("\n\n---Final Result---\n")
                print(chunk)
                print("\n---End Result---\n")