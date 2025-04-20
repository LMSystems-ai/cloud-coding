#!/usr/bin/env python
"""
Aider LangGraph React App - AI-assisted coding using ReAct Agent with LangGraph and Aider SDK
"""

import builtins
import contextlib
import io
import os
import subprocess
import json
import sys
from typing import Any, Dict, List, Optional, Annotated, Sequence, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Import Aider SDK
from cloudcode import Local


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


# Define the agent state
class AgentState(TypedDict):
    """The state of the agent."""
    # add_messages is a reducer
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Additional fields from the original state
    working_dir: str


# Default working directory if none provided
DEFAULT_WORKING_DIR = os.getcwd()


# Create tool functions for the agent that take working_dir from state
def create_file(path: str, content: str, working_dir: str = DEFAULT_WORKING_DIR) -> bool:
    """Create a new file at the specified path with the given content."""
    # Check if the directory is a git repository
    use_git = is_git_repository(working_dir)

    # Initialize SDK with the provided working directory
    sdk = Local(
        working_dir=working_dir,
        model="gpt-4.1",
        use_git=use_git,
        api_key=os.getenv("CLOUD_CODE_API_KEY")
    )
    return sdk.create_file(path, content)


def read_file(path: str, working_dir: str = DEFAULT_WORKING_DIR) -> str:
    """Read the content of a file at the specified path."""
    # Check if the directory is a git repository
    use_git = is_git_repository(working_dir)

    # Initialize SDK with the provided working directory
    sdk = Local(
        working_dir=working_dir,
        model="gpt-4.1",
        use_git=use_git,
        api_key=os.getenv("CLOUD_CODE_API_KEY")
    )
    return sdk.read_file(path)


def search_files(query: str, glob_patterns: List[str], working_dir: str = DEFAULT_WORKING_DIR) -> Dict[str, List[Dict[str, Any]]]:
    """Search for text in files matching the glob patterns."""
    # Check if the directory is a git repository
    use_git = is_git_repository(working_dir)

    # Initialize SDK with the provided working directory
    sdk = Local(
        working_dir=working_dir,
        model="gpt-4.1",
        use_git=use_git,
        api_key=os.getenv("CLOUD_CODE_API_KEY")
    )
    return sdk.search_files(query, glob_patterns)


def code(prompt: str, editable_files: List[str], readonly_files: Optional[List[str]] = None, working_dir: str = DEFAULT_WORKING_DIR) -> Dict[str, Any]:
    """Run an AI coding task with the given prompt and files."""
    # Check if the directory is a git repository
    use_git = is_git_repository(working_dir)

    # Initialize SDK with the provided working directory
    sdk = Local(
        working_dir=working_dir,
        model="gpt-4.1",
        use_git=use_git,
        api_key=os.getenv("CLOUD_CODE_API_KEY")
    )
    return sdk.code(prompt, editable_files, readonly_files or [])


def code_headless(prompt: str, editable_files: List[str], readonly_files: Optional[List[str]] = None, task_id: Optional[str] = None, working_dir: str = DEFAULT_WORKING_DIR) -> Dict[str, Any]:
    """Run an AI coding task in headless mode without waiting for results."""
    # Check if the directory is a git repository
    use_git = is_git_repository(working_dir)

    # Initialize SDK with the provided working directory
    sdk = Local(
        working_dir=working_dir,
        model="gpt-4.1",
        use_git=use_git,
        api_key=os.getenv("CLOUD_CODE_API_KEY")
    )
    return sdk.code_headless(prompt, editable_files, readonly_files or [], task_id)


def get_headless_task_status(task_id: str, working_dir: str = DEFAULT_WORKING_DIR) -> Dict[str, Any]:
    """Get the status of a headless coding task."""
    # Check if the directory is a git repository
    use_git = is_git_repository(working_dir)

    # Initialize SDK with the provided working directory
    sdk = Local(
        working_dir=working_dir,
        model="gpt-4.1",
        use_git=use_git,
        api_key=os.getenv("CLOUD_CODE_API_KEY")
    )
    return sdk.get_headless_task_status(task_id)



# Define the tools list and create a lookup dictionary by name
tools = [
    create_file,
    read_file,
    search_files,
    code,
    code_headless,
    get_headless_task_status,
]

tools_by_name = {tool.__name__: tool for tool in tools}

# Initialize the language model and bind tools
model = init_chat_model("o4-mini", model_provider="openai")
model = model.bind_tools(tools)

# Define the tool node for the react agent
def tool_node(state: AgentState):
    outputs = []
    working_dir = state["working_dir"]

    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # Extract working_dir parameter if the tool accepts it
        if "working_dir" in tools_by_name[tool_name].__annotations__:
            tool_args["working_dir"] = working_dir

        tool_result = tools_by_name[tool_name](**tool_args)

        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_name,
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": outputs}

# Define the node that calls the model
def call_model(state: AgentState, config: RunnableConfig):
    system_prompt = SystemMessage(
        "You are an AI coding assistant that helps with programming tasks. "
        "You can create files, read files, search for content in files, and help write code. "
        "Use the tools provided to assist the user effectively. "
        "For coding, you must only use the code and code_headless tools. These tools take in single step coding tasks described in natural language. "
        "When working on multiple coding tasks that could run in parallel, you can use "
        "code_headless to start tasks without waiting for them to complete, and "
        "get_headless_task_status to check on their progress later."
    )

    response = model.invoke([system_prompt] + state["messages"], config)

    # Return messages to be added to the state
    return {"messages": [response]}

# Define the conditional edge that determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    # If there is no tool call, then we finish
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"

# Create the react agent graph
def create_react_agent():
    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    # Set the entrypoint as `agent`
    workflow.set_entry_point("agent")

    # Add a conditional edge
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )

    # Add a normal edge from `tools` to `agent`
    workflow.add_edge("tools", "agent")

    # Compile the graph
    return workflow.compile()

# Create the agent
agent = create_react_agent()

if __name__ == "__main__":
    # Default working directory for CLI mode
    working_dir = DEFAULT_WORKING_DIR

    # Check if the working directory is a git repository
    git_enabled = is_git_repository(working_dir)

    print(f"🚀 Aider LangGraph React App initialized with:")
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
        for step in agent.stream(
            {"messages": messages, "working_dir": working_dir},
            stream_mode=["values", "messages"],
            config={"configurable": {"thread_id": 1}},
        ):
            if "messages" in step:
                content = step["messages"][-1].content
                if content:
                    print(content, end="")
            elif "values" in step:
                print("\n\n---Final Result---\n")
                print(step["values"])
                print("\n---End Result---\n")
