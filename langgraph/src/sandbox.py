#!/usr/bin/env python
"""
LMSYS LangGraph Sandbox App - AI-assisted coding using ReAct Agent with LangGraph and LMSYS SDK Sandbox
"""

import os
import json
import sys
from typing import Any, Dict, List, Optional, Annotated, Sequence
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Import LMSYS Sandbox SDK
from lmsys import SandboxSDK


# Define the agent state
class AgentState(TypedDict):
    """The state of the agent."""
    # add_messages is a reducer
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Sandbox information
    sandbox_id: Optional[str]
    working_dir: str
    sdk: Optional[Any]


# Default values
DEFAULT_WORKING_DIR = "/home/user"
DEFAULT_SANDBOX_TIMEOUT = 300  # 5 minutes


# Initialize Sandbox
def initialize_sandbox(state: AgentState) -> AgentState:
    """Initialize the sandbox environment."""
    sdk = SandboxSDK(
        model="gpt-4.1",
        lmsys_api_key=os.getenv("LMSYS_API_KEY"),
        sandbox_timeout=DEFAULT_SANDBOX_TIMEOUT,
        user_id="react-agent-user"
    )

    sandbox_info = sdk.get_sandbox_info()

    state["sdk"] = sdk
    state["sandbox_id"] = sandbox_info["sandbox_id"]
    state["working_dir"] = sdk.working_dir

    # Create a demo file to showcase file creation
    sdk.create_file(
        "hello.py",
        """def greet(name):
    return f"Hello, {name}!"

if __name__ == "__main__":
    print(greet("World"))
"""
    )

    return state


def cleanup_sandbox(state: AgentState) -> Dict:
    """Clean up and terminate the sandbox environment."""
    if state.get("sdk"):
        return state["sdk"].kill_sandbox()
    return {"success": False, "message": "No sandbox to terminate"}


# Create tool functions for the agent that use the sandbox
def create_file(path: str, content: str, state: AgentState) -> Dict:
    """Create a new file at the specified path with the given content."""
    sdk = state.get("sdk")
    if sdk is None:
        return {
            "success": False,
            "error": "Sandbox SDK not initialized"
        }

    try:
        sdk.write_to_sandbox(content=content, path=os.path.join(state["working_dir"], path))
        return {
            "success": True,
            "message": f"File {path} created successfully."
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def read_file(path: str, state: AgentState) -> Dict:
    """Read the content of a file at the specified path."""
    sdk = state.get("sdk")
    if sdk is None:
        return {
            "success": False,
            "error": "Sandbox SDK not initialized"
        }

    try:
        content = sdk.read_sandbox_file(os.path.join(state["working_dir"], path))
        return {
            "success": True,
            "content": content
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def search_files(query: str, glob_patterns: List[str], state: AgentState) -> Dict:
    """Search for text in files matching the glob patterns."""
    sdk = state.get("sdk")
    if sdk is None:
        return {
            "success": False,
            "error": "Sandbox SDK not initialized"
        }

    search_results = {}

    for pattern in glob_patterns:
        # Use grep to search for files matching the pattern and containing the query
        cmd = f"cd {state['working_dir']} && grep -r --include='{pattern}' -l '{query}' ."
        result = sdk.run_command(cmd)

        if result["exit_code"] == 0:
            # For each found file, get the matching lines
            files = result["stdout"].strip().split('\n')
            search_results[pattern] = []

            for file in files:
                if file:  # Skip empty lines
                    cmd = f"cd {state['working_dir']} && grep -n '{query}' '{file}'"
                    line_result = sdk.run_command(cmd)

                    if line_result["exit_code"] == 0:
                        lines = line_result["stdout"].strip().split('\n')
                        matches = []

                        for line in lines:
                            if line:
                                line_parts = line.split(':', 1)
                                if len(line_parts) >= 2:
                                    line_num = line_parts[0]
                                    content = line_parts[1]
                                    matches.append({
                                        "line": int(line_num),
                                        "content": content
                                    })

                        search_results[pattern].append({
                            "file": file,
                            "matches": matches
                        })

    return search_results


def code(prompt: str, editable_files: List[str], readonly_files: Optional[List[str]] = None, state: AgentState = None) -> Dict[str, Any]:
    """Run an AI coding task with the given prompt and files."""
    sdk = state.get("sdk")
    if sdk is None:
        return {
            "success": False,
            "error": "Sandbox SDK not initialized"
        }

    try:
        result = sdk.sandbox_code(
            prompt=prompt,
            editable_files=editable_files,
            readonly_files=readonly_files or []
        )
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def run_command(command: str, state: AgentState) -> Dict:
    """Run a command in the sandbox environment."""
    sdk = state.get("sdk")
    if sdk is None:
        return {
            "success": False,
            "error": "Sandbox SDK not initialized",
            "exit_code": 1,
            "stdout": "",
            "stderr": "Sandbox SDK not initialized"
        }

    try:
        result = sdk.run_command(command)
        return {
            "exit_code": result["exit_code"],
            "stdout": result["stdout"],
            "stderr": result["stderr"]
        }
    except Exception as e:
        return {
            "exit_code": 1,
            "stdout": "",
            "stderr": str(e)
        }


def bash(command: str, state: AgentState) -> Dict:
    """Run a bash command in the sandbox environment.
    This is a more user-friendly alias for run_command that emphasizes shell command execution."""
    return run_command(command, state)


# Define the tools list and create a lookup dictionary by name
tools = [
    create_file,
    read_file,
    search_files,
    code,
    run_command,
    bash
]

tools_by_name = {tool.__name__: tool for tool in tools}

# Initialize the language model and bind tools
model = init_chat_model("gpt-4.1", model_provider="openai")
model = model.bind_tools(tools)

# Define the tool node for the react agent
def tool_node(state: AgentState):
    outputs = []

    # Ensure SDK is initialized before proceeding
    if state.get("sdk") is None:
        try:
            state = initialize_sandbox(state)
        except Exception as e:
            return {"messages": [ToolMessage(
                content=json.dumps({"error": f"Failed to initialize sandbox: {str(e)}"}),
                name="initialization_error",
                tool_call_id="init_error"
            )]}

    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # Always pass our initialized state to the tools
        # Remove any state from args to avoid conflicts
        if 'state' in tool_args:
            del tool_args['state']

        try:
            # Add state as a parameter for all tools
            tool_result = tools_by_name[tool_name](**tool_args, state=state)
        except Exception as e:
            tool_result = {
                "success": False,
                "error": f"Error executing {tool_name}: {str(e)}"
            }

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
        "You are an AI coding assistant that helps with programming tasks in a sandbox environment. "
        "You can create files, read files, search for content in files, and write code. "
        "You can also run bash commands in the sandbox environment using the bash tool, "
        "or run commands with the run_command tool. "
        "Use the tools provided to assist the user effectively."
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
    # Initialize state
    state = {
        "messages": [],
        "working_dir": DEFAULT_WORKING_DIR,
        "sandbox_id": None,
        "sdk": None
    }

    try:
        # Initialize the sandbox - make sure this succeeds before proceeding
        try:
            state = initialize_sandbox(state)
            print(f"🚀 LangGraph Sandbox App initialized with:")
            print(f"   - Sandbox ID: {state['sandbox_id']}")
            print(f"   - Working directory: {state['working_dir']}")
            print(f"   - Model: gpt-4.1")
        except Exception as e:
            print(f"❌ Failed to initialize sandbox: {str(e)}")
            print("Please check your LMSYS_API_KEY environment variable.")
            sys.exit(1)

        print("\nEnter your coding request (or 'quit' to exit):")

        while True:
            user_input = input("\n> ")
            if user_input.lower() in ("quit", "exit", "q"):
                break

            messages = [{"role": "user", "content": user_input}]

            print("\nProcessing your request...\n")

            # Stream the agent's response
            for step in agent.stream(
                {"messages": messages, **state},
                stream_mode=["messages"],
                config={"configurable": {"thread_id": 1}},
            ):
                if "messages" in step:
                    content = step["messages"][-1].content
                    if content:
                        print(content, end="")

    finally:
        # Clean up sandbox
        if state.get("sdk"):
            print("\nCleaning up sandbox...")
            result = cleanup_sandbox(state)
            print(f"Sandbox termination: {result['message']}")
            print(f"Termination success: {result['success']}")
