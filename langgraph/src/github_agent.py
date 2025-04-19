#!/usr/bin/env python
"""
Cloud Code LangGraph React App - AI-assisted coding using ReAct Agent with LangGraph and Cloud Code SDK Sandbox
"""


import os
import json
from typing import Any, Dict, List, Optional, Annotated, Sequence
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Import Aider SDK
from cloudcode import SandboxSDK


# Define the agent state
class AgentState(TypedDict):
    """The state of the agent."""
    # add_messages is a reducer
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Additional fields from the original state
    working_dir: str
    # GitHub information
    repo_url: Optional[str]
    github_token: Optional[str]
    github_username: Optional[str]
    branch: Optional[str]
    # Sandbox information
    sandbox_id: Optional[str]
    sdk: Optional[Any]


# Default values
DEFAULT_WORKING_DIR = os.getcwd()
DEFAULT_SANDBOX_TIMEOUT = 300  # 5 minutes


# Initialize Sandbox
def initialize_sandbox(state: AgentState) -> AgentState:
    """Initialize the sandbox environment and clone the repository if provided."""
    sdk = SandboxSDK(
        model="gpt-4.1",
        api_key=os.getenv("CLOUD_CODE_API_KEY"),
        sandbox_timeout=DEFAULT_SANDBOX_TIMEOUT,
        user_id="react-agent-user"
    )

    sandbox_info = sdk.get_sandbox_info()

    state["sdk"] = sdk
    state["sandbox_id"] = sandbox_info["sandbox_id"]
    state["working_dir"] = sdk.working_dir

    # Clone repository if URL is provided
    if state.get("repo_url"):
        clone_repository(state)

    return state


def clone_repository(state: AgentState) -> Dict:
    """Clone a GitHub repository to the sandbox."""
    sdk = state["sdk"]
    repo_url = state["repo_url"]
    github_username = state.get("github_username")
    github_token = state.get("github_token")
    branch = state.get("branch")

    # Format the repository URL with credentials if provided
    clone_url = repo_url
    if github_username and github_token:
        proto, rest = repo_url.split("://", 1)
        clone_url = f"{proto}://{github_username}:{github_token}@{rest}"

    # Clone the repository
    clone_cmd = f"cd {state['working_dir']} && git clone \"{clone_url}\""
    result = sdk.run_command(clone_cmd)

    # Configure git if cloning was successful
    if result["exit_code"] == 0:
        repo_name = repo_url.split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        sdk.run_command(f"cd {state['working_dir']}/{repo_name} && git config user.email 'react-agent@example.com'")
        sdk.run_command(f"cd {state['working_dir']}/{repo_name} && git config user.name 'React Agent'")

        # Checkout specific branch if provided
        if branch:
            checkout_cmd = f"cd {state['working_dir']}/{repo_name} && git checkout {branch}"
            sdk.run_command(checkout_cmd)

    return result


def cleanup_sandbox(state: AgentState) -> Dict:
    """Clean up and terminate the sandbox environment."""
    if state.get("sdk"):
        return state["sdk"].kill_sandbox()
    return {"success": False, "message": "No sandbox to terminate"}


# Create tool functions for the agent that use the sandbox
def create_file(path: str, content: str, state: AgentState) -> Dict:
    """Create a new file at the specified path with the given content."""
    sdk = state["sdk"]
    result = sdk.run_command(f"cat > {path} << 'EOL'\n{content}\nEOL")
    return {
        "success": result["exit_code"] == 0,
        "message": result["stdout"] if result["exit_code"] == 0 else result["stderr"]
    }


def read_file(path: str, state: AgentState) -> Dict:
    """Read the content of a file at the specified path."""
    sdk = state["sdk"]
    result = sdk.run_command(f"cat {path}")
    if result["exit_code"] == 0:
        return {
            "success": True,
            "content": result["stdout"]
        }
    else:
        return {
            "success": False,
            "error": result["stderr"]
        }


def search_files(query: str, glob_patterns: List[str], state: AgentState) -> Dict:
    """Search for text in files matching the glob patterns."""
    sdk = state["sdk"]
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
    sdk = state["sdk"]
    result = sdk.sandbox_code(
        prompt=prompt,
        editable_files=editable_files,
        readonly_files=readonly_files or []
    )
    return result


def run_command(command: str, state: AgentState) -> Dict:
    """Run a command in the sandbox environment."""
    sdk = state["sdk"]
    result = sdk.run_command(command)
    return {
        "exit_code": result["exit_code"],
        "stdout": result["stdout"],
        "stderr": result["stderr"]
    }


def bash(command: str, state: AgentState) -> Dict:
    """Run a bash command in the sandbox environment.
    This is a more user-friendly alias for run_command that emphasizes shell command execution."""
    return run_command(command, state)


def push_changes(branch_name: str, commit_message: str, state: AgentState) -> Dict:
    """Create a branch and push changes to GitHub."""
    sdk = state["sdk"]
    repo_url = state["repo_url"]
    github_username = state.get("github_username")
    github_token = state.get("github_token")

    if not repo_url or not branch_name:
        return {
            "success": False,
            "error": "Repository URL or branch name not provided"
        }

    repo_name = repo_url.split("/")[-1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    repo_dir = f"{state['working_dir']}/{repo_name}"

    # Create and checkout a new branch
    create_branch_cmd = f"cd {repo_dir} && git checkout -b {branch_name}"
    branch_result = sdk.run_command(create_branch_cmd)

    if branch_result["exit_code"] != 0:
        return {
            "success": False,
            "error": f"Failed to create branch: {branch_result['stderr']}"
        }

    # Add all changes
    add_cmd = f"cd {repo_dir} && git add ."
    add_result = sdk.run_command(add_cmd)

    if add_result["exit_code"] != 0:
        return {
            "success": False,
            "error": f"Failed to add changes: {add_result['stderr']}"
        }

    # Commit changes
    commit_cmd = f"cd {repo_dir} && git commit -m \"{commit_message}\""
    commit_result = sdk.run_command(commit_cmd)

    # Only proceed if commit was successful or if there was nothing to commit
    if commit_result["exit_code"] != 0 and "nothing to commit" not in commit_result["stderr"]:
        return {
            "success": False,
            "error": f"Failed to commit changes: {commit_result['stderr']}"
        }

    # Set up credentials for push
    if github_username and github_token:
        sdk.run_command(f"cd {repo_dir} && git config credential.helper 'store --file=/tmp/git-credentials'")
        credential_url = f"https://{github_username}:{github_token}@github.com"
        sdk.run_command(f"echo '{credential_url}' > /tmp/git-credentials")

    # Push changes
    push_cmd = f"cd {repo_dir} && git push -u origin {branch_name}"
    push_result = sdk.run_command(push_cmd)

    # Clean up credentials
    sdk.run_command("rm -f /tmp/git-credentials")

    if push_result["exit_code"] != 0:
        return {
            "success": False,
            "error": f"Failed to push changes: {push_result['stderr']}"
        }

    return {
        "success": True,
        "message": f"Successfully pushed changes to branch '{branch_name}'"
    }


# Define the tools list and create a lookup dictionary by name
tools = [
    create_file,
    read_file,
    search_files,
    code,
    run_command,
    bash,
    push_changes
]

tools_by_name = {tool.__name__: tool for tool in tools}

# Initialize the language model and bind tools
model = init_chat_model("gpt-4.1", model_provider="openai")
model = model.bind_tools(tools)

# Define the tool node for the react agent
def tool_node(state: AgentState):
    outputs = []

    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # Add state as a parameter for all tools
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
        "You are an AI coding assistant that helps with programming tasks in a sandbox environment. "
        "You can create files, read files, search for content in files, and write code. "
        "You can also run bash commands in the sandbox environment using the bash tool, "
        "run commands with the run_command tool, and push changes to GitHub. "
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
    import argparse

    parser = argparse.ArgumentParser(description="Aider LangGraph React App with Sandbox")
    parser.add_argument("--repo", help="GitHub repository URL to clone")
    parser.add_argument("--branch", help="Branch to checkout")
    parser.add_argument("--username", help="GitHub username")
    parser.add_argument("--token", help="GitHub personal access token")

    args = parser.parse_args()

    # Initialize state with GitHub information if provided
    state = {
        "messages": [{"role": "system", "content": "You are an AI assistant helping with coding tasks in a sandbox environment."}],
        "working_dir": DEFAULT_WORKING_DIR,
        "repo_url": args.repo,
        "github_username": args.username,
        "github_token": args.token,
        "branch": args.branch,
        "sandbox_id": None,
        "sdk": None
    }

    try:
        # Initialize the sandbox
        state = initialize_sandbox(state)

        print(f"🚀 Aider LangGraph React App initialized with:")
        print(f"   - Sandbox ID: {state['sandbox_id']}")
        print(f"   - Working directory: {state['working_dir']}")

        if state.get("repo_url"):
            repo_name = state["repo_url"].split("/")[-1]
            if repo_name.endswith(".git"):
                repo_name = repo_name[:-4]
            print(f"   - Repository: {repo_name}")
            print(f"   - Branch: {state.get('branch') or 'default'}")

        print(f"   - Model: gpt-4.1")
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

    finally:
        # Clean up sandbox
        if state.get("sdk"):
            print("Cleaning up sandbox...")
            result = cleanup_sandbox(state)
            print(f"Sandbox termination: {result['message']}")
            print(f"Termination success: {result['success']}")
