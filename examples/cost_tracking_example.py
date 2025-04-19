#!/usr/bin/env python
"""
Example demonstrating cost tracking with the LMSYS SDK.

This script shows how to track and report API costs when using the SDK
for AI-assisted coding tasks in both local and sandbox environments.
"""

from lmsys import Local, SandboxSDK
import os
import json

def run_local_example():
    """Run a simple local example with cost tracking."""
    print("\n=== Running Local Example with Cost Tracking ===")

    # Initialize the SDK with your API keys
    agent = Local(
        working_dir=os.getcwd(),
        model="gpt-4.1-nano",  # Use a model that reports costs
        use_git=False,
        lmsys_api_key=os.getenv("LMSYS_API_KEY"),
    )

    # Create a simple Python file to modify
    example_file = "example_math.py"
    agent.create_file(
        example_file,
        """def add(a, b):
    return a + b
"""
    )

    print(f"Created file {example_file}")

    # Run a coding task
    print("\nRunning coding task...")
    result = agent.code(
        prompt="Add a multiply function to this file that multiplies two numbers.",
        editable_files=[example_file]
    )

    # Display cost information
    print("\nTask completed!")

    if "cost" in result and result["cost"]:
        print("\nCost information for this run:")
        for key, value in result["cost"].items():
            print(f"  - {key}: {value}")
    else:
        print("\nNo cost information available for this run.")

    # Show the file content after the task
    print(f"\nUpdated content of {example_file}:")
    print(agent.read_file(example_file))

    # Show total costs
    total_costs = agent.get_total_cost()
    print("\nTotal costs for all runs:")
    for key, value in total_costs.items():
        print(f"  - {key}: ${value:.6f}")

    return result

def run_sandbox_example():
    """Run a simple sandbox example with cost tracking."""
    print("\n=== Running Sandbox Example with Cost Tracking ===")

    # Initialize the Sandbox SDK with your API keys
    sdk = SandboxSDK(
        model="gpt-4.1-nano",  # Use a model that reports costs
        lmsys_api_key=os.getenv("LMSYS_API_KEY"),
        sandbox_timeout=30,
    )

    # Get information about the sandbox
    sandbox_info = sdk.get_sandbox_info()
    print(f"Sandbox created: {sandbox_info['sandbox_id']}")

    # Create a simple Python file in the sandbox
    example_file = "example_calc.py"
    sdk.create_file(
        example_file,
        """def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""
    )

    print(f"Created file {example_file} in sandbox")

    # Run a coding task
    print("\nRunning coding task in sandbox...")
    result = sdk.sandbox_code(
        prompt="Add multiply and divide functions to this calculator file. Make sure to handle division by zero.",
        editable_files=[example_file]
    )

    # Display cost information
    print("\nTask completed!")

    if "cost" in result and result["cost"]:
        print("\nCost information for this sandbox run:")
        for key, value in result["cost"].items():
            print(f"  - {key}: {value}")
    else:
        print("\nNo cost information available for this sandbox run.")

    # Show the file content after the task
    print(f"\nUpdated content of {example_file} in sandbox:")
    print(sdk.read_sandbox_file(example_file))

    # Show total costs
    total_costs = sdk.get_total_cost()
    print("\nTotal costs for all sandbox runs:")
    for key, value in total_costs.items():
        print(f"  - {key}: ${value:.6f}")

    # Terminate the sandbox when done
    sdk.kill_sandbox()
    print("\nSandbox terminated.")

    return result

def run_detailed_cost_history_example():
    """Example showing how to access and analyze the cost history."""
    print("\n=== Cost History Analysis Example ===")

    # Initialize the SDK
    agent = Local(
        working_dir=os.getcwd(),
        model="gpt-4.1-nano",
        use_git=False,
        lmsys_api_key=os.getenv("LMSYS_API_KEY"),
    )

    # Create a test file
    test_file = "test_functions.py"
    agent.create_file(
        test_file,
        """def greet(name):
    return f"Hello, {name}!"
"""
    )

    # Run multiple tasks to build up history
    prompts = [
        "Add a farewell function that takes a name and returns a goodbye message.",
        "Add a function to capitalize a name with proper title case.",
        "Add docstrings to all functions in the file."
    ]

    for i, prompt in enumerate(prompts):
        print(f"\nRunning task {i+1}: {prompt}")
        result = agent.code(prompt=prompt, editable_files=[test_file])

        if "cost" in result and result["cost"]:
            print(f"  Task {i+1} cost: ${result['cost'].get('message_cost', 0):.4f}")

    # Get and display the full cost history
    cost_history = agent.get_cost_history()

    print("\nDetailed cost history:")
    print(f"Total of {len(cost_history)} entries")

    total_message_cost = 0
    total_session_cost = 0

    for i, entry in enumerate(cost_history):
        cost = entry.get("cost", {})
        message_cost = cost.get("message_cost", 0)
        session_cost = cost.get("session_cost", 0)
        timestamp = entry.get("timestamp", "unknown")

        print(f"\nRun #{i+1} at {timestamp}")
        print(f"  Prompt: {entry.get('prompt', 'N/A')[:50]}...")
        print(f"  Files: {entry.get('files', [])}")
        print(f"  Message Cost: ${message_cost:.6f}")
        print(f"  Session Cost: ${session_cost:.6f}")

        total_message_cost += message_cost
        total_session_cost += session_cost

    print("\nSummary:")
    print(f"  Total Message Cost: ${total_message_cost:.6f}")
    print(f"  Total Session Cost: ${total_session_cost:.6f}")
    print(f"  Total Cost: ${total_message_cost + total_session_cost:.6f}")

    # Export cost history to JSON
    with open("cost_history.json", "w") as f:
        json.dump(cost_history, f, indent=2)

    print("\nCost history exported to cost_history.json")

if __name__ == "__main__":
    # Check for required API keys
    if not os.environ.get("LMSYS_API_KEY"):
        print("Error: LMSYS_API_KEY environment variable is required.")
        exit(1)

    # Run the local example
    local_result = run_local_example()


    # Run the detailed cost history example
    run_detailed_cost_history_example()

    print("\nAll examples completed successfully.")