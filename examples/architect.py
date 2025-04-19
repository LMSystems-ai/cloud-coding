#!/usr/bin/env python
"""
Example of using LMSYS SDK in architect mode (in the current directory).

This demonstrates the two-model approach:
1. A main model (planner) creates a plan for code changes
2. An editor model implements the concrete changes to files
"""

from lmsys import Local
import os


def main():
    # Use the current directory
    cwd = os.getcwd()
    example_file = "example.py"

    # Initialize the Aider SDK in architect mode
    agent = Local(
        working_dir=cwd,
        model="o4-mini",  # Main (planner) model
        editor_model="gpt-4.1",  # Editor model for implementing changes
        architect_mode=True,
        lmsys_api_key=os.getenv("LMSYS_API_KEY")
    )

    # Create or overwrite a simple Python file to modify using the SDK
    agent.create_file(
        example_file,
        """def add(a, b):
    return a + b
"""
    )

    # Run a coding task using the two-model workflow
    result = agent.code(
        prompt="hello world",
        editable_files=[example_file]
    )

    # Print the results
    print("\nTask completed!")
    print(f"Success: {result['success']}")
    print("\nChanges made:")
    print(result["diff"])


    # Display cost information
    print("\nTask completed!")

    if "cost" in result and result["cost"]:
        print("\nCost information for this run:")

        # Check if we're in architect mode to show planner and editor costs separately
        if agent.architect_mode and 'planner_cost' in result['cost'] and 'editor_cost' in result['cost']:
            print(f"  - Planner model cost: ${result['cost'].get('planner_cost', 0):.6f}")
            print(f"  - Editor model cost: ${result['cost'].get('editor_cost', 0):.6f}")
            print(f"  - Combined message cost: ${result['cost'].get('message_cost', 0):.6f}")
        else:
            print(f"  - Message cost: ${result['cost'].get('message_cost', 0):.6f}")

        print(f"  - Session cost: ${result['cost'].get('session_cost', 0):.6f}")

        if 'tokens' in result['cost']:
            tokens = result['cost']['tokens']
            if isinstance(tokens, dict):
                print(f"  - Input tokens: {tokens.get('input', 0)}")
                print(f"  - Output tokens: {tokens.get('output', 0)}")
                print(f"  - Total tokens: {tokens.get('input', 0) + tokens.get('output', 0)}")
    else:
        print("\nNo cost information available for this run.")

    # Show the file content after the task
    print(f"\nUpdated content of {example_file}:")
    print(agent.read_file(example_file))

    # Show total costs
    total_costs = agent.get_total_cost()
    print("\nTotal costs for all runs in this session:")
    print(f"  - Message costs: ${total_costs['total_message_cost']:.6f}")
    print(f"  - Session costs: ${total_costs['total_session_cost']:.6f}")
    print(f"  - Combined total: ${total_costs['total_cost']:.6f}")

    # Add information about billing
    print("\nNote: The combined total cost is what has been logged to our usage tracking system.")
    print("      This reflects the true cost of running in architect mode, which uses two models.")
    print("      (a planner model and an editor model working together)")

if __name__ == "__main__":
    main()
