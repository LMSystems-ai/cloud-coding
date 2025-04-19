#!/usr/bin/env python
"""
Example of using Cloud Code SDK in architect mode (in the current directory).

This demonstrates the two-model approach:
1. A main model (planner) creates a plan for code changes
2. An editor model implements the concrete changes to files
"""

from cloudcode import Local
import os


def main():

    # Initialize the Aider SDK in architect mode
    agent = Local(
        working_dir="/Users/seansullivan/auto-prompt/",
        model="o4-mini",  # Optional: specify model
        editor_model="gpt-4.1",  # Optional: specify editor model
        architect_mode=True,
        use_git=True,
        api_key=os.getenv("CLOUD_CODE_API_KEY")
    )


    # Run a coding task using the two-model workflow
    result = agent.code(
        prompt="How can i make the api keys for the pr",
        read_only_files=["app/components/video_player.tsx"]
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
