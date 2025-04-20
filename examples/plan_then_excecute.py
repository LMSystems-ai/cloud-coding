"""
Example usage of the Cloud Code SDK for AI-assisted coding with a two-step process:
1. Planning phase - analyze codebase and create a plan
2. Execution phase - execute the plan to accomplish the task
"""

from cloudcode import Local
import os


# Initialize the SDK with your project directory and API keys
# This must be a git repository if use_git=True (default)
sdk = Local(
    working_dir="/Users/seansullivan/with-supabase-app/",
    model="o4-mini",  # Optional: specify model
    editor_model="gpt-4.1",  # Optional: specify editor model
    architect_mode=True,
    use_git=True,
    api_key=os.getenv("CLOUD_CODE_API_KEY")
)


# User's goal/task
user_goal = f"""
add a folder to this website which allows me to create files underneath it, each being it's own page. impelement boilerplate
tehnical documentation components for a python sdk package which we're showcasing in these pages.
"""



# Step 1: Planning Phase
# The AI will analyze the codebase and create a detailed plan in a markdown file
planning_result = sdk.code(
    prompt=f"Analyze the codebase to create a detailed plan for this task: {user_goal}. "
           f"Write a comprehensive plan to the file 'plan.md'. "
           f"The plan MUST be written directly to the plan.md file, not just displayed. "
           f"Include the high level goal, steps, file analysis, and implementation details.",
    editable_files=["plan.md"],
    readonly_files=[]
)

# Check if the planning was successful
if planning_result["success"]:
    print("Planning phase completed successfully!")
    print("Plan created:")
    print(planning_result["diff"])

    # Step 2: Execution Phase
    # The AI will read the plan and execute the actual task
    execution_result = sdk.code(
        prompt=f"Read the plan in 'plan.md' and execute it to accomplish this task: {user_goal}. "
               f"Follow the plan step by step to make the necessary changes.",
        editable_files=[],
        readonly_files=["plan.md"]
    )

    # Check if the execution was successful
    if execution_result["success"]:
        print("\nExecution phase completed successfully!")
        print("Changes made:")
        print(execution_result["diff"])
    else:
        print("\nExecution phase failed or made no meaningful changes.")

else:
    print("Planning phase failed or made no meaningful plan.")
    print("Please review and try again.")