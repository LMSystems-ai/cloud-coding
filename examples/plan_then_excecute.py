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

quickstart_code = """
from cloudcode import Local
import os

cwd = os.getcwd()

# Initialize the SDK with your project directory and API keys
# This must be a git repository if use_git=True (default)
sdk = Local(
    working_dir=cwd,
    api_key=os.getenv("CLOUD_CODE_API_KEY")
)

# Create a new file with content
sdk.create_file(
    "src/calculator.py", # specify file path
    "print('hello world')"
)

# Run an AI coding task to improve the calculator
result = sdk.code(
    prompt="Add a multiply and divide function to the calculator.py file. Make sure to handle division by zero in the divide function.",
    editable_files=["src/calculator.py"],
    readonly_files=[]
)

# Check if the operation was successful
if result["success"]:
    print("Coding task was successful!")
    print("Changes made:")
    print(result["diff"])
else:
    print("Coding task failed or made no meaningful changes.")

# To check what content is now in the file after the AI changes
updated_content = sdk.read_file("src/calculator.py")
print("Updated file content:", updated_content)
"""

# User's goal/task
user_goal = f"""
add a 'quick start' section to our main homepage (the root home page.tsx file with the sign in info on it)
Have the quick start section include a code block example that looks like this:
{quickstart_code}
have this quick start section be located below the main body container of the page where the sign in info is located. add syntax hghlighting to the code block.
"""



# Step 1: Planning Phase
# The AI will analyze the codebase and create a detailed plan in a markdown file
plan_file = "plan.md"
planning_result = sdk.code(
    prompt=f"Analyze the codebase to create a detailed plan for this task: {user_goal}. "
           f"Write a comprehensive plan to the file 'plan.md'. "
           f"The plan MUST be written directly to the plan.md file, not just displayed. "
           f"Include the high level goal, steps, file analysis, and implementation details.",
    editable_files=[plan_file],
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
        prompt=f"Read the plan in '{plan_file}' and execute it to accomplish this task: {user_goal}. "
               f"Follow the plan step by step to make the necessary changes.",
        editable_files=[],
        readonly_files=[plan_file]
    )

    # Check if the execution was successful
    if execution_result["success"]:
        print("\nExecution phase completed successfully!")
        print("Changes made:")
        print(execution_result["diff"])
    else:
        print("\nExecution phase failed or made no meaningful changes.")

    # To check what content is now in the file after the AI changes
    updated_content = sdk.read_file("docs.md")
    print("\nUpdated docs.md content:", updated_content)
else:
    print("Planning phase failed or made no meaningful plan.")
    print("Please review and try again.")