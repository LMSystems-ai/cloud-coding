"""
Example usage of the LMSYS SDK for AI-assisted coding
"""

from lmsys import Local
import os

cwd = os.getcwd()

# Initialize the SDK with your project directory and API keys
# This must be a git repository if use_git=True (default)
sdk = Local(
    working_dir=cwd,
    model="o4-mini",  # Optional: specify model
    editor_model="gpt-4.1",  # Optional: specify editor model
    architect_mode=True,
    use_git=False,  # Optional: set to False to disable git requirements
    lmsys_api_key=os.getenv("LMSYS_API_KEY")
)

# Run an AI coding task to improve the calculator
result = sdk.code(
    prompt="Add comments to each of the functions in github_agent.py to explain what they do",
    editable_files=["github_agent.py"],
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