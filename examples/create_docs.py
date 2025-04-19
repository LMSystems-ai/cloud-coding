"""
Example usage of the LMSYS SDK for AI-assisted coding
"""

from lmsys import Local
import os


# Initialize the SDK with your project directory and API keys
# This must be a git repository if use_git=True (default)
sdk = Local(
    working_dir="/Users/seansullivan/auto-prompt/",
    model="o4-mini",  # Optional: specify model
    editor_model="gpt-4.1",  # Optional: specify editor model
    architect_mode=True,
    lmsys_api_key=os.getenv("LMSYS_API_KEY")
)

# Run an AI coding task to improve the calculator
result = sdk.code(
    prompt="add simple quick start examples to the docs.md file based on our simple examples in example_usage.py, add_comments.py, and headless_example.py. be sure to start with the most simple, minimal code example, then go into more depth of the sdk capabilities.",
    editable_files=["docs.md"],
    readonly_files=["lmsys.py", "example_usage.py", "add_comments.py", "headless_example.py"]
)

# Check if the operation was successful
if result["success"]:
    print("Coding task was successful!")
    print("Changes made:")
    print(result["diff"])
else:
    print("Coding task failed or made no meaningful changes.")

# To check what content is now in the file after the AI changes
updated_content = sdk.read_file("docs.md")
print("Updated file content:", updated_content)