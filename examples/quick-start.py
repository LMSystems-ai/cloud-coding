"""
Example usage of the Cloud Code SDK for AI-assisted coding
"""

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
    """def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""
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
