"""
Example usage of the LMSYS SDK for headless AI-assisted coding tasks
"""

import time
from lmsys import Local
import os

# Initialize the SDK with your project directory and API keys
sdk = Local(
    working_dir="/Users/seansullivan/aider-sdk-testing/",
    model="gpt-4.1-nano",
    use_git=False,
    lmsys_api_key=os.getenv("LMSYS_API_KEY")
)

# Create a test file
sdk.create_file(
    "src/calculator.py",
    """def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""
)

print("Starting a headless coding task...")

# Start a headless coding task
task_response = sdk.code_headless(
    prompt="Add multiply and divide functions to the calculator.",
    editable_files=["src/calculator.py"],
    readonly_files=[]
)

# Get the task ID from the response
task_id = task_response["task_id"]
print(f"Task started with ID: {task_id}")

# Poll for task status
for _ in range(10):  # Try up to 10 times
    time.sleep(2)  # Wait 2 seconds between polls

    # Check the status of the task
    status = sdk.get_headless_task_status(task_id)
    print(f"Task status: {status['status']}")

    # If the task is complete, display the results
    if status["status"] == "completed":
        print("\nTask completed!")
        print("Changes made:")
        print(status["result"]["diff"])

        # Show the updated file content
        updated_content = sdk.read_file("src/calculator.py")
        print("\nUpdated file content:")
        print(updated_content)
        break
    elif status["status"] == "failed":
        print(f"Task failed with error: {status.get('error', 'Unknown error')}")
        break

# Example of running multiple headless tasks in parallel
tasks = []
for i in range(3):
    task = sdk.code_headless(
        prompt=f"Add a comment to document the {'add' if i == 0 else 'subtract' if i == 1 else 'multiply'} function.",
        editable_files=["src/calculator.py"],
        readonly_files=[],
        task_id=f"task-{i+1}"
    )
    tasks.append(task["task_id"])
    print(f"Started parallel task {i+1} with ID: {task['task_id']}")

# Wait for all tasks to complete
time.sleep(10)

# Check status of all tasks
for task_id in tasks:
    status = sdk.get_headless_task_status(task_id)
    print(f"Task {task_id} status: {status['status']}")