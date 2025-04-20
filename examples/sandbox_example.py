"""
Example usage of the Sandbox Cloud Code SDK for AI-assisted coding in a sandbox
"""

from cloudcode import SandboxSDK
import os
import tempfile

# Initialize the Sandbox SDK with your API keys
sdk = SandboxSDK(
    model="gpt-4.1",  # Main (planner) model
    editor_model="gpt-4.1-nano",  # Editor model for implementing changes
    architect_mode=True,
    api_key=os.getenv("CLOUD_CODE_API_KEY"),
    # Optional: specify a user ID for tracking sessions
    user_id="user123",
    # Optional: set sandbox timeout (in seconds)
    sandbox_timeout=600,
)

# Get information about the sandbox
sandbox_info = sdk.get_sandbox_info()
print(f"Sandbox created: {sandbox_info['sandbox_id']}")
print(f"Sandbox will expire at: {sandbox_info['end_at']}")

# Method 1: Create a file in the sandbox (internal method)
calc_file_path = os.path.join(sdk.working_dir, "calculator.py")
print(f"Creating file at: {calc_file_path}")
sdk.create_file(
    "calculator.py",
    """def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""
)

# Method 2: Use write_to_sandbox for a single file
sdk.write_to_sandbox(
    content="""def square(x):
    return x * x

def cube(x):
    return x * x * x
""",
    path=os.path.join(sdk.working_dir, "math_functions.py")
)

# Method 3: Create a temporary directory with multiple files and upload them all
temp_dir = tempfile.mkdtemp()
try:
    # Create a few files in the temp directory
    with open(os.path.join(temp_dir, "utils.py"), "w") as f:
        f.write("""def is_even(num):
    return num % 2 == 0

def is_odd(num):
    return num % 2 == 1
""")

    with open(os.path.join(temp_dir, "constants.py"), "w") as f:
        f.write("""PI = 3.14159
E = 2.71828
""")

    # Upload the entire directory to the sandbox
    print(f"Uploading directory: {temp_dir}")
    uploaded_files = sdk.write_to_sandbox(
        content="",  # Ignored when local_directory is provided
        local_directory=temp_dir,
        sandbox_directory=os.path.join(sdk.working_dir, "utils")
    )
    print(f"Uploaded {len(uploaded_files)} files to the sandbox")

finally:
    # Clean up the temporary directory
    import shutil
    shutil.rmtree(temp_dir)

# Run commands in the sandbox to verify files
result = sdk.run_command("find /home/user -type f | sort")
print("Files in sandbox:")
print(result["stdout"])

# Use AI to improve code in the sandbox
result = sdk.sandbox_code(
    prompt="Add a multiply and divide function to the calculator file. Make sure to handle division by zero in the divide function.",
    editable_files=[calc_file_path],
    readonly_files=[]
)

# Check if the operation was successful
if result["success"]:
    print("Coding task was successful!")
    print("Changes made:")
    print(result["diff"])
else:
    print("Coding task failed or made no meaningful changes.")

# Read files directly from the sandbox using the SDK
print("\nReading files from the sandbox:")

# Read the improved calculator file
calculator_content = sdk.read_sandbox_file(calc_file_path)
print("\nCalculator.py:")
print(calculator_content)

# Read the math_functions.py file
math_file_path = os.path.join(sdk.working_dir, "math_functions.py")
math_content = sdk.read_sandbox_file(math_file_path)
print("\nMath_functions.py:")
print(math_content)

# Read one of the utility files
utils_file_path = os.path.join(sdk.working_dir, "utils/utils.py")
utils_content = sdk.read_sandbox_file(utils_file_path)
print("\nUtils/utils.py:")
print(utils_content)

# You can also read as bytes if needed
constants_file_path = os.path.join(sdk.working_dir, "utils/constants.py")
constants_content_bytes = sdk.read_sandbox_file(constants_file_path, as_string=False)
print(f"\nConstants.py (as bytes, first 10 bytes): {constants_content_bytes[:10]}")
# Then convert to string when needed
constants_content = constants_content_bytes.decode('utf-8')
print(f"Constants.py (decoded): \n{constants_content}")

# Download a file from sandbox to local filesystem
local_calc_path = "local_calculator.py"
sdk.download_file(calc_file_path, local_calc_path)
print(f"\nDownloaded calculator.py to {local_calc_path}")

# Extend sandbox timeout if needed
sdk.extend_sandbox_timeout(600)  # Add 10 more minutes

# Show sandbox information
print("\nSandbox information:")
print(f"Sandbox ID: {sdk.sandbox_id}")
print(f"Working directory: {sdk.working_dir}")
print("You can reconnect to this sandbox later using this ID.")

# ----- Sandbox Management -----
print("\n----- Sandbox Management -----")

# Get the current timeout information
sandbox_info = sdk.get_sandbox_info()
current_end_time = sandbox_info["end_at"]
print(f"Current sandbox end time: {current_end_time}")

# Extend the sandbox lifetime
print("\nExtending sandbox lifetime by 10 minutes...")
sdk.extend_sandbox_timeout(600)  # 10 minutes = 600 seconds

# Verify the timeout was extended
updated_info = sdk.get_sandbox_info()
new_end_time = updated_info["end_at"]
print(f"New sandbox end time: {new_end_time}")

# Demonstrate sandbox killing (commented out so example can be run without actually terminating)
print("\n# To kill the sandbox when done, uncomment the next line:")
print("# kill_result = sdk.kill_sandbox()")
print("# print(f\"Sandbox termination: {kill_result['message']}\")")

# If you really want to kill the sandbox, uncomment these lines:
kill_result = sdk.kill_sandbox()
print(f"Sandbox termination: {kill_result['message']}")
print(f"Termination success: {kill_result['success']}")

print("\nExample completed successfully.")