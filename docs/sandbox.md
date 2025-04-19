# SandboxSDK Documentation

The `SandboxSDK` extends LMSYS's AI coding capabilities to an isolated Remote Coding sandbox environment. This allows you to safely run AI-assisted code modifications, test code, and execute commands in a secure, cloud-based environment.

## Getting Started

### Installation

```bash
pip install lmsys
```

### Basic Usage

```python
from lmsys import SandboxSDK
import os

# Initialize the SDK with your API keys
sdk = SandboxSDK(
    model="gpt-4.1",  # Choose your preferred AI model
    lmsys_api_key=os.getenv("LMSYS_API_KEY"),
    sandbox_timeout=600,  # 10 minutes (in seconds)
)

# Check sandbox info
info = sdk.get_sandbox_info()
print(f"Sandbox created with ID: {info['sandbox_id']}")
print(f"Expires at: {info['end_at']}")
```

## Working with Files

### Creating Files

```python
# Create a file in the sandbox
sdk.create_file(
    "hello.py", #file path followed by file contents
    """
def greet(name):
    return f"Hello, {name}!"

print(greet("World"))
"""
)

# Run the file
result = sdk.run_command("python3 hello.py")
print(result["stdout"])  # Outputs: Hello, World!
```

### Writing Files to Sandbox

```python
# Method 1: Create a single file with write_to_sandbox
sdk.write_to_sandbox(
    content="print('Hello from a new file!')",
    path="/home/user/new_file.py"
)

# Method 2: Upload a local file
sdk.upload_file("local_script.py", "/home/user/uploaded_script.py")

# Method 3: Upload an entire directory
sdk.write_to_sandbox(
    content="",  # Ignored for directory uploads
    local_directory="./my_project",
    sandbox_directory="/home/user/project"
)
```

### Reading Files

```python
# Read a file from the sandbox
content = sdk.read_sandbox_file("/home/user/hello.py")
print(content)

# Read binary data
binary_data = sdk.read_sandbox_file("/home/user/image.png", as_string=False)

# Download a file from the sandbox to local machine
sdk.download_file("/home/user/results.txt", "local_results.txt")
```

## AI-Powered Code Modification

```python
# Use AI to improve code in the sandbox
result = sdk.sandbox_code(
    prompt="Add error handling to hello.py and create a new function that greets multiple people",
    editable_files=["/home/user/hello.py"],
    readonly_files=[]  # Files for reference only
)

if result["success"]:
    print("Code was improved successfully!")

    # See the modified file
    modified_code = sdk.read_sandbox_file("/home/user/hello.py")
    print(modified_code)
else:
    print("No significant changes were made")
```

## Running Commands in the Sandbox

```python
# Execute a shell command
result = sdk.run_command("ls -la")
print(result["stdout"])

# Install packages
sdk.run_command("pip install pandas numpy matplotlib")

# Run a Python script
sdk.run_command("python3 /home/user/analysis.py")

# Run multiple commands
sdk.run_command("cd /home/user && mkdir data && python3 -m http.server 8080 &")
```

## Sandbox Management

### Extending Timeout

```python
# Extend the sandbox lifetime by 30 minutes
sdk.extend_sandbox_timeout(1800)  # seconds

# Check the new expiration time
info = sdk.get_sandbox_info()
print(f"New expiration time: {info['end_at']}")
```

### Shutting Down a Sandbox

```python
# Terminate the sandbox when done
kill_result = sdk.kill_sandbox()
if kill_result["success"]:
    print(f"Sandbox successfully terminated")
else:
    print(f"Error: {kill_result['error']}")
```

## Real-World Examples

### Example 1: Data Analysis Project

```python
# Initialize the sandbox
sdk = SandboxSDK(model="gpt-4.1", lmsys_api_key=os.getenv("LMSYS_API_KEY"))

# Create a data analysis script
sdk.create_file(
    "analysis.py",
    """
# Analyze dataset
import pandas as pd

# TODO: Add data loading and analysis code
"""
)

# Upload a dataset
sdk.upload_file("sales_data.csv", "/home/user/sales_data.csv")

# Use AI to implement the analysis
sdk.sandbox_code(
    prompt="""
    Modify analysis.py to:
    1. Load the sales_data.csv file
    2. Perform basic statistical analysis
    3. Generate a summary report
    4. Create visualizations of the key metrics
    """,
    editable_files=["/home/user/analysis.py"]
)

# Install required packages
sdk.run_command("pip install pandas matplotlib seaborn")

# Run the analysis
sdk.run_command("python3 /home/user/analysis.py")

# Download the results
sdk.download_file("/home/user/analysis_report.html", "local_report.html")
```

### Example 2: Web Application Development

```python
# Initialize the sandbox
sdk = SandboxSDK(model="gpt-4.1", lmsys_api_key=os.getenv("LMSYS_API_KEY"))

# Upload your existing web app codebase
sdk.write_to_sandbox(
    content="",
    local_directory="./my_flask_app",
    sandbox_directory="/home/user/webapp"
)

# Use AI to add a new feature
sdk.sandbox_code(
    prompt="Add user authentication with login/logout functionality to the Flask application",
    editable_files=[
        "/home/user/webapp/app.py",
        "/home/user/webapp/models.py",
        "/home/user/webapp/templates/login.html"
    ],
    readonly_files=["/home/user/webapp/templates/base.html"]
)

# Install dependencies
sdk.run_command("cd /home/user/webapp && pip install -r requirements.txt")

# Run the application
sdk.run_command("cd /home/user/webapp && python app.py")

# Download the modified codebase
for file in ["app.py", "models.py", "templates/login.html"]:
    sdk.download_file(f"/home/user/webapp/{file}", f"./my_flask_app/{file}")
```

## Reconnecting to an Existing Sandbox

```python
# Save the sandbox ID for later use
sandbox_id = sdk.sandbox_id

# Later, reconnect to the same sandbox
reconnected_sdk = SandboxSDK(
    model="gpt-4.1",
    lmsys_api_key=os.getenv("LMSYS_API_KEY"),
    sandbox_id=sandbox_id  # Connect to existing sandbox
)

# Continue working with the same files and environment
content = reconnected_sdk.read_sandbox_file("/home/user/hello.py")
```

## Best Practices

1. **Use Appropriate Timeouts**: Set the initial timeout based on your expected usage. Extend only when needed.
2. **Clean Up When Done**: Always kill sandboxes when finished to free up resources.
3. **Error Handling**: Check return values from operations to ensure success.
4. **File Organization**: Use consistent paths in the sandbox to maintain organization.
5. **Security**: Never store sensitive credentials in sandbox files.

## Troubleshooting

### Files Not Found
If files appear to exist but can't be accessed:
```python
# Force create the file
sdk.run_command(f"touch /home/user/missing_file.py")
```

### Sandbox Connection Issues
If the sandbox connection fails:
```python
# Initialize with a longer timeout and retry mechanism
sdk = SandboxSDK(
    model="gpt-4.1",
    lmsys_api_key=os.getenv("LMSYS_API_KEY"),
    sandbox_timeout=1200  # 20 minutes
)
```
