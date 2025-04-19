"""
GitHub Agent - Clone, code, and push changes using Aider Sandbox SDK

Edit the variables in the CONFIG section below to set your workflow.
"""

import os
from typing import Dict, List, Optional
from lmsys import SandboxSDK


# =====================
# CONFIGURATION SECTION
# =====================
# Fill in your values below
REPO_URL = "https://github.com/RVCA212/lmsys-blog"  # GitHub repo to clone
BRANCH = None  # Branch to checkout (None for default)
PROMPT = "change the readme to say it's a blog for lmsystems.ai to talk about compounding ai agents"
EDIT_FILES = ["README.md"]  # List of files to edit (relative to repo root)
READONLY_FILES = []  # List of files to read but not edit
PUSH_BRANCH = "feature/ai-edits"  # New branch to push changes to
COMMIT_MESSAGE = "change the readme to say it's a blog for lmsystems.ai to talk about compounding ai agents"
INSTALL_CMD = None  # e.g., 'pip install -r requirements.txt' or None
RUN_CMD = None  # e.g., 'pytest' or None
LMSYS_API_KEY = os.environ.get("LMSYS_API_KEY")  # Or hardcode your key
SANDBOX_TIMEOUT = 1800  # 30 minutes
USER_ID = "user123"  # Optional
GITHUB_USERNAME = "RVCA212"  # Or hardcode your username
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")  # Or hardcode your personal access token
# =====================



class GitHubAgent:
    def __init__(
        self,
        model: str = "gpt-4.1",
        lmsys_api_key: Optional[str] = None,
        sandbox_timeout: int = 1800,
        user_id: Optional[str] = None
    ):
        """
        Initialize the GitHubAgent:
        - Set up the SandboxSDK with architect mode
        - Store sandbox working directory
        - Print sandbox ID and expiration
        """
        # ensure API key is set
        if not lmsys_api_key:
            lmsys_api_key = os.environ.get("LMSYS_API_KEY")
        self.sdk = SandboxSDK(
            model="gpt-4.1",  # Main (planner) model
            editor_model="gpt-4.1-nano",  # Editor model for implementing changes
            architect_mode=True,
            lmsys_api_key=lmsys_api_key,
            sandbox_timeout=sandbox_timeout,
            user_id=user_id,
        )
        self.repo_url = None
        self.repo_name = None
        self.working_dir = self.sdk.working_dir
        sandbox_info = self.sdk.get_sandbox_info()
        print(f"Sandbox created: {sandbox_info['sandbox_id']}")
        print(f"Sandbox will expire at: {sandbox_info['end_at']}")



    def clone_repository(self, repo_url: str, branch: Optional[str] = None) -> bool:
        """
        Clone a GitHub repository into the sandbox:
        - Inject credentials if provided
        - Configure git user/email
        - Optionally checkout a specific branch
        Returns True on success, False otherwise.
        """
        # record repo URL and compute local repo name
        self.repo_url = repo_url
        self.repo_name = repo_url.split("/")[-1]
        if self.repo_name.endswith(".git"):
            self.repo_name = self.repo_name[:-4]
        # Inject credentials into clone URL if provided
        username = GITHUB_USERNAME
        token = GITHUB_TOKEN
        clone_url = repo_url
        if username and token:
            # Prepend credentials for HTTPS clone
            proto, rest = repo_url.split("://", 1)
            clone_url = f"{proto}://{username}:{token}@{rest}"
        # Clone the repository using URL with credentials if available
        clone_cmd = f"cd {self.working_dir} && git clone \"{clone_url}\""
        result = self.sdk.run_command(clone_cmd)
        if result["exit_code"] != 0:
            print(f"Error cloning repository: {result['stderr']}")
            return False
        self.sdk.run_command(f"cd {self.working_dir}/{self.repo_name} && git config user.email 'github-agent@example.com'")
        self.sdk.run_command(f"cd {self.working_dir}/{self.repo_name} && git config user.name 'GitHub Agent'")
        if branch:
            checkout_cmd = f"cd {self.working_dir}/{self.repo_name} && git checkout {branch}"
            checkout_result = self.sdk.run_command(checkout_cmd)
            if checkout_result["exit_code"] != 0:
                print(f"Error checking out branch '{branch}': {checkout_result['stderr']}")
                return False
        print(f"Successfully cloned repository: {self.repo_name}")
        return True




    def run_code_task(self, prompt: str, files: List[str], readonly_files: Optional[List[str]] = None) -> Dict:
        """
        Submit a coding task to the sandbox editor:
        - Provide a natural-language prompt
        - Specify editable files and optional read-only files
        Returns the sandbox result dict containing success flag, diff, etc.
        """
        # ensure a repo has been cloned
        if not self.repo_name:
            print("No repository has been cloned yet.")
            return {"success": False, "error": "No repository cloned"}
        full_file_paths = [os.path.join(self.working_dir, self.repo_name, f) for f in files]
        full_readonly_paths = [os.path.join(self.working_dir, self.repo_name, f) for f in readonly_files] if readonly_files else []
        result = self.sdk.sandbox_code(
            prompt=prompt,
            editable_files=full_file_paths,
            readonly_files=full_readonly_paths
        )
        if result["success"]:
            print("Coding task completed successfully.")
            print("Changes made:")
            print(result["diff"])
        else:
            print("Coding task failed or made no meaningful changes.")
        return result



    def create_branch_and_push(self, branch_name: str, commit_message: str = "Changes made by GitHub Agent", github_username: Optional[str] = None, github_token: Optional[str] = None) -> bool:
        """
        Create a new git branch, stage and commit all changes,
        then push to origin using provided GitHub credentials.
        Returns True on success, False on any step failure.
        """
        # ensure a repo has been cloned
        if not self.repo_name:
            print("No repository has been cloned yet.")
            return False
        repo_dir = os.path.join(self.working_dir, self.repo_name)
        create_branch_cmd = f"cd {repo_dir} && git checkout -b {branch_name}"
        branch_result = self.sdk.run_command(create_branch_cmd)
        if branch_result["exit_code"] != 0:
            print(f"Error creating branch: {branch_result['stderr']}")
            return False
        add_cmd = f"cd {repo_dir} && git add ."
        add_result = self.sdk.run_command(add_cmd)
        if add_result["exit_code"] != 0:
            print(f"Error adding changes: {add_result['stderr']}")
            return False
        commit_cmd = f"cd {repo_dir} && git commit -m \"{commit_message}\""
        commit_result = self.sdk.run_command(commit_cmd)
        if commit_result["exit_code"] != 0 and "nothing to commit" not in commit_result["stderr"]:
            print(f"Error committing changes: {commit_result['stderr']}")
            return False
        if not github_username:
            github_username = "RVCA212"
        if not github_token:
            github_token = os.environ.get("GITHUB_TOKEN")
        if not github_username or not github_token:
            print("GitHub username/token not provided. Set GITHUB_USERNAME and GITHUB_TOKEN env vars or hardcode in config.")
            return False
        self.sdk.run_command(f"cd {repo_dir} && git config credential.helper 'store --file=/tmp/git-credentials'")
        credential_url = f"https://{github_username}:{github_token}@github.com"
        self.sdk.run_command(f"echo '{credential_url}' > /tmp/git-credentials")
        push_cmd = f"cd {repo_dir} && git push -u origin {branch_name}"
        push_result = self.sdk.run_command(push_cmd)
        self.sdk.run_command("rm -f /tmp/git-credentials")
        if push_result["exit_code"] != 0:
            print(f"Error pushing to GitHub: {push_result['stderr']}")
            return False
        print(f"Successfully pushed changes to branch '{branch_name}'")
        return True



    def install_dependencies(self, command: str) -> Dict:
        """
        Install dependencies by running the specified shell command
        inside the cloned repository directory.
        Returns the SDK run_command result dict.
        """
        # ensure a repo has been cloned
        if not self.repo_name:
            print("No repository has been cloned yet.")
            return {"exit_code": 1, "stdout": "", "stderr": "No repository cloned"}
        repo_dir = os.path.join(self.working_dir, self.repo_name)
        full_cmd = f"cd {repo_dir} && {command}"
        return self.sdk.run_command(full_cmd)



    def run_command_in_repo(self, command: str) -> Dict:
        """
        Run an arbitrary shell command in the cloned repository directory.
        Returns the SDK run_command result dict.
        """
        # ensure a repo has been cloned
        if not self.repo_name:
            print("No repository has been cloned yet.")
            return {"exit_code": 1, "stdout": "", "stderr": "No repository cloned"}
        repo_dir = os.path.join(self.working_dir, self.repo_name)
        full_cmd = f"cd {repo_dir} && {command}"
        return self.sdk.run_command(full_cmd)

    def extend_timeout(self, seconds: int = 600) -> None:
        """
        Extend the sandbox lifetime by the given number of seconds
        and print the new expiration time.
        """
        # call SDK to increase timeout
        self.sdk.extend_sandbox_timeout(seconds)
        updated_info = self.sdk.get_sandbox_info()
        print(f"Sandbox timeout extended. New expiration: {updated_info['end_at']}")

    def cleanup(self) -> Dict:
        """
        Terminate the sandbox and clean up all resources.
        Returns the SDK kill_sandbox result dict.
        """
        # kill the sandbox
        return self.sdk.kill_sandbox()


def main():
    """
    Orchestrator:
    1. Clone the repository
    2. Install dependencies (if specified)
    3. Run arbitrary commands (if specified)
    4. Run the coding/editing task
    5. Create a new branch and push changes
    6. Clean up the sandbox
    """
    agent = GitHubAgent(
        model="gpt-4.1",
        lmsys_api_key=LMSYS_API_KEY,
        sandbox_timeout=SANDBOX_TIMEOUT,
        user_id=USER_ID
    )
    try:
        # Clone repository
        if REPO_URL:
            success = agent.clone_repository(REPO_URL, BRANCH)
            if not success:
                print("Failed to clone repository. Exiting.")
                agent.cleanup()
                return
        # Install dependencies
        if INSTALL_CMD:
            print(f"Installing dependencies: {INSTALL_CMD}")
            result = agent.install_dependencies(INSTALL_CMD)
            if result["exit_code"] != 0:
                print(f"Error installing dependencies: {result['stderr']}")
        # Run command
        if RUN_CMD:
            print(f"Running command: {RUN_CMD}")
            result = agent.run_command_in_repo(RUN_CMD)
            print(f"Exit code: {result['exit_code']}")
            print(f"Output: {result['stdout']}")
            if result["stderr"]:
                print(f"Errors: {result['stderr']}")
        # Run coding task
        if PROMPT and EDIT_FILES:
            print(f"Running coding task: {PROMPT}")
            agent.run_code_task(PROMPT, EDIT_FILES, READONLY_FILES)
        # Push changes
        if PUSH_BRANCH:
            print(f"Pushing changes to branch: {PUSH_BRANCH}")
            agent.create_branch_and_push(PUSH_BRANCH, COMMIT_MESSAGE, GITHUB_USERNAME, GITHUB_TOKEN)
    finally:
        print("Cleaning up sandbox...")
        kill_result = agent.cleanup()
        print(f"Sandbox termination: {kill_result['message']}")
        print(f"Termination success: {kill_result['success']}")

if __name__ == "__main__":
    main()
