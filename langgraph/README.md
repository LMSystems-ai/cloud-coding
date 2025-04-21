# Using LMSYS SDK with Langgraph

## Quick Start

```bash
pip install -e .
```

```bash
export OPENAI_API_KEY=

export ANTHROPIC_API_KEY=
```

```bash
langgraph dev
```

# Agents

- Agent: `react.py` is a [react agent](https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/) with the cloudcode sdk as tools for coding in a specific local directory

- Sandbox: `sandbox.py` is the same as the react agent but uses the sandbox sdk from cloudcode to make code changes in a remote sandbox directory

- Codeact: `codeact.py` uses langgraph [codeact](https://github.com/langchain-ai/langgraph-codeact) with the cloudcode sdk for coding in a specific local directory

- Websearch: *WIP* `websearch.py` is 2 react agents: a planner agent whose job it is to research the web and the codebase to generate a plan, then a coder agent to execute the plan.
Currently the handoff between the planner and coder isn't working.