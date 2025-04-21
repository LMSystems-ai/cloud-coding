#!/usr/bin/env python
"""
Aider LangGraph React App - AI-assisted coding using a two‑stage workflow:
1. **Planner Agent** – a ReAct agent with only research tools that drafts a concrete plan for the coding task.
2. **Coder Agent** – the original coding assistant that executes the plan, creating / modifying code as needed.

Both agents run inside a LangGraph state‑machine so that planning happens first, then execution, with iterative tool‑calls in each stage.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import os
import subprocess
import json
import sys
from typing import Any, Dict, List, Optional, Annotated, Sequence, TypedDict
import logging

# ---------------------------------------------------------------------------
#  Local imports / path setup
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from trustcall import create_extractor  # safe tool‑calling wrapper

# External helper that does a web search with Exa.
from tools.exa_search import invoke_agent_with_prepended_text

# Aider SDK wrapper
from cloudcode import Local

# ---------------------------------------------------------------------------
#  Utility helpers
# ---------------------------------------------------------------------------

def is_git_repository(path: str) -> bool:
    """Return True if *path* is inside a Git work‑tree."""
    git_dir = os.path.join(path, ".git")
    if os.path.isdir(git_dir):
        return True
    try:
        result = subprocess.run(
            ["git", "-C", path, "rev-parse", "--is-inside-work-tree"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

# ---------------------------------------------------------------------------
#  LangGraph shared state‑object definition
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    """Shared state across planner + coder stages."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    working_dir: str
    plan: Optional[str]
    in_coding_stage: bool
    plan_written: bool
    plan_completed: bool


DEFAULT_WORKING_DIR = os.getcwd()

# ---------------------------------------------------------------------------
#  Tool implementations (same as before)
# ---------------------------------------------------------------------------

def _init_sdk(working_dir: str):
    return Local(
        working_dir=working_dir,
        model="o4-mini",
        editor_model="gpt-4.1",
        architect_mode=True,
        use_git=is_git_repository(working_dir),
        api_key=os.getenv("CLOUD_CODE_API_KEY"),
    )


def create_file(path: str, content: str, *, working_dir: str = DEFAULT_WORKING_DIR) -> bool:
    return _init_sdk(working_dir).create_file(path, content)


def read_file(path: str, *, working_dir: str = DEFAULT_WORKING_DIR) -> str:
    return _init_sdk(working_dir).read_file(path)


def search_files(
    query: str,
    glob_patterns: List[str],
    *,
    working_dir: str = DEFAULT_WORKING_DIR,
) -> Dict[str, List[Dict[str, Any]]]:
    return _init_sdk(working_dir).search_files(query, glob_patterns)


def code(
    prompt: str,
    editable_files: List[str],
    readonly_files: Optional[List[str]] | None = None,
    *,
    working_dir: str = DEFAULT_WORKING_DIR,
) -> Dict[str, Any]:
    return _init_sdk(working_dir).code(prompt, editable_files, readonly_files or [])


def code_headless(
    prompt: str,
    editable_files: List[str],
    readonly_files: Optional[List[str]] | None = None,
    task_id: Optional[str] | None = None,
    *,
    working_dir: str = DEFAULT_WORKING_DIR,
) -> Dict[str, Any]:
    return _init_sdk(working_dir).code_headless(
        prompt, editable_files, readonly_files or [], task_id
    )


def get_headless_task_status(task_id: str, *, working_dir: str = DEFAULT_WORKING_DIR) -> Dict[str, Any]:
    return _init_sdk(working_dir).get_headless_task_status(task_id)


# ------------------- research‑only helpers -------------------

def web_search_expert(query: str) -> dict:
    """Do a web search and return raw text results (simple wrapper)."""
    try:
        agent_response = invoke_agent_with_prepended_text(query)
        if agent_response and agent_response.get("messages"):
            last = agent_response["messages"][-1]
            if getattr(last, "content", ""):
                return {"result": last.content}
        return {"result": f"Performed web search for: {query}."}
    except Exception as exc:  # pragma: no cover
        return {"result": f"Web‑search error: {exc}"}


# ------------------- simple plan storage -------------------

def write_plan(plan_content: str) -> dict:
    """Store *plan_content* into state via tool‑node side‑effect."""
    return {"result": "Plan updated", "plan": plan_content}


def read_plan() -> dict:
    return {"result": "Read plan"}

# ---------------------------------------------------------------------------
#  Tool collections for each agent stage
# ---------------------------------------------------------------------------

# Full tool‑set for the coding agent
CODER_TOOLS = [
    read_file,
    search_files,
    code,
    code_headless,
    get_headless_task_status,
    web_search_expert,
    write_plan,
    read_plan,
]

# Research‑only tools for the planner agent
PLANNER_TOOLS = [read_file, search_files, web_search_expert, write_plan, read_plan]

TOOLS_BY_NAME: Dict[str, Any] = {t.__name__: t for t in CODER_TOOLS}

# ---------------------------------------------------------------------------
#  Create trust‑call extractors (one per stage)
# ---------------------------------------------------------------------------
model_coder = init_chat_model("o4-mini", model_provider="openai")
model_planner = model_coder  # same backend model, different system‑prompt + tools

coder_extractor = create_extractor(
    model_coder, tools=CODER_TOOLS, tool_choice="any", enable_inserts=False
)

planner_extractor = create_extractor(
    model_planner, tools=PLANNER_TOOLS, tool_choice="any", enable_inserts=False
)

# ---------------------------------------------------------------------------
#  Generic tool‑execution node (shared)
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def tool_node(state: AgentState):
    outputs: List[ToolMessage] = []
    working_dir = state.get("working_dir", DEFAULT_WORKING_DIR)
    current_plan = state.get("plan")
    plan_just_written = False

    last_msg = state["messages"][-1]
    for tc in getattr(last_msg, "tool_calls", []):
        name = tc["name"]
        kwargs = tc["args"]
        if name in TOOLS_BY_NAME and "working_dir" in TOOLS_BY_NAME[name].__code__.co_varnames:
            kwargs["working_dir"] = working_dir

        # Special‑case plan storage in state -------------------
        if name == "write_plan":
            state["plan"] = kwargs.get("plan_content", "")
            result = write_plan(**kwargs)
            plan_just_written = True
            logger.info(f"[tool_node] write_plan called. Plan content set. plan_just_written=True. Plan: {state['plan']}")

            # Check if the plan has an explicit end marker
            plan_content = kwargs.get("plan_content", "")
            if "-- End Plan --" in plan_content or "end plan" in plan_content.lower():
                state["plan_completed"] = True
                logger.info("[tool_node] Plan completed marker found. plan_completed=True.")

        elif name == "read_plan":
            result = {"result": "Current plan", "plan": current_plan or "No plan yet"}
        else:
            result = TOOLS_BY_NAME[name](**kwargs)

        outputs.append(
            ToolMessage(content=json.dumps(result), name=name, tool_call_id=tc["id"])
        )

    # If we just wrote a plan, mark that we should transition to coding stage
    if plan_just_written:
        state["plan_written"] = True
        logger.info(f"[tool_node] plan_written set to True after write_plan. State: plan_written={state['plan_written']}, plan_completed={state.get('plan_completed', False)}")

    return {"messages": outputs}


# ---------------------------------------------------------------------------
#  ReAct call‑model helpers (different prompts per stage)
# ---------------------------------------------------------------------------

def call_planner_model(state: AgentState, config: RunnableConfig):
    # Check if a plan has been written and not yet transitioned to coding stage
    if state.get("plan") and not state.get("in_coding_stage", False):
        state["plan_written"] = True
        logger.info(f"[call_planner_model] plan_written set to True. State: plan_written={state['plan_written']}, plan_completed={state.get('plan_completed', False)}")

    sys_prompt = SystemMessage(
        """
        You are a powerful agentic **Planner Agent**, specialized in deep technical research for coding tasks. Your primary role is to thoroughly analyze the user's request, perform targeted research using your available tools (web search and file system exploration), and then create a precise, actionable implementation plan.

        Your task includes the following responsibilities:

        1. **Understand the Task:**
            - Carefully interpret the user's coding requirements.
            - Clarify any ambiguities proactively.

        2. **Research Phase:**
            - Leverage `web_search_expert`, `search_files`, and `read_file` to gather accurate, detailed, and relevant information.
            - Always include concrete technical examples, real-world code snippets, or accurate references directly from authoritative sources.

        3. **Implementation Planning:**
            - Clearly outline a structured, step-by-step plan that a developer can easily follow.
            - Include specific coding steps, file modifications, and required dependencies, if applicable.

        4. **Documentation of the Plan:**
            - Save your finalized plan using `write_plan` (pass `plan_content`).
            - Clearly indicate the plan's completion by ending it with the explicit marker "-- End Plan --".

        5. **Workflow Transition:**
            - Once you've stored a complete plan, reply succinctly with "Planning complete. Moving to implementation." without invoking further tool calls. This signals the transition to the coding phase.

        Guidelines for tool usage:
        - Only invoke tools when necessary and beneficial to the accuracy and completeness of your plan.
        - Clearly state your intention before each tool use, including how the information supports your planning.
        - NEVER fabricate or guess technical details—always ground your information in verified research.

        Your ultimate goal is to ensure the subsequent implementation stage proceeds smoothly, efficiently, and accurately based on your rigorously researched plan.
        """
    )

    messages = [sys_prompt] + state["messages"]
    res = planner_extractor.invoke({"messages": messages}, config)

    # Check if the response contains any indication that planning is complete
    response_content = res["messages"][-1].content or ""
    planning_complete_indicators = [
        "plan is complete",
        "planning complete",
        "plan has been finalized",
        "implementation plan is ready",
        "here's the plan",
        "the planning phase is complete",
        "-- end plan --"
    ]

    # If we have a plan and the response indicates completion or has no tool calls
    if state.get("plan") and (
        any(indicator.lower() in response_content.lower() for indicator in planning_complete_indicators) or
        not getattr(res["messages"][-1], "tool_calls", None)
    ):
        state["plan_completed"] = True
        logger.info(f"[call_planner_model] plan_completed set to True. State: plan_written={state.get('plan_written', False)}, plan_completed={state['plan_completed']}")

    return {"messages": [res["messages"][-1]]}


def call_coder_model(state: AgentState, config: RunnableConfig):
    # Get the original user message and plan
    user_message = next((msg.content for msg in state["messages"] if getattr(msg, "type", None) == "human"), "")
    plan = state.get("plan", "No plan available")

    sys_prompt = SystemMessage(
        f"""
        You are the **Coder Agent** whose job is to orchestrate coding tasks performed by a 'junior developer' agent,
        represented by the 'code' and 'code_headless' tools. This junior developer has basic software engineering skills
        but no prior knowledge of the specific coding task or access to the internet.

        **USER'S ORIGINAL REQUEST:**
        {user_message}

        **IMPLEMENTATION PLAN:**
        {plan}

        Your primary responsibilities are:

        1. Follow the implementation plan outlined above.
        2. Break down complex tasks into smaller, manageable units clearly understandable to a junior developer.
        3. Create detailed and meaningful prompts for each coding task. Include accurate examples, explicit instructions,
           context, and any critical information the junior developer wouldn't have.
        4. Demonstrate empathy and understanding for the junior developer's limitations by anticipating points of confusion
           and proactively addressing them.

        Always:
        • Delegate specific, clearly-scoped tasks via the `code` and `code_headless` tools.
        • Use `get_headless_task_status` to monitor the progress of asynchronous tasks.
        • Conduct additional research first when encountering any new ground-truth information critical for accurate code generation.

        Your goal is to enable the junior developer to produce correct, robust, and maintainable code by carefully guiding them
        through each step with thoughtfully crafted prompts.
        """
    )
    messages = [sys_prompt] + state["messages"]
    res = coder_extractor.invoke({"messages": messages}, config)
    return {"messages": [res["messages"][-1]]}

# ---------------------------------------------------------------------------
#  Helper to pick next edge (shared for both stages)
# ---------------------------------------------------------------------------

def should_continue_planner(state: AgentState):
    """Specific edge selector for the planner stage."""
    last = state["messages"][-1]

    # Check for plan-related content in the last message
    plan_content = ""
    if hasattr(last, "content"):
        plan_content = last.content or ""

    # Look for planning completion markers in the message content
    has_completion_marker = any(marker in plan_content.lower() for marker in [
        "-- end plan --",
        "plan is complete",
        "planning complete",
        "moving to implementation",
        "ready for coding",
        "plan has been finalized"
    ])

    logger.info(f"[should_continue_planner] Checking routing. has_plan={bool(state.get('plan'))}, plan_completed={state.get('plan_completed', False)}, plan_written={state.get('plan_written', False)}, has_completion_marker={has_completion_marker}, last_tool_calls={getattr(last, 'tool_calls', None)}")

    # First, check explicit completion markers in either state or message content
    if state.get("plan_completed", False) or has_completion_marker:
        logger.info("[should_continue_planner] Routing: Plan explicitly completed, returning 'end' (to coder agent)")
        # Set persistent state for future nodes
        state["plan_completed"] = True
        state["plan_written"] = True
        state["in_coding_stage"] = True
        return "end"

    # Next, check if we have a plan AND the planner has no more tool calls
    # This indicates the planner is finished
    if state.get("plan") and not getattr(last, "tool_calls", None):
        logger.info("[should_continue_planner] Routing: plan exists and no tool_calls, returning 'end' (to coder agent)")
        # Set persistent state for future nodes
        state["plan_completed"] = True
        state["plan_written"] = True
        state["in_coding_stage"] = True
        return "end"

    # Check for plan content in the tool results (from read_plan or write_plan)
    if hasattr(last, "content") and "plan" in getattr(last, "content", ""):
        try:
            # Try to extract plan from tool message content (JSON)
            import json
            tool_result = json.loads(last.content)
            if isinstance(tool_result, dict) and "plan" in tool_result and tool_result["plan"]:
                # Set state for future nodes
                state["plan"] = tool_result["plan"]
                state["plan_written"] = True

                # If we see completion markers in the plan content
                if any(marker in tool_result["plan"].lower() for marker in ["-- end plan --", "end plan"]):
                    logger.info("[should_continue_planner] Routing: End marker found in plan content, returning 'end' (to coder agent)")
                    state["plan_completed"] = True
                    state["in_coding_stage"] = True
                    return "end"
        except:
            pass

    # Otherwise continue with tool calls
    if getattr(last, "tool_calls", None):
        logger.info("[should_continue_planner] Routing: tool_calls present, returning 'continue' (stay in planner)")
        return "continue"

    # Default case: if we have a plan of any kind, we're done planning
    if state.get("plan"):
        logger.info("[should_continue_planner] Routing: plan exists in default case, returning 'end' (to coder agent)")
        state["in_coding_stage"] = True
        return "end"

    logger.info("[should_continue_planner] Routing: default fallback, returning 'end' (to coder agent)")
    return "end"

def should_continue_coder(state: AgentState):
    """Specific edge selector for the coder stage."""
    last = state["messages"][-1]

    if getattr(last, "tool_calls", None):
        return "continue"
    return "end"

def should_continue_planning(state: AgentState):
    """
    Check if planning is complete based on tool execution results.
    Routes directly from planner_tools to either planner or coder.
    """
    # Check if planning is complete based on state from tool_node
    logger.info(f"[should_continue_planning] Checking if planning complete: plan_written={state.get('plan_written', False)}, plan_completed={state.get('plan_completed', False)}")

    # If either plan_written or plan_completed is True, we should transition to coding
    if state.get("plan_completed", False):
        logger.info("[should_continue_planning] Planning complete (has end marker), routing to coder")
        state["in_coding_stage"] = True
        return "to_coder"

    # Check if we've just written a plan with the write_plan tool
    if state.get("plan_written", False) and state.get("plan"):
        # Check for end markers in the plan
        plan = state.get("plan", "")
        if "-- end plan --" in plan.lower() or "end plan" in plan.lower():
            logger.info("[should_continue_planning] Plan has end marker, routing to coder")
            state["plan_completed"] = True
            state["in_coding_stage"] = True
            return "to_coder"

    logger.info("[should_continue_planning] Planning not complete, continue planning")
    return "continue_planning"

# ---------------------------------------------------------------------------
#  Build LangGraph workflow
# ---------------------------------------------------------------------------

def build_workflow():
    g = StateGraph(AgentState)

    # Stage 1 – planning -----------------------------------------------------
    g.add_node("planner", call_planner_model)
    g.add_node("planner_tools", tool_node)

    # Modified routing:
    # 1. Add conditional edges from planner_tools to either planner or directly to coder
    g.add_conditional_edges(
        "planner_tools",
        should_continue_planning,
        {
            "continue_planning": "planner",  # Continue planning
            "to_coder": "coder"              # Skip planner, go directly to coder
        }
    )

    # Keep original planner->planner_tools edge for tool calls
    g.add_conditional_edges(
        "planner",
        should_continue_planner,
        {"continue": "planner_tools", "end": "coder"},
    )

    # Stage 2 – coding -------------------------------------------------------
    g.add_node("coder", call_coder_model)
    g.add_node("coder_tools", tool_node)

    g.add_conditional_edges(
        "coder",
        should_continue_coder,
        {"continue": "coder_tools", "end": END},
    )
    g.add_edge("coder_tools", "coder")

    g.set_entry_point("planner")
    return g.compile()


agent = build_workflow()

# ---------------------------------------------------------------------------
#  Simple CLI wrapper (unchanged except new banner)  -------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    working_dir = DEFAULT_WORKING_DIR
    git_enabled = is_git_repository(working_dir)

    print("🚀 Aider LangGraph React App initialized (Planner → Coder workflow)")
    print(f"   – Working directory: {working_dir}")
    print(f"   – Git support: {'enabled' if git_enabled else 'disabled'} (auto‑detected)")
    print("   – Model: gpt‑4.1 (backend: o4‑mini trust‑call)")
    print("\nEnter your coding request (or 'quit' to exit):")

    while True:
        user_input = input("\n> ")
        if user_input.lower() in {"quit", "exit", "q"}:
            break

        messages = [{"role": "user", "content": user_input}]
        print("\nProcessing your request...\n")

        for step in agent.stream(
            {
                "messages": messages,
                "working_dir": working_dir,
                "plan": None,
                "in_coding_stage": False,
                "plan_written": False,
                "plan_completed": False  # Add new state flag for plan completion
            },
            stream_mode=["values", "messages"],
            config={"configurable": {"thread_id": 1}},
        ):
            if "messages" in step:
                content = step["messages"][-1].content
                if content:
                    print(content, end="")
            elif "values" in step:
                print("\n\n---Final Result---\n")
                print(step["values"])
                print("\n---End Result---\n")
