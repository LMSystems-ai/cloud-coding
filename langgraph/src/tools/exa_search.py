import json
import os
from typing import Annotated, Sequence, TypedDict, Dict, Any, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from .get_url import get_url_tool
from exa_py import Exa

# Define graph state
class AgentState(TypedDict):
    """The state of the agent."""
    # add_messages is a reducer
    # See https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers
    messages: Annotated[Sequence[BaseMessage], add_messages]
    links: Optional[str]

# # Define model
# model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
# model = ChatGroq(model="llama-3.1-8b-instant")
model = ChatOpenAI(model="gpt-4.1")
# model = ChatAnthropic(model="claude-3-7-sonnet-latest")


# Set up Exa client
exa_client = Exa(api_key=os.getenv("EXA_API_KEY"))

def exa_search_node(state: AgentState) -> Dict[str, Any]:
    """
    Node that uses Exa to search for relevant information based on the user query.
    It adds search results to state as links that will be prepended to messages.
    """
    # Get the user message
    user_message = state["messages"][-1].content

    # Call Exa API to get search results
    try:
        print(f"Searching Exa for: {user_message}")
        result = exa_client.search_and_contents(
            user_message,
            livecrawl="always",
            summary=True,
            num_results = 50
        )

        # Debug: Print the response type
        print(f"Exa API Response type: {type(result).__name__}")

        # Extract URLs and summaries
        links_text = ""

        # Print available attributes or methods
        print(f"Available attributes: {dir(result)}")

        # Handle response properly based on structure
        if hasattr(result, 'results') and result.results:
            print(f"Found {len(result.results)} results")

            for item in result.results:
                # Debug the item structure
                print(f"Result item type: {type(item).__name__}")
                print(f"Result item attributes: {dir(item)}")

                # Get URL and summary using getattr to safely handle missing attributes
                url = getattr(item, 'url', '')
                summary = getattr(item, 'summary', '')

                # Debug the extracted values
                print(f"URL: {url}")
                print(f"Summary length: {len(summary) if summary else 0}")

                links_text += f"- {url}: {summary}\n\n"
        else:
            print("No results found in response structure")

        # Return updated state with links
        print(f"Final links_text length: {len(links_text)}")
        return {"links": links_text}
    except Exception as e:
        import traceback
        print(f"Error in Exa search: {str(e)}")
        print(f"Exception details: {traceback.format_exc()}")

        # Still return empty links, but with more debugging info
        return {"links": "", "error": str(e)}

# Define nodes for the graph
def tool_node(state: AgentState):
    # If the tool is get_url_tool, we call it directly
    if state["messages"][-1].tool_calls[0]["name"] == "get_url":
        return get_url_tool(state)

    # For compatibility with any other tools (though we shouldn't have any now)
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        try:
            tool_result = tools_by_name[tool_call["name"]](state)
            for output in tool_result["messages"]:
                outputs.append(output)
        except Exception as e:
            # Handle errors gracefully
            error_message = f"Error executing tool {tool_call['name']}: {str(e)}"
            outputs.append(
                ToolMessage(
                    content=json.dumps({"error": error_message}),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
    return {"messages": outputs}


tools = [get_url_tool]
tools_by_name = {tool.__name__: tool for tool in tools}


def call_model(
    state: AgentState,
    config: RunnableConfig,
):
    # Updated system prompt that only mentions the get_url tool
    system_prompt = SystemMessage(
    """You are a technical documentation research assistant specializing in providing accurate, verified, and example-backed answers to software engineering, programming, and SDK-related queries. Your primary responsibility is to always retrieve precise technical details—such as exact code syntax, API usage examples, required parameters, configuration details, and best practices—from official documentation or authoritative technical sources.

You should ALWAYS perform thorough research using your retrieval tool and NEVER generate or fabricate technical details that are not explicitly found in the retrieved sources. Prioritize providing real, ground-truth code examples and technical content directly extracted from the original documentation.

Your tool:
- get_url: Retrieves webpage content in markdown format.
Parameters:
    - urls: A single URL or list of URLs to fetch content from. Example: "https://example.com" or ["https://example.com", "https://docs.example.com"]
    - timeout (optional, default 10.0): Request timeout in seconds.
    - follow_redirects (optional, default true): Follows HTTP redirects.

Guidelines:
1. Always search official technical documentation, API references, developer guides, and reputable resources for coding examples.
2. Use the provided retrieval tool to obtain exact, relevant documentation content that explicitly answers the user's technical question.
3. If necessary, retrieve information from multiple authoritative URLs to ensure comprehensive coverage and accuracy.
4. Compose your response exclusively from the retrieved documentation, clearly presenting accurate explanations, complete and functional code snippets, exact parameter details, and specific configuration examples.
5. NEVER fabricate or guess any technical information. Provide only truthful and explicitly documented information and code.
6. Clearly organize and structure your response with distinct sections for explanations, code syntax, usage examples, and parameter details to aid the software engineer in easily understanding and implementing the provided solutions.
"""
    )

    # Bind the get_url tool to the model
    bound_model = model.bind_tools([
        {
            "type": "function",
            "function": {
                "name": "get_url",
                "description": "Fetches content from URLs and converts it to markdown format",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "urls": {
                            "type": ["string", "array"],
                            "description": "A single URL or a list of URLs to fetch content from. Example: 'https://example.com' or ['https://example.com', 'https://docs.langchain.com']",
                            "items": {
                                "type": "string"
                            }
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Timeout in seconds for the HTTP request (default: 10.0)"
                        },
                        "follow_redirects": {
                            "type": "boolean",
                            "description": "Whether to follow HTTP redirects (default: true)"
                        }
                    },
                    "required": ["urls"]
                }
            }
        }
    ])

    # Get the user message and prepend links if available
    messages = state["messages"]

    # Create inputs for the model with proper message types
    input_messages = [system_prompt]

    # If we have links available, modify the first user message
    if "links" in state and state["links"]:
        # Get the first message which should be the user query
        if messages and len(messages) > 0:
            # Get the original message content
            original_message = messages[0]
            original_content = original_message.content

            # Print debugging info
            print(f"Original message type: {type(original_message).__name__}")
            print(f"Original content: {original_content}")

            # Create a new enriched content
            enriched_content = f"Here are some relevant links:\n\n{state['links']}\n\nUse the get_url tool to explore these pages to find information for the user given their query: <query>{original_content}</query>. Only respond with content found from the web, do not generate any content which you cannot source."

            # Add all messages, replacing the first one with the enriched content
            # Keep the same message type, usually HumanMessage
            from langchain_core.messages import HumanMessage
            input_messages.append(HumanMessage(content=enriched_content))

            # Add any remaining messages after the first one
            for msg in messages[1:]:
                input_messages.append(msg)
        else:
            # No messages to modify, so just use system prompt
            input_messages = [system_prompt] + list(messages)
    else:
        # No links to add, so use messages as is
        input_messages = [system_prompt] + list(messages)

    # Debug info
    print(f"Number of input messages: {len(input_messages)}")
    for i, msg in enumerate(input_messages):
        print(f"Message {i} type: {type(msg).__name__}, content length: {len(msg.content)}")

    # Invoke the model with properly formatted messages
    response = bound_model.invoke(input_messages, config)

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the conditional edge that determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"

# Define the graph
def create_react_agent():
    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the nodes
    workflow.add_node("exa_search", exa_search_node)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    # Set the entry point as the Exa search node
    workflow.set_entry_point("exa_search")

    # Add an edge from exa_search to agent
    workflow.add_edge("exa_search", "agent")

    # Add conditional edges for the agent node
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )

    # Add an edge from tools to agent
    workflow.add_edge("tools", "agent")

    # Now we can compile our graph
    return workflow.compile()

# Create the agent
react_agent = create_react_agent()

# Helper function to prepend text to user messages
def invoke_agent_with_prepended_text(user_input: str, prepend_text: str = ""):
    """
    Invokes the react agent with text prepended to the user's message.

    Args:
        user_input: The original user message
        prepend_text: Text to prepend to the user message

    Returns:
        The agent's response
    """
    # Just use the user input directly - preprocessing happens in the exa_search_node
    inputs = {"messages": [("user", user_input)], "links": ""}
    return react_agent.invoke(inputs)

# Helper function for formatting the stream nicely
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            print(f"Message type: {type(message).__name__}")
            print(f"Content: {message.content}")
            if hasattr(message, 'tool_calls') and message.tool_calls:
                print(f"Tool calls: {message.tool_calls}")

# Example usage
def run_example():
    # Example of using the agent with Exa search
    return invoke_agent_with_prepended_text(
        "How can I use pinecone hybrid search as a tool with a langchain chatmodel like openai?"
    )

if __name__ == "__main__":
    result = run_example()
    print("Final result:", result)
