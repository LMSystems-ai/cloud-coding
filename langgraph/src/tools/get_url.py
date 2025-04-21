import json
import logging
from typing import Dict, Any, List, Union

import httpx
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)

# Default timeout for HTTP requests
DEFAULT_TIMEOUT = 10.0

def extract_markdown_content(url: str, timeout: float = DEFAULT_TIMEOUT, follow_redirects: bool = True) -> Dict[str, Any]:
    """
    Extracts the main content of a webpage and converts it to Markdown format.
    Ignores unwanted elements and focuses on text and code blocks.

    Args:
        url: The URL to fetch content from
        timeout: Timeout in seconds for the HTTP request
        follow_redirects: Whether to follow HTTP redirects

    Returns:
        A dictionary containing the status and either the markdown content or error message
    """
    try:
        # Fetch the webpage content
        with requests.Session() as session:
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            })
            response = session.get(url, timeout=timeout, allow_redirects=follow_redirects)
            response.raise_for_status()

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove unwanted elements
        for tag in soup(["script", "style", "svg", "iframe", "nav", "footer"]):
            tag.decompose()

        # Extract the body content
        body = soup.find("body") or soup

        # Convert HTML to Markdown using markdownify
        markdown_content = markdownify(str(body), heading_style="ATX")

        logger.info(f"Successfully fetched and converted content from: {url}")

        return {
            "status": "success",
            "url": url,
            "content": markdown_content.strip()
        }
    except requests.HTTPError as e:
        error_msg = f"HTTP error occurred: {e.response.status_code}"
        logger.error(f"{error_msg} when fetching {url}")
        return {
            "status": "error",
            "url": url,
            "error": error_msg,
            "details": str(e)
        }
    except requests.RequestException as e:
        error_msg = f"Request error occurred: {str(e)}"
        logger.error(f"{error_msg} when fetching {url}")
        return {
            "status": "error",
            "url": url,
            "error": error_msg
        }
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        logger.error(f"{error_msg} when fetching {url}")
        return {
            "status": "error",
            "url": url,
            "error": error_msg
        }

async def fetch_url_content(url: str, timeout: float = DEFAULT_TIMEOUT, follow_redirects: bool = True) -> Dict[str, Any]:
    """
    Fetches content from a URL and converts it to markdown format.

    Args:
        url: The URL to fetch content from
        timeout: Timeout in seconds for the HTTP request
        follow_redirects: Whether to follow HTTP redirects

    Returns:
        A dictionary containing the status and either the markdown content or error message
    """
    try:
        async with httpx.AsyncClient(follow_redirects=follow_redirects, timeout=timeout) as client:
            logger.info(f"Fetching content from URL: {url}")
            response = await client.get(url)
            response.raise_for_status()

            # Convert HTML content to markdown
            markdown_content = markdownify(response.text)
            logger.info(f"Successfully fetched and converted content from: {url}")

            return {
                "status": "success",
                "url": url,
                "content": markdown_content
            }
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error occurred: {e.response.status_code}"
        logger.error(f"{error_msg} when fetching {url}")
        return {
            "status": "error",
            "url": url,
            "error": error_msg,
            "details": str(e)
        }
    except httpx.RequestError as e:
        error_msg = f"Request error occurred: {str(e)}"
        logger.error(f"{error_msg} when fetching {url}")
        return {
            "status": "error",
            "url": url,
            "error": error_msg
        }
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        logger.error(f"{error_msg} when fetching {url}")
        return {
            "status": "error",
            "url": url,
            "error": error_msg
        }

def get_url_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool to fetch content from a URL or multiple URLs and convert to markdown.

    Usage:
        This tool takes either a single URL or a list of URLs as input and returns
        the content of the page(s) in markdown format. It can be used to fetch
        documentation or other web content for analysis.

    Args:
        state: The current agent state containing tool calls

    Returns:
        Dictionary with messages containing tool results
    """
    outputs = []

    # Process each tool call
    for tool_call in state["messages"][-1].tool_calls:
        if tool_call["name"] == "get_url":
            try:
                args = tool_call["args"]
                urls = args.get("urls", args.get("url", ""))
                timeout = args.get("timeout", DEFAULT_TIMEOUT)
                follow_redirects = args.get("follow_redirects", True)

                # Handle either a single URL or a list of URLs
                if isinstance(urls, str):
                    if urls:
                        urls = [urls]
                    else:
                        urls = []

                logger.info(f"Executing get_url tool for {len(urls)} URL(s)")

                if not urls:
                    tool_result = {
                        "status": "error",
                        "error": "No URLs provided"
                    }
                else:
                    # Fetch content for each URL
                    results = []
                    for url in urls:
                        result = extract_markdown_content(url, timeout, follow_redirects)
                        results.append(result)

                    # Combine results
                    if len(results) == 1:
                        tool_result = results[0]
                    else:
                        tool_result = {
                            "status": "success",
                            "results": results
                        }

            except Exception as e:
                logger.exception(f"Error processing get_url tool: {e}")
                tool_result = {
                    "status": "error",
                    "error": f"Tool execution error: {str(e)}"
                }

            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name="get_url",
                    tool_call_id=tool_call["id"],
                )
            )

    return {"messages": outputs}
