from copy import deepcopy

import requests
import tiktoken
import trafilatura

SEARXNG_URL = "http://localhost:8080"

MAX_TOKENS = 4096
MAX_RESULTS = 5

from trafilatura.settings import DEFAULT_CONFIG

config = deepcopy(DEFAULT_CONFIG)
config["DEFAULT"]["DOWNLOAD_TIMEOUT"] = "10"

encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(encoding.encode(text))


def web_search(query: str) -> str:

    # 1. Search using SearXNG
    response = requests.get(
        f"{SEARXNG_URL}/search",
        params={
            "q": query,
            "format": "json",
            "sort_by": "date"
        },
        timeout=15
    )

    response.raise_for_status()

    search_results = response.json().get("results", [])

    total_tokens = 0

    search_results_text = ""
    # 2. Extract content from search results
    for idx, result in enumerate(search_results[:MAX_RESULTS]):

        url = result.get("url")
        snippet = result.get("content", "")
        published_date = result.get("publishedDate")

        if not url:
            continue

        try:
            # Download webpage
            downloaded = trafilatura.fetch_url(url, config=config)

            if not downloaded:
                continue

            # Extract readable content
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True
            )

            if not text:
                continue

            content = url + "\n" + snippet + "\n" + ("Published Date: " + (published_date or "")) + "\n" + text

            # 3. Enforce token limit
            remaining_tokens = MAX_TOKENS - total_tokens

            if remaining_tokens <= 0:
                break

            content_tokens = encoding.encode(content)

            if len(content_tokens) > remaining_tokens:
                content = encoding.decode(
                    content_tokens[:remaining_tokens]
                )

            tokens_used = count_tokens(content)

            total_tokens += tokens_used

            # 4. Append the output
            search_results_text += content + "\n"
        except (requests.exceptions.RequestException, TimeoutError) as e:
            print(f"Fetching error for {url}: {e}")
            continue

    return search_results_text


web_search_tool = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and retrieve relevant webpage content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    }

if __name__ == "__main__":
    query = "What's today's NYC weather?"
    results = web_search(query)
    for result in results:
        print(result)