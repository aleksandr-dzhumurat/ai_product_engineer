
I code project stop working, clarify with docs

[python package](https://github.com/google/adk-python)

[Google agents toolkit](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.mcp_tool)

## Agent examples

The directory contains two agent examples with different goals:

| | `dummy_agent.py` | `google_agent_kit_poc.py` |
|---|---|---|
| **Purpose** | Template-based conversational assistant for preparing an event. | Travel-planning proof of concept with live place lookups. |
| **Framework** | LangChain `ConversationChain` with in-memory conversation history. | PydanticAI `Agent` with Google Places tools. |
| **Interaction** | Interactive multi-turn dialogue; conversations are held in memory by `user_id`. | Runs one sample query about Paphos; it does not keep a conversation going. |
| **Tools / output** | Uses the LLM and `message_templates`; detects a final response by parsing JSON. | Uses `get_coordinates` and `search_places` to retrieve places, addresses, and Maps links. |
| **Model defaults** | `google/gemma-3-27b-it` via the Token Factory endpoint. | `Qwen/Qwen3-32B-fast` via the Nebius Studio endpoint. |

Both scripts load the repository `.env` and allow model and endpoint overrides through environment variables. The tools-powered example requires `GOOGLE_MAPS_API_KEY` for Google Places and Geocoding API access. Despite the Google ADK links above, `google_agent_kit_poc.py` currently uses PydanticAI; it does not import or use Google ADK.

```
uv venv --python 3.13;
source .venv/bin/activate;
uv pip install -r requirements.txt
```


```shell
uv run python dummy_agent.py
```

Run tools-powered agent

```shell
uv run python google_agent_kit_poc.py
```