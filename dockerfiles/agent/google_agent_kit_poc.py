# --- AI Travel Assistant Agent with Google Places API Integration ---
# Rewritten with PydanticAI (https://ai.pydantic.dev/) running on Nebius
# Token Factory: https://docs.tokenfactory.nebius.com/integrations/agents/pydantic
#
# The agent talks to Nebius via the OpenAI-compatible API using NebiusProvider;
# NEBIUS_API_KEY and optional LLM_MODEL / LLM_BASE_URL overrides are read from
# the repo .env. Google Places access stays in plain Python tool functions,
# which PydanticAI exposes to the model automatically.

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load this agent's .env regardless of the current working directory, and fallback to repo root .env.
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from geocoding_api import GooglePlaceApi
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.agent import InstrumentationSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.nebius import NebiusProvider

# Initialize Langfuse tracing if credentials are present
langfuse_client = None
if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
    from langfuse import get_client

    langfuse_client = get_client()
    Agent.instrument_all(InstrumentationSettings(include_content=True, version=1))
    print("[SYSTEM] Langfuse tracing enabled.")

# --- Configuration ---
DEFAULT_BASE_URL = "https://api.studio.nebius.com/v1"
MODEL_NAME = (
    os.environ.get("LLM_MODEL")
    or os.environ.get("NEBIUS_MODEL")
    or "Qwen/Qwen3-30B-A3B-Instruct-2507"
)

api_key = os.environ.get("NEBIUS_API_KEY")
if not api_key:
    raise ValueError("NEBIUS_API_KEY environment variable is required")

TRAVEL_INSTRUCTION = """
You are a World-Class AI Travel Planner specializing in finding places, calculating routes,
and providing travel recommendations using Google Places and Maps services.

Your capabilities include:
- Finding coordinates for any location worldwide using the get_coordinates tool
- Searching for restaurants, hotels, attractions near specific locations using the search_places tool
- Providing detailed place information including addresses and clickable Google Maps links

Always use your tools to find the most accurate and up-to-date information.
When a user asks about places or locations, use the appropriate tools to get real data.
Always provide specific, actionable travel advice with maps links.
"""

# --- Tools (PydanticAI turns these plain functions into model tools) ---

# Initialize GooglePlaceApi globally
places_api = GooglePlaceApi()

def get_coordinates(address: str) -> dict[str, Any]:
    """
    Get coordinates for a specific address or location.

    Args:
        address: Address or location name to geocode (e.g., "Paris, France")

    Returns:
        Dictionary with coordinates and shareable Google Maps link
    """
    try:
        coords = places_api.find_place(address)
        lat, lng = coords.split(',')
        maps_link = places_api.shareble_link(lat, lng)

        return {
            "address": address,
            "coordinates": coords,
            "latitude": float(lat),
            "longitude": float(lng),
            "maps_link": maps_link,
            "status": "success"
        }

    except Exception as e:
        return {
            "error": f"Failed to geocode address: {e!s}",
            "address": address,
            "status": "error"
        }

def search_places(query: str, location: str) -> dict[str, Any]:
    """
    Search for places using Google Places API.

    Args:
        query: What to search for (e.g., "restaurants", "hotels", "tourist attractions")
        location: Location to search near (e.g., "Limassol, Cyprus")

    Returns:
        Dictionary with search results including names, addresses, and maps links
    """
    try:
        coords = places_api.find_place(location)
        print(f"[TOOL] Found coordinates for {location}: {coords}")
        results = places_api.api_request(query, coords)
        response = {
            "query": query,
            "location": location,
            "coordinates": coords,
            "results_count": len(results),
            "places": [],
            "status": "success"
        }
        for place in results[:5]:  # Limit to top 5 results
            place_info = {
                "name": place.get('name', 'Unknown'),
                "address": place.get('formatted_address', 'No address available'),
                "place_id": place.get('place_id', ''),
            }
            if place.get('geometry') and place['geometry'].get('location'):
                place_lat = place['geometry']['location']['lat']
                place_lng = place['geometry']['location']['lng']
                place_info["maps_link"] = places_api.shareble_link(place_lat, place_lng)
                place_info["coordinates"] = f"{place_lat},{place_lng}"
            elif coords:
                lat, lng = coords.split(',')
                place_info["maps_link"] = places_api.shareble_link(lat, lng)
                place_info["coordinates"] = coords
            response["places"].append(place_info)
        return response

    except Exception as e:
        return {
            "error": f"Failed to search places: {e!s}",
            "query": query,
            "location": location,
            "status": "error"
        }

# --- Agent ---

def create_travel_agent() -> Agent:
    """Create the travel agent with Google Places API tools on Nebius."""
    base_url = (
        os.environ.get("LLM_BASE_URL")
        or os.environ.get("NEBIUS_BASE_URL")
        or DEFAULT_BASE_URL
    )
    if base_url.rstrip("/") != DEFAULT_BASE_URL:
        # Non-default Nebius endpoint: bring our own OpenAI-compatible client
        provider = NebiusProvider(
            api_key=api_key,
            openai_client=AsyncOpenAI(base_url=base_url, api_key=api_key),
        )
    else:
        provider = NebiusProvider(api_key=api_key)

    model = OpenAIChatModel(MODEL_NAME, provider=provider)
    return Agent(
        model,
        instructions=TRAVEL_INSTRUCTION,
        tools=[get_coordinates, search_places],
    )

async def main():
    """Main execution function"""

    print("--- AI Travel Assistant Agent with Google Places API ---")
    print(f"Agent: TravelPlannerAgent using {MODEL_NAME} via Nebius (PydanticAI)")
    print("Tools: get_coordinates, search_places (Google Places API)")
    print("-" * 60)

    try:
        agent = create_travel_agent()
        user_query_text = "I'm planning a trip to Paphos, Cyprus. Can you help me find good restaurants and tourist attractions there? Please provide Google Maps links for each place."
        print(f"[USER] {user_query_text}")

        print("\n[AGENT] Processing your request using Google Places tools...")
        print("-" * 60)
        result = await agent.run(user_query_text)
        print(f"[AGENT RESPONSE] {result.output}")
        usage = result.usage()
        print(f"[SYSTEM] Token usage: requests={usage.requests} total={usage.total_tokens}")

    except Exception as e:
        print(f"[ERROR] Failed to run agent: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if langfuse_client:
            print("[SYSTEM] Flushing Langfuse traces...")
            langfuse_client.flush()

    print("\n" + "-" * 60)
    print("[SYSTEM] Agent run finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(
            f"\nAn error occurred during execution. Error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)
