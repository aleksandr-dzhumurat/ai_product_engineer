# --- AI Travel Assistant Agent with Google Places API Integration ---
# This agent uses the Google ADK to orchestrate the Gemini model, giving it
# the ability to call Google Places API via proper ADK function tools.

import asyncio
import os
import sys
from typing import Any, Dict

from dotenv import load_dotenv
from geocoding_api import GooglePlaceApi

print(load_dotenv())

from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools.function_tool import FunctionTool
from google.genai.types import Content, Part

# --- Configuration ---
AGENT_NAME = "GooglePlacesAgent"
MODEL_NAME = "models/gemini-2.0-flash-live-001"
APP_NAME = "TravelPlannerApp"

# Set up Google API authentication
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is required")

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

# --- Tool Creation ---

# Initialize GooglePlaceApi globally
places_api = GooglePlaceApi()

def get_coordinates(address: str) -> Dict[str, Any]:
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
            "error": f"Failed to geocode address: {str(e)}",
            "address": address,
            "status": "error"
        }

def search_places(query: str, location: str) -> Dict[str, Any]:
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
            "error": f"Failed to search places: {str(e)}",
            "query": query,
            "location": location,
            "status": "error"
        }

def create_google_places_tools():
    """
    Create ADK FunctionTool instances for Google Places API integration.
    """
    coordinates_tool = FunctionTool(
        func=get_coordinates
    )
    search_tool = FunctionTool(
        func=search_places
    )

    return [coordinates_tool, search_tool]

# ---  ---

def create_travel_agent():
    """Agent Initialization
    
    Create the travel agent with Google Places API tools
    """
    llm = Gemini(
        model_name=MODEL_NAME,
        api_key=api_key
    )
    tools = create_google_places_tools()
    agent = LlmAgent(
        name=AGENT_NAME,
        model=llm,
        instruction=TRAVEL_INSTRUCTION,
        tools=tools
    )
    return agent

async def main():
    """Main execution function"""

    print("--- AI Travel Assistant Agent with Google Places API ---")
    print(f"Agent: {AGENT_NAME} using {MODEL_NAME}")
    print("Tools: Google Places API (via GooglePlaceApi + FunctionTool)")
    print("-" * 60)

    try:
        agent = create_travel_agent()
        runner = InMemoryRunner(
            agent=agent,
            app_name=APP_NAME,
        )
        user_id = "test_user_001"
        session = await runner.session_service.create_session(
            app_name=APP_NAME, user_id=user_id
        )
        session_id = session.id

        print(f"[SYSTEM] Session created for User ID: {user_id}")
        print(f"[SYSTEM] Agent tools: {[tool.name for tool in agent.tools]}")
        print("-" * 60)
        user_query_text = "I'm planning a trip to Paphos, Cyprus. Can you help me find good restaurants and tourist attractions there? Please provide Google Maps links for each place."
        print(f"[USER] {user_query_text}")

        # Create content for agent
        user_content = Content(
            parts=[Part(text=user_query_text)],
            role="user"
        )

        print("\n[AGENT] Processing your request using Google Places tools...")
        print("-" * 60)
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_content
        ):
            if hasattr(event, 'content') and event.content:
                print(f"[AGENT RESPONSE] {event.content}")
            elif hasattr(event, 'text') and event.text:
                print(f"[AGENT RESPONSE] {event.text}")
            else:
                print(f"[AGENT EVENT] {event}")

    except Exception as e:
        print(f"[ERROR] Failed to run agent: {e}")
        import traceback
        traceback.print_exc()

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