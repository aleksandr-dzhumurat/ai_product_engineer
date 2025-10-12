import os
import typing as t
from urllib.parse import quote

import requests


class GooglePlaceApi:
    def __init__(self):
        self.API_TYPE = 'textsearch' # findplacefromtext
        self.url = f"https://maps.googleapis.com/maps/api/place/{self.API_TYPE}/json"
        self.google_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        if not self.google_api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY environment variable is required")
        self.cache = None

    def shareble_link(self, lat, lng):
        # maps_url = f"https://www.google.com/maps/place/?q=place_id:{place_id}"
        location_param = quote(f'{lat},{lng}')
        maps_url =  f"https://www.google.com/maps?q={location_param}"
        return maps_url

    def shareble_link_pretty(self, place_id):
        """ATTENTION: do NOT use to avoid additional payments, x2"""
        fields = 'name,url'

        params = {
            "place_id": place_id,
            "fields": fields,
            "key": self.google_api_key
        }
        url = "https://maps.googleapis.com/maps/api/place/details/json"
        response = requests.get(url, params=params, headers={}).json()
        response = response['result']['url']

        return response

    def api_request(self, q: str, lat_lng: t.Optional[str]):
        if lat_lng is None:
            lat, lng = 34.707130,33.022617  # Lemesos city centre
        else:
            lat, lng = lat_lng.split(',')
        radius = 5000
        params = {
            "inputtype": "textquery",
            "fields": "formatted_address,name,place_id,geometry",
            "key": self.google_api_key
        }
        if self.API_TYPE == 'findplacefromtext':
            location = f'circle:{radius}@{lat},{lng}'
            params.update({"locationbias": location, "input": q})
        elif self.API_TYPE == 'textsearch':
            location_param = f'{lat},{lng}'
            params.update({"location": location_param, 'radius': radius, "query": q})

        response = requests.get(self.url, params=params, headers={}).json()
        if self.API_TYPE == 'findplacefromtext':
            response = response['candidates']
        else:
            response = response['results']
        return response

    def find_place(self, place_name: str):
        """place_name Serbia, Belgrade"""
        params = {
            "address": place_name,
            "key": self.google_api_key
        }
        url = 'https://maps.googleapis.com/maps/api/geocode/json'
        response = requests.get(url, params=params)
        data = response.json()

        if data['status'] == 'OK':
            location = data['results'][0]['geometry']['location']
            latitude = location['lat']
            longitude = location['lng']
        else:
            print(data)
            print('Geocoding failed')
        return f'{latitude},{longitude}'


def main():
    """Test the GooglePlaceApi functionality"""
    try:
        api = GooglePlaceApi()
        print("Testing GooglePlaceApi...")

        # Test 1: Find coordinates for a city
        print("\n1. Testing find_place() - Geocoding")
        place = "Paris, France"
        coords = api.find_place(place)
        print(f"Coordinates for {place}: {coords}")

        # Test 2: Generate shareable link
        print("\n2. Testing shareble_link()")
        lat, lng = coords.split(',')
        share_link = api.shareble_link(lat, lng)
        print(f"Shareable link: {share_link}")

        # Test 3: Search for restaurants near the location
        print("\n3. Testing api_request() - Place Search")
        restaurants = api.api_request("restaurants", coords)
        print(f"Found {len(restaurants)} restaurants near {place}:")
        for i, restaurant in enumerate(restaurants[:3], 1):
            name = restaurant.get('name', 'N/A')
            address = restaurant.get('formatted_address', 'N/A')
            print(f"  {i}. {name} - {address}")

        # Test 4: Search for tourist attractions
        print("\n4. Testing tourist attractions search")
        attractions = api.api_request("tourist attractions", coords)
        print(f"Found {len(attractions)} attractions:")
        for i, attraction in enumerate(attractions[:3], 1):
            name = attraction.get('name', 'N/A')
            address = attraction.get('formatted_address', 'N/A')
            print(f"  {i}. {name} - {address}")

    except Exception as e:
        print(f"Error during testing: {e}")
        print("Make sure GOOGLE_MAPS_API_KEY is set in your environment")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()