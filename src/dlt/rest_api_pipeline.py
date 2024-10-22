from typing import Any, Optional

import dlt
from dlt.common.pendulum import pendulum
from dlt.sources.rest_api import (
    RESTAPIConfig,
    check_connection,
    rest_api_resources,
    rest_api_source,
)

import os
access_token = os.environ['GITHUB_API_TOKEN']

@dlt.source(name="github")
def github_source(access_token: Optional[str] = access_token) -> Any:
    # Create a REST API configuration for the GitHub API
    # Use RESTAPIConfig to get autocompletion and type checking
    config: RESTAPIConfig = {
        "client": {
            "base_url": "https://api.github.com",
            "auth": {
                "token": access_token, # <--- we already configured access_token above
            },
            "paginator": "header_link" # <---- set up paginator type
        },
        "resources": [  # <--- list resources
            {
                "name": "issues",
                "endpoint": {
                    "path": "repos/dlt-hub/dlt/issues",
                    "params": {
                        "state": "open",
                    },
                },
            },
            {
                "name": "issue_comments", # <-- here we declare dlt.transformer
                "endpoint": {
                    "path": "repos/dlt-hub/dlt/issues/{issue_number}/comments",
                    "params": {
                        "issue_number": {
                            "type": "resolve", # <--- use type 'resolve' to resolve {issue_number} for transformer
                            "resource": "issues",
                            "field": "number",
                        },

                    },
                },
            },
            {
                "name": "contributors",
                "endpoint": {
                    "path": "repos/dlt-hub/dlt/contributors",
                },
            },
        ],
    }

    yield from rest_api_resources(config)


def load_github() -> None:
    pipeline = dlt.pipeline(
        pipeline_name="rest_api_github",
        destination='duckdb',
        dataset_name="rest_api_data",
    )

    load_info = pipeline.run(github_source())
    print(load_info)  # noqa: T201


# def load_pokemon() -> None:
#     pipeline = dlt.pipeline(
#         pipeline_name="rest_api_pokemon",
#         destination='duckdb',
#         dataset_name="rest_api_data",
#     )

#     pokemon_source = rest_api_source(
#         {
#             "client": {
#                 "base_url": "https://pokeapi.co/api/v2/",
#                 # If you leave out the paginator, it will be inferred from the API:
#                 # "paginator": "json_link",
#             },
#             "resource_defaults": {
#                 "endpoint": {
#                     "params": {
#                         "limit": 1000,
#                     },
#                 },
#             },
#             "resources": [
#                 "pokemon",
#                 "berry",
#                 "location",
#             ],
#         }
#     )

#     def check_network_and_authentication() -> None:
#         (can_connect, error_msg) = check_connection(
#             pokemon_source,
#             "not_existing_endpoint",
#         )
#         if not can_connect:
#             pass  # do something with the error message

#     check_network_and_authentication()

#     load_info = pipeline.run(pokemon_source)
#     print(load_info)  # noqa: T201


if __name__ == "__main__":
    load_github()
    # load_pokemon()
